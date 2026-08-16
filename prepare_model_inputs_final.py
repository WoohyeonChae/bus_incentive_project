from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import re
import sqlite3
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# 기본 컬럼명
# -----------------------------------------------------------------------------

OD_DATE_COL = "기준_날짜"
OD_ORIGIN_COL = "승차_정류장/역사_ID"
OD_DEST_COL = "하차_정류장/역사_ID"
OD_TOTAL_COL = "승객_수"
OD_ZIP_PATTERN = re.compile(r"^kscc_dx_ra_od_(\d{8}|\d{6})\.zip$", re.IGNORECASE)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "서울 버스 API 6종의 정리 결과와 OA-21222 OD ZIP을 결합하여 "
            "현상금 경로모형용 od_hourly.csv, segments.csv, stops.csv를 생성합니다."
        )
    )
    parser.add_argument("--data-dir", default="data", help="수집 결과의 루트 data 폴더")
    parser.add_argument(
        "--output-dir",
        default="data/model_input",
        help="최종 모델 입력 저장 폴더",
    )
    parser.add_argument(
        "--od-raw-dir",
        default="data/od/raw",
        help="OA-21222 OD ZIP 파일 폴더",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifest.json",
        help="download_all_bus_data.py가 만든 manifest.json",
    )
    parser.add_argument(
        "--dates",
        help="manifest 대신 사용할 날짜 목록(쉼표 구분 YYYYMMDD)",
    )
    parser.add_argument(
        "--hours",
        help="manifest 대신 사용할 시간대 목록(쉼표 구분, 예: 7,8,9,17,18,19)",
    )
    parser.add_argument(
        "--travel-time-unit",
        choices=("auto", "seconds", "minutes"),
        default="auto",
        help=(
            "AVG_OPERATION_TIME의 단위. auto는 양수 중앙값이 15보다 크면 "
            "초, 아니면 분으로 판단합니다."
        ),
    )
    parser.add_argument(
        "--min-od-passengers",
        type=float,
        default=0.0,
        help="od_hourly.csv에 남길 날짜·시간·OD 최소 승객수",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="OD ZIP 내부 CSV를 읽는 청크 크기",
    )
    parser.add_argument(
        "--keep-od-outside-network",
        action="store_true",
        help="버스 그래프에 존재하지 않는 OD도 od_hourly.csv에 유지",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="OD 임시 DB와 기존 최종 결과를 지우고 다시 생성",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# 공통 유틸리티
# -----------------------------------------------------------------------------


def normalize_id_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_date_series(series: pd.Series) -> pd.Series:
    return (
        normalize_id_series(series)
        .str.replace("-", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.slice(0, 8)
        .str.zfill(8)
    )


def parse_int_list(value: str, minimum: int, maximum: int) -> List[int]:
    result = sorted({int(token.strip()) for token in value.split(",") if token.strip()})
    if not result or any(number < minimum or number > maximum for number in result):
        raise ValueError(f"값은 {minimum}~{maximum} 범위여야 합니다: {value}")
    return result


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as error:
            last_error = error
    raise RuntimeError(f"CSV 인코딩을 판별하지 못했습니다: {path}: {last_error}")


def resolve_column(
    columns: Iterable[str],
    aliases: Sequence[str],
    label: str,
    required: bool = True,
) -> Optional[str]:
    columns_list = list(columns)
    exact = {str(column).upper(): str(column) for column in columns_list}
    for alias in aliases:
        if alias.upper() in exact:
            return exact[alias.upper()]

    normalized = {
        re.sub(r"[^A-Z0-9가-힣]", "", str(column).upper()): str(column)
        for column in columns_list
    }
    for alias in aliases:
        key = re.sub(r"[^A-Z0-9가-힣]", "", alias.upper())
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"'{label}' 컬럼을 찾지 못했습니다. 후보={list(aliases)}, "
            f"실제={columns_list}"
        )
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def haversine_meters(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    # pandas/표준 라이브러리만 사용하기 위한 벡터화 구현
    import numpy as np

    lat1r = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1r = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2r = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2r = np.radians(pd.to_numeric(lon2, errors="coerce"))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return pd.Series(2.0 * 6_371_000.0 * np.arcsin(np.sqrt(a)), index=lat1.index)


def load_manifest(path: Path, args: argparse.Namespace) -> Tuple[List[str], List[int]]:
    payload: Dict[str, object] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if args.dates:
        dates = sorted({token.strip() for token in args.dates.split(",") if token.strip()})
    else:
        dates = sorted({str(value).replace("-", "") for value in payload.get("target_dates", [])})

    if args.hours:
        hours = parse_int_list(args.hours, 0, 23)
    else:
        raw_hours = payload.get("hours", ["07", "08", "09", "17", "18", "19"])
        hours = sorted({int(value) for value in raw_hours})

    if not dates:
        raise RuntimeError(
            "대상 날짜를 찾지 못했습니다. data/manifest.json을 확인하거나 --dates를 지정하세요."
        )
    invalid = [value for value in dates if not re.fullmatch(r"\d{8}", value)]
    if invalid:
        raise ValueError(f"잘못된 날짜 형식: {invalid}")
    return dates, hours


def clear_output(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir() and path.name == "_work":
            import shutil
            shutil.rmtree(path)


# -----------------------------------------------------------------------------
# 마스터 테이블
# -----------------------------------------------------------------------------


def build_stops(stop_master_path: Path) -> pd.DataFrame:
    raw = read_csv_auto(stop_master_path, dtype=str, low_memory=False)

    stop_id_col = resolve_column(raw.columns, ("CRTR_ID", "STOP_ID", "STOPS_ID"), "정류장 ID")
    stop_name_col = resolve_column(raw.columns, ("CRTR_NM", "STOP_NM", "STOP_NAME"), "정류장명", False)
    stop_type_col = resolve_column(raw.columns, ("CRTR_TYPE", "STOP_TYPE"), "정류장 유형", False)
    stop_no_col = resolve_column(raw.columns, ("CRTR_NO", "STOP_NO", "ARS_ID"), "정류장 번호", False)
    lat_col = resolve_column(raw.columns, ("LAT", "LATITUDE", "Y"), "위도", False)
    lon_col = resolve_column(raw.columns, ("LOT", "LON", "LNG", "LONGITUDE", "X"), "경도", False)

    stops = pd.DataFrame({"stop_id": normalize_id_series(raw[stop_id_col])})
    stops["stop_name"] = raw[stop_name_col].astype("string") if stop_name_col else pd.NA
    stops["stop_type"] = raw[stop_type_col].astype("string") if stop_type_col else pd.NA
    stops["stop_number"] = normalize_id_series(raw[stop_no_col]) if stop_no_col else pd.NA
    stops["latitude"] = safe_numeric(raw[lat_col]) if lat_col else pd.NA
    stops["longitude"] = safe_numeric(raw[lon_col]) if lon_col else pd.NA

    stops = stops.dropna(subset=["stop_id"]).drop_duplicates("stop_id", keep="first")
    return stops


def build_routes(route_master_path: Path) -> pd.DataFrame:
    if not route_master_path.exists():
        return pd.DataFrame(columns=["route_id", "route_name", "route_type", "route_distance"])

    raw = read_csv_auto(route_master_path, dtype=str, low_memory=False)
    route_id_col = resolve_column(raw.columns, ("RTE_ID", "ROUTE_ID", "routeId"), "노선 ID")
    route_name_col = resolve_column(raw.columns, ("RTE_NM", "ROUTE_NM", "routeNm"), "노선명", False)
    route_type_col = resolve_column(raw.columns, ("RTE_TYPE", "ROUTE_TYPE", "RTE_TY", "routeTy"), "노선 유형", False)
    distance_col = resolve_column(raw.columns, ("DSTNC", "DISTANCE", "RTE_DSTNC", "routeDistance"), "노선 거리", False)

    routes = pd.DataFrame({"route_id": normalize_id_series(raw[route_id_col])})
    routes["route_name"] = raw[route_name_col].astype("string") if route_name_col else pd.NA
    routes["route_type"] = raw[route_type_col].astype("string") if route_type_col else pd.NA
    routes["route_distance"] = safe_numeric(raw[distance_col]) if distance_col else pd.NA
    return routes.dropna(subset=["route_id"]).drop_duplicates("route_id", keep="first")


def build_route_stop_edges(route_stop_master_path: Path) -> pd.DataFrame:
    if not route_stop_master_path.exists():
        return pd.DataFrame(
            columns=["route_id", "from_stop_id", "to_stop_id", "master_stop_sequence", "link_distance_m"]
        )

    raw = read_csv_auto(route_stop_master_path, dtype=str, low_memory=False)
    route_col = resolve_column(raw.columns, ("RTE_ID", "ROUTE_ID"), "노선 ID")
    stop_col = resolve_column(raw.columns, ("CRTR_ID", "STOP_ID", "STOPS_ID"), "정류장 ID")
    seq_col = resolve_column(raw.columns, ("CRTR_SEQ", "STOPS_SEQ", "STOP_SEQ", "STA_SN"), "정류장 순서")
    distance_col = resolve_column(raw.columns, ("LNKG_LEN", "LINK_LEN", "LINK_DISTANCE", "DSTNC"), "링크 거리", False)

    frame = pd.DataFrame(
        {
            "route_id": normalize_id_series(raw[route_col]),
            "from_stop_id": normalize_id_series(raw[stop_col]),
            "master_stop_sequence": safe_numeric(raw[seq_col]),
            "link_distance_m": safe_numeric(raw[distance_col]) if distance_col else pd.NA,
        }
    ).dropna(subset=["route_id", "from_stop_id", "master_stop_sequence"])

    frame = frame.sort_values(["route_id", "master_stop_sequence"], kind="stable")
    frame["to_stop_id"] = frame.groupby("route_id", sort=False)["from_stop_id"].shift(-1)
    frame = frame.dropna(subset=["to_stop_id"])
    return frame.drop_duplicates(
        ["route_id", "from_stop_id", "to_stop_id", "master_stop_sequence"],
        keep="first",
    )


# -----------------------------------------------------------------------------
# 버스 구간 모델 입력
# -----------------------------------------------------------------------------


def infer_time_unit(values: pd.Series, requested: str) -> Tuple[str, float]:
    positive = safe_numeric(values)
    positive = positive[positive > 0]
    median = float(positive.median()) if not positive.empty else math.nan

    if requested == "seconds":
        return "seconds", median
    if requested == "minutes":
        return "minutes", median

    # 버스 정류장 간 평균시간은 보통 수분 이내다. 원시 중앙값이 15보다 크면 초로 판단.
    inferred = "seconds" if not math.isnan(median) and median > 15 else "minutes"
    return inferred, median


def build_segments(
    model_segments_path: Path,
    target_dates: Set[str],
    target_hours: Set[int],
    stops: pd.DataFrame,
    routes: pd.DataFrame,
    route_stop_edges: pd.DataFrame,
    travel_time_unit: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    raw = read_csv_auto(model_segments_path, dtype=str, low_memory=False)

    required = [
        "CRTR_DD",
        "RTE_ID",
        "DPTRE_STOPS_ID",
        "ARVL_STOPS_ID",
        "HOUR",
        "PASSENGERS",
        "AVG_OPERATION_TIME",
        "BUS_OPERATIONS",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(f"model_segments.csv 필수 컬럼 누락: {missing}")

    frame = pd.DataFrame(
        {
            "date": normalize_date_series(raw["CRTR_DD"]),
            "hour": safe_numeric(raw["HOUR"]),
            "route_id": normalize_id_series(raw["RTE_ID"]),
            "from_stop_id": normalize_id_series(raw["DPTRE_STOPS_ID"]),
            "to_stop_id": normalize_id_series(raw["ARVL_STOPS_ID"]),
            "stop_sequence": safe_numeric(raw["STOPS_SEQ"]) if "STOPS_SEQ" in raw.columns else pd.NA,
            "section_passengers": safe_numeric(raw["PASSENGERS"]),
            "travel_time_raw": safe_numeric(raw["AVG_OPERATION_TIME"]),
            "trips": safe_numeric(raw["BUS_OPERATIONS"]),
        }
    )

    frame = frame[
        frame["date"].isin(target_dates)
        & frame["hour"].isin(target_hours)
    ].copy()
    frame["hour"] = frame["hour"].astype("int16")

    frame = frame.dropna(
        subset=[
            "date",
            "hour",
            "route_id",
            "from_stop_id",
            "to_stop_id",
            "section_passengers",
            "travel_time_raw",
            "trips",
        ]
    )
    frame = frame[(frame["section_passengers"] >= 0) & (frame["travel_time_raw"] > 0) & (frame["trips"] > 0)]

    key = ["date", "hour", "route_id", "from_stop_id", "to_stop_id"]
    duplicate_rows = int(frame.duplicated(key, keep=False).sum())

    # model.py의 SEG_KEY와 동일하게 날짜·시간·노선·물리구간당 한 행으로 집계
    frame = (
        frame.groupby(key, as_index=False, dropna=False)
        .agg(
            stop_sequence=("stop_sequence", "min"),
            section_passengers=("section_passengers", "sum"),
            travel_time_raw=("travel_time_raw", "mean"),
            trips=("trips", "max"),
        )
    )
    frame["avg_onboard"] = frame["section_passengers"] / frame["trips"]

    inferred_unit, time_median = infer_time_unit(frame["travel_time_raw"], travel_time_unit)
    if inferred_unit == "seconds":
        frame["travel_time"] = frame["travel_time_raw"] / 60.0
    else:
        frame["travel_time"] = frame["travel_time_raw"]

    # 노선 정류장마스터에서 거리 결합
    if not route_stop_edges.empty:
        edge_lookup = route_stop_edges[
            ["route_id", "from_stop_id", "to_stop_id", "master_stop_sequence", "link_distance_m"]
        ].copy()
        edge_lookup = edge_lookup.drop_duplicates(
            ["route_id", "from_stop_id", "to_stop_id"], keep="first"
        )
        frame = frame.merge(
            edge_lookup,
            on=["route_id", "from_stop_id", "to_stop_id"],
            how="left",
            validate="many_to_one",
        )
    else:
        frame["master_stop_sequence"] = pd.NA
        frame["link_distance_m"] = pd.NA

    # 정류장명과 좌표 결합
    stop_meta = stops[["stop_id", "stop_name", "latitude", "longitude"]].copy()
    from_meta = stop_meta.rename(
        columns={
            "stop_id": "from_stop_id",
            "stop_name": "from_stop_name",
            "latitude": "from_latitude",
            "longitude": "from_longitude",
        }
    )
    to_meta = stop_meta.rename(
        columns={
            "stop_id": "to_stop_id",
            "stop_name": "to_stop_name",
            "latitude": "to_latitude",
            "longitude": "to_longitude",
        }
    )
    frame = frame.merge(from_meta, on="from_stop_id", how="left", validate="many_to_one")
    frame = frame.merge(to_meta, on="to_stop_id", how="left", validate="many_to_one")

    straight = haversine_meters(
        frame["from_latitude"],
        frame["from_longitude"],
        frame["to_latitude"],
        frame["to_longitude"],
    )
    frame["distance_source"] = "route_stop_master"
    missing_distance = frame["link_distance_m"].isna() | (frame["link_distance_m"] <= 0)
    frame.loc[missing_distance, "link_distance_m"] = straight[missing_distance]
    frame.loc[missing_distance, "distance_source"] = "haversine"

    still_missing = frame["link_distance_m"].isna() | (frame["link_distance_m"] <= 0)
    dropped_missing_distance = int(still_missing.sum())
    if dropped_missing_distance:
        frame = frame.loc[~still_missing].copy()
    frame["distance"] = frame["link_distance_m"]

    if not routes.empty:
        frame = frame.merge(routes, on="route_id", how="left", validate="many_to_one")

    output_columns = [
        "date",
        "hour",
        "route_id",
        "from_stop_id",
        "to_stop_id",
        "travel_time",
        "distance",
        "avg_onboard",
        "trips",
        "section_passengers",
        "travel_time_raw",
        "stop_sequence",
        "master_stop_sequence",
        "distance_source",
        "route_name",
        "route_type",
        "from_stop_name",
        "to_stop_name",
        "from_latitude",
        "from_longitude",
        "to_latitude",
        "to_longitude",
    ]
    for column in output_columns:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[output_columns].sort_values(
        ["date", "hour", "route_id", "stop_sequence", "from_stop_id", "to_stop_id"],
        kind="stable",
    )

    qa = {
        "raw_rows": int(len(raw)),
        "output_rows": int(len(frame)),
        "duplicate_rows_before_grouping": duplicate_rows,
        "available_dates": sorted(frame["date"].dropna().astype(str).unique().tolist()),
        "travel_time_unit": inferred_unit,
        "travel_time_raw_median": time_median,
        "distance_source_counts": {
            str(key): int(value)
            for key, value in frame["distance_source"].value_counts(dropna=False).items()
        },
        "dropped_missing_distance_rows": dropped_missing_distance,
        "missing_from_stop_metadata": int(frame["from_stop_name"].isna().sum()),
        "missing_to_stop_metadata": int(frame["to_stop_name"].isna().sum()),
    }
    return frame, qa


# -----------------------------------------------------------------------------
# OA-21222 버스 OD 원본 ZIP 처리
# -----------------------------------------------------------------------------


def discover_od_zip_assignments(
    raw_dir: Path,
    target_dates: Sequence[str],
) -> Tuple[Dict[Path, Set[str]], List[str]]:
    daily: Dict[str, Path] = {}
    monthly: Dict[str, Path] = {}

    for path in sorted(raw_dir.glob("*.zip")):
        match = OD_ZIP_PATTERN.match(path.name)
        if not match:
            continue
        token = match.group(1)
        if len(token) == 8:
            daily[token] = path
        else:
            monthly[token] = path

    assignments: Dict[Path, Set[str]] = {}
    missing: List[str] = []
    for current_date in target_dates:
        selected: Optional[Path] = None
        if current_date in daily:
            selected = daily[current_date]
        elif current_date[:6] in monthly:
            selected = monthly[current_date[:6]]

        if selected is None:
            missing.append(current_date)
        else:
            assignments.setdefault(selected, set()).add(current_date)

    return assignments, missing


def extract_csv_date(member: str) -> Optional[str]:
    match = re.search(r"(\d{8})(?=\.csv$)", Path(member).name, re.IGNORECASE)
    return match.group(1) if match else None


def detect_zip_csv_encoding(archive: zipfile.ZipFile, member: str) -> str:
    for encoding in ("cp949", "utf-8-sig", "euc-kr", "utf-8"):
        try:
            with archive.open(member) as binary:
                with io.TextIOWrapper(binary, encoding=encoding, newline="") as text:
                    pd.read_csv(text, nrows=0)
            return encoding
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise RuntimeError(f"{member}의 인코딩을 판별하지 못했습니다.")


def initialize_od_db(db_path: Path, fresh: bool) -> sqlite3.Connection:
    if fresh and db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS daily_od (
            date TEXT NOT NULL,
            hour INTEGER NOT NULL,
            origin_stop_id TEXT NOT NULL,
            destination_stop_id TEXT NOT NULL,
            passengers REAL NOT NULL,
            PRIMARY KEY(date, hour, origin_stop_id, destination_stop_id)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS processed_sources (
            source_key TEXT PRIMARY KEY,
            processed_at TEXT NOT NULL
        );
        """
    )
    connection.commit()
    return connection


def process_od_zip(
    zip_path: Path,
    selected_dates: Set[str],
    hours: Sequence[int],
    connection: sqlite3.Connection,
    chunk_size: int,
) -> Dict[str, int]:
    stat = zip_path.stat()
    source_key = "|".join(
        [
            str(zip_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            ",".join(sorted(selected_dates)),
        ]
    )

    if connection.execute(
        "SELECT 1 FROM processed_sources WHERE source_key=?",
        (source_key,),
    ).fetchone():
        print(f"[OD SKIP] {zip_path.name}: 이미 처리됨")
        return {
            "source_rows": 0,
            "bus_rows": 0,
            "same_stop_rows_removed": 0,
            "stored_rows": 0,
        }

    hour_columns = [f"승객_수_{hour:02d}시" for hour in hours]
    required_columns = [
        OD_DATE_COL,
        OD_ORIGIN_COL,
        OD_DEST_COL,
        OD_TOTAL_COL,
        *hour_columns,
    ]
    column_to_hour = {
        f"승객_수_{hour:02d}시": hour
        for hour in hours
    }
    metrics = {
        "source_rows": 0,
        "bus_rows": 0,
        "same_stop_rows_removed": 0,
        "stored_rows": 0,
    }

    print(f"[OD PROCESS] {zip_path.name}: {', '.join(sorted(selected_dates))}")

    # ZIP 하나를 하나의 트랜잭션으로 처리한다.
    # 중간에 오류/중단이 발생하면 해당 ZIP에서 추가한 내용을 전부 롤백하여
    # 재실행 시 승객수가 중복 합산되지 않도록 한다.
    connection.execute("BEGIN IMMEDIATE")

    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [
                member
                for member in archive.namelist()
                if member.lower().endswith(".csv")
                and not member.startswith("__MACOSX/")
            ]
            selected_members = [
                (member, extract_csv_date(member))
                for member in members
                if extract_csv_date(member) in selected_dates
            ]
            found_dates = {
                current_date
                for _, current_date in selected_members
                if current_date
            }
            missing = sorted(selected_dates - found_dates)
            if missing:
                raise RuntimeError(
                    f"{zip_path.name} 내부에 날짜 CSV가 없습니다: {missing}"
                )

            for member, current_date in sorted(
                selected_members,
                key=lambda item: str(item[1]),
            ):
                encoding = detect_zip_csv_encoding(archive, member)
                print(f"  {member} ({encoding})")

                with archive.open(member) as binary:
                    with io.TextIOWrapper(
                        binary,
                        encoding=encoding,
                        newline="",
                    ) as text:
                        header = pd.read_csv(text, nrows=0)

                missing_columns = [
                    column
                    for column in required_columns
                    if column not in header.columns
                ]
                if missing_columns:
                    raise RuntimeError(
                        f"{member} 필수 컬럼 누락: {missing_columns}"
                    )

                with archive.open(member) as binary:
                    with io.TextIOWrapper(
                        binary,
                        encoding=encoding,
                        newline="",
                    ) as text:
                        reader = pd.read_csv(
                            text,
                            usecols=required_columns,
                            dtype={
                                OD_DATE_COL: "string",
                                OD_ORIGIN_COL: "string",
                                OD_DEST_COL: "string",
                            },
                            chunksize=chunk_size,
                            low_memory=False,
                        )

                        for chunk in reader:
                            metrics["source_rows"] += len(chunk)

                            chunk[OD_DATE_COL] = normalize_date_series(
                                chunk[OD_DATE_COL]
                            )
                            chunk = chunk[
                                chunk[OD_DATE_COL].isin(selected_dates)
                            ].copy()
                            if chunk.empty:
                                continue

                            chunk[OD_ORIGIN_COL] = normalize_id_series(
                                chunk[OD_ORIGIN_COL]
                            )
                            chunk[OD_DEST_COL] = normalize_id_series(
                                chunk[OD_DEST_COL]
                            )

                            # 버스 정류장 OD: 출발/도착 모두 9자리 정류장 ID.
                            chunk = chunk[
                                chunk[OD_ORIGIN_COL].str.fullmatch(
                                    r"\d{9}",
                                    na=False,
                                )
                                & chunk[OD_DEST_COL].str.fullmatch(
                                    r"\d{9}",
                                    na=False,
                                )
                            ].copy()
                            if chunk.empty:
                                continue

                            metrics["bus_rows"] += len(chunk)

                            # 같은 정류장 출발-도착은 경로추천 OD에서 제외한다.
                            same_stop = (
                                chunk[OD_ORIGIN_COL]
                                == chunk[OD_DEST_COL]
                            )
                            metrics["same_stop_rows_removed"] += int(
                                same_stop.sum()
                            )
                            if same_stop.any():
                                chunk = chunk.loc[~same_stop].copy()
                            if chunk.empty:
                                continue

                            for column in hour_columns:
                                chunk[column] = pd.to_numeric(
                                    chunk[column],
                                    errors="coerce",
                                ).fillna(0)

                            long_chunk = chunk.melt(
                                id_vars=[
                                    OD_DATE_COL,
                                    OD_ORIGIN_COL,
                                    OD_DEST_COL,
                                ],
                                value_vars=hour_columns,
                                var_name="hour_column",
                                value_name="passengers",
                            )
                            long_chunk["hour"] = (
                                long_chunk["hour_column"]
                                .map(column_to_hour)
                                .astype(int)
                            )
                            long_chunk = long_chunk[
                                long_chunk["passengers"] > 0
                            ].copy()
                            if long_chunk.empty:
                                continue

                            grouped = (
                                long_chunk.groupby(
                                    [
                                        OD_DATE_COL,
                                        "hour",
                                        OD_ORIGIN_COL,
                                        OD_DEST_COL,
                                    ],
                                    as_index=False,
                                    sort=False,
                                )["passengers"]
                                .sum()
                            )

                            records = [
                                (
                                    str(row[0]),
                                    int(row[1]),
                                    str(row[2]),
                                    str(row[3]),
                                    float(row[4]),
                                )
                                for row in grouped.itertuples(
                                    index=False,
                                    name=None,
                                )
                            ]

                            connection.executemany(
                                """
                                INSERT INTO daily_od(
                                    date,
                                    hour,
                                    origin_stop_id,
                                    destination_stop_id,
                                    passengers
                                )
                                VALUES (?, ?, ?, ?, ?)
                                ON CONFLICT(
                                    date,
                                    hour,
                                    origin_stop_id,
                                    destination_stop_id
                                )
                                DO UPDATE SET
                                    passengers = passengers + excluded.passengers
                                """,
                                records,
                            )
                            metrics["stored_rows"] += len(records)

        connection.execute(
            """
            INSERT INTO processed_sources(source_key, processed_at)
            VALUES (?, ?)
            """,
            (
                source_key,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        connection.commit()

    except BaseException:
        connection.rollback()
        raise

    return metrics

def build_od_database(
    od_raw_dir: Path,
    target_dates: Sequence[str],
    hours: Sequence[int],
    db_path: Path,
    chunk_size: int,
    fresh: bool,
) -> Tuple[sqlite3.Connection, Dict[str, object]]:
    assignments, missing_zip_dates = discover_od_zip_assignments(od_raw_dir, target_dates)
    if not assignments:
        raise RuntimeError(f"OD ZIP을 찾지 못했습니다: {od_raw_dir}")

    connection = initialize_od_db(db_path, fresh)
    source_metrics: List[Dict[str, object]] = []
    for zip_path, selected_dates in sorted(assignments.items(), key=lambda item: item[0].name):
        metrics = process_od_zip(
            zip_path,
            selected_dates,
            hours,
            connection,
            chunk_size,
        )
        source_metrics.append({"zip": zip_path.name, "dates": sorted(selected_dates), **metrics})

    available_dates = [
        str(row[0])
        for row in connection.execute("SELECT DISTINCT date FROM daily_od ORDER BY date").fetchall()
    ]
    return connection, {
        "missing_zip_dates": missing_zip_dates,
        "available_dates": available_dates,
        "sources": source_metrics,
    }


def create_network_stop_table(connection: sqlite3.Connection, stop_ids: Iterable[str]) -> None:
    connection.execute("DROP TABLE IF EXISTS network_stops")
    connection.execute("CREATE TEMP TABLE network_stops(stop_id TEXT PRIMARY KEY)")
    connection.executemany(
        "INSERT OR IGNORE INTO network_stops(stop_id) VALUES (?)",
        [(str(stop_id),) for stop_id in set(stop_ids)],
    )
    connection.commit()


def export_od_hourly(
    connection: sqlite3.Connection,
    output_path: Path,
    used_dates: Sequence[str],
    min_passengers: float,
    network_only: bool,
) -> int:
    placeholders = ",".join("?" for _ in used_dates)
    joins = ""
    where_network = ""
    if network_only:
        joins = (
            " JOIN network_stops o ON d.origin_stop_id=o.stop_id "
            " JOIN network_stops z ON d.destination_stop_id=z.stop_id "
        )
    query = f"""
        SELECT
            d.date AS date,
            d.hour AS hour,
            d.origin_stop_id AS origin_stop_id,
            d.destination_stop_id AS destination_stop_id,
            d.passengers AS passengers
        FROM daily_od d
        {joins}
        WHERE d.date IN ({placeholders})
          AND d.passengers >= ?
        ORDER BY d.date, d.hour, d.origin_stop_id, d.destination_stop_id
    """
    params: Tuple[object, ...] = tuple(used_dates) + (float(min_passengers),)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    first = True
    total = 0
    for chunk in pd.read_sql_query(query, connection, params=params, chunksize=200_000):
        chunk.to_csv(output_path, mode="w" if first else "a", header=first, index=False, encoding="utf-8-sig")
        first = False
        total += len(chunk)
    if first:
        pd.DataFrame(columns=["date", "hour", "origin_stop_id", "destination_stop_id", "passengers"]).to_csv(
            output_path, index=False, encoding="utf-8-sig"
        )
    return total


def export_od_baseline(
    connection: sqlite3.Connection,
    output_path: Path,
    used_dates: Sequence[str],
    network_only: bool,
) -> int:
    placeholders = ",".join("?" for _ in used_dates)
    joins = ""
    if network_only:
        joins = (
            " JOIN network_stops o ON d.origin_stop_id=o.stop_id "
            " JOIN network_stops z ON d.destination_stop_id=z.stop_id "
        )
    query = f"""
        SELECT
            d.origin_stop_id,
            d.destination_stop_id,
            d.hour,
            SUM(d.passengers) AS total_passengers,
            SUM(d.passengers * d.passengers) AS sum_squared,
            COUNT(*) AS positive_days
        FROM daily_od d
        {joins}
        WHERE d.date IN ({placeholders})
        GROUP BY d.origin_stop_id, d.destination_stop_id, d.hour
        ORDER BY d.hour, d.origin_stop_id, d.destination_stop_id
    """
    n_days = len(used_dates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    first = True
    total = 0
    with gzip.open(output_path, "wt", encoding="utf-8-sig", newline="") as output:
        for chunk in pd.read_sql_query(query, connection, params=tuple(used_dates), chunksize=200_000):
            chunk["observations"] = n_days
            chunk["avg_passengers"] = chunk["total_passengers"] / n_days
            if n_days > 1:
                variance = (
                    chunk["sum_squared"]
                    - chunk["total_passengers"] * chunk["total_passengers"] / n_days
                ).clip(lower=0) / (n_days - 1)
                chunk["std_passengers"] = variance.pow(0.5)
            else:
                chunk["std_passengers"] = 0.0
            columns = [
                "origin_stop_id",
                "destination_stop_id",
                "hour",
                "total_passengers",
                "avg_passengers",
                "std_passengers",
                "positive_days",
                "observations",
            ]
            chunk[columns].to_csv(output, index=False, header=first)
            first = False
            total += len(chunk)
    return total


# -----------------------------------------------------------------------------
# Baseline 및 출력
# -----------------------------------------------------------------------------


def build_segment_baseline(segments: pd.DataFrame, used_dates: Sequence[str]) -> pd.DataFrame:
    frame = segments[segments["date"].isin(used_dates)].copy()
    keys = ["hour", "route_id", "from_stop_id", "to_stop_id"]
    baseline = (
        frame.groupby(keys, as_index=False, dropna=False)
        .agg(
            travel_time=("travel_time", "mean"),
            travel_time_std=("travel_time", "std"),
            distance=("distance", "mean"),
            avg_onboard=("avg_onboard", "mean"),
            avg_onboard_std=("avg_onboard", "std"),
            trips=("trips", "mean"),
            section_passengers=("section_passengers", "mean"),
            observations=("date", "nunique"),
            stop_sequence=("stop_sequence", "min"),
            route_name=("route_name", "first"),
            route_type=("route_type", "first"),
            from_stop_name=("from_stop_name", "first"),
            to_stop_name=("to_stop_name", "first"),
            distance_source=("distance_source", "first"),
        )
    )
    return baseline.sort_values(["hour", "route_id", "stop_sequence", "from_stop_id"], kind="stable")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    od_raw_dir = Path(args.od_raw_dir).resolve()
    manifest_path = Path(args.manifest).resolve()

    if args.fresh:
        clear_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    target_dates, target_hours = load_manifest(manifest_path, args)
    target_date_set = set(target_dates)
    target_hour_set = set(target_hours)

    model_segments_path = data_dir / "model" / "model_segments.csv"
    stop_master_path = data_dir / "master" / "stop_master.csv"
    route_master_path = data_dir / "master" / "route_master.csv"
    route_stop_master_path = data_dir / "master" / "route_stop_master.csv"

    required_paths = [model_segments_path, stop_master_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("필수 파일이 없습니다:\n" + "\n".join(missing_paths))

    print("=" * 72)
    print("현상금 버스모형 최종 입력 생성")
    print("=" * 72)
    print("대상 날짜:", ", ".join(target_dates))
    print("대상 시간:", ", ".join(f"{hour:02d}" for hour in target_hours))

    print("\n[1/5] 마스터 읽기")
    stops = build_stops(stop_master_path)
    routes = build_routes(route_master_path)
    route_stop_edges = build_route_stop_edges(route_stop_master_path)
    print(f"  정류장: {len(stops):,}행")
    print(f"  노선: {len(routes):,}행")
    print(f"  노선 구간 마스터: {len(route_stop_edges):,}행")

    print("\n[2/5] 버스 구간 데이터 정리")
    segments, segment_qa = build_segments(
        model_segments_path,
        target_date_set,
        target_hour_set,
        stops,
        routes,
        route_stop_edges,
        args.travel_time_unit,
    )
    bus_dates = sorted(segments["date"].dropna().astype(str).unique().tolist())
    print(f"  일별 구간 행: {len(segments):,}")
    print("  버스 데이터 날짜:", ", ".join(bus_dates))
    print(
        "  운행시간 단위:",
        segment_qa["travel_time_unit"],
        f"(원시 중앙값={segment_qa['travel_time_raw_median']})",
    )

    print("\n[3/5] OD ZIP 처리")
    od_connection, od_qa = build_od_database(
        od_raw_dir,
        target_dates,
        target_hours,
        work_dir / "od_bus.sqlite",
        args.chunk_size,
        args.fresh,
    )
    od_dates = od_qa["available_dates"]
    print("  OD 데이터 날짜:", ", ".join(od_dates))

    used_dates = sorted(set(target_dates) & set(bus_dates) & set(od_dates))
    if not used_dates:
        od_connection.close()
        raise RuntimeError(
            "버스 구간 데이터와 OD 데이터가 공통으로 가진 날짜가 없습니다."
        )
    missing_common = sorted(set(target_dates) - set(used_dates))
    print("  최종 공통 날짜:", ", ".join(used_dates))
    if missing_common:
        print("  [WARNING] 제외된 날짜:", ", ".join(missing_common))

    segments = segments[segments["date"].isin(used_dates)].copy()
    network_stop_ids = set(segments["from_stop_id"]) | set(segments["to_stop_id"])
    create_network_stop_table(od_connection, network_stop_ids)

    print("\n[4/5] 모델 입력 CSV 생성")
    segments_path = output_dir / "segments.csv"
    segments.to_csv(segments_path, index=False, encoding="utf-8-sig")

    od_path = output_dir / "od_hourly.csv"
    od_rows = export_od_hourly(
        od_connection,
        od_path,
        used_dates,
        args.min_od_passengers,
        network_only=not args.keep_od_outside_network,
    )

    od_baseline_path = output_dir / "od_baseline.csv.gz"
    od_baseline_rows = export_od_baseline(
        od_connection,
        od_baseline_path,
        used_dates,
        network_only=not args.keep_od_outside_network,
    )

    segment_baseline = build_segment_baseline(segments, used_dates)
    segment_baseline_path = output_dir / "segments_baseline.csv.gz"
    segment_baseline.to_csv(
        segment_baseline_path,
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )

    # 실제 사용되는 정류장만 저장
    od_used_stops = set()
    for chunk in pd.read_csv(
        od_path,
        usecols=["origin_stop_id", "destination_stop_id"],
        dtype=str,
        chunksize=200_000,
        low_memory=False,
    ):
        od_used_stops.update(chunk["origin_stop_id"].dropna().tolist())
        od_used_stops.update(chunk["destination_stop_id"].dropna().tolist())
    used_stop_ids = network_stop_ids | od_used_stops
    stops_output = stops[stops["stop_id"].isin(used_stop_ids)].copy()
    stops_output["in_network"] = stops_output["stop_id"].isin(network_stop_ids)
    stops_output["in_od"] = stops_output["stop_id"].isin(od_used_stops)
    stops_path = output_dir / "stops.csv"
    stops_output.to_csv(stops_path, index=False, encoding="utf-8-sig")

    routes_used = routes[routes["route_id"].isin(set(segments["route_id"]))].copy()
    routes_path = output_dir / "routes.csv"
    routes_used.to_csv(routes_path, index=False, encoding="utf-8-sig")

    od_connection.close()

    print("\n[5/5] 품질 보고서")
    qa = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_dates": target_dates,
        "used_dates": used_dates,
        "excluded_dates": missing_common,
        "hours": target_hours,
        "segments": segment_qa,
        "od": od_qa,
        "row_counts": {
            "segments": int(len(segments)),
            "od_hourly": int(od_rows),
            "segments_baseline": int(len(segment_baseline)),
            "od_baseline": int(od_baseline_rows),
            "stops": int(len(stops_output)),
            "routes": int(len(routes_used)),
        },
        "notes": [
            "od_hourly.csv는 출발·도착 ID가 모두 9자리인 버스 정류장 OD만 사용합니다.",
            "출발 정류장과 도착 정류장이 같은 OD는 경로추천 대상에서 제외합니다.",
            "거리 정보를 마스터/좌표 어느 쪽에서도 만들 수 없는 구간은 segments.csv에서 제외합니다.",
            "OD ZIP은 ZIP 단위 트랜잭션으로 처리하여 중단 후 재실행 시 중복 합산을 방지합니다.",
            "4자리 OD ID는 버스 전용 모델에서 제외합니다.",
            "avg_onboard는 구간 총승객수 / 해당 시간대 운행횟수의 추정치입니다.",
        ],
    }
    qa_path = output_dir / "model_input_manifest.json"
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n완료")
    print("-", segments_path)
    print("-", od_path)
    print("-", stops_path)
    print("-", routes_path)
    print("-", segment_baseline_path)
    print("-", od_baseline_path)
    print("-", qa_path)
    print()
    print("현재 model.py에는 다음 폴더를 --input으로 지정하면 됩니다:")
    print(output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\n오류: {error}", file=sys.stderr)
        raise SystemExit(1)
