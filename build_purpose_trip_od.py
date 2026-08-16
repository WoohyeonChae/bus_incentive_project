from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import sqlite3
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


# T-Data data_id=20006 / KSCC_DX_CARD aliases.
ALIASES: Dict[str, Tuple[str, ...]] = {
    "card_id": ("TRCR_ID", "교통카드_ID", "교통카드ID"),
    "ride_dtm": ("RIDE_DTM", "승차_일시", "승차일시"),
    "serial_num": ("SERIAL_NUM", "일련_번호", "일련번호", "일련_수"),
    "transfer_id": ("TRNC_ID", "환승_이용단위", "환승이용단위"),
    "mode_code": ("TRNS_MNS_CD", "교통_수단_코드", "교통수단코드"),
    "transfer_count": ("TRTR_FCNT", "환승_이용횟수", "환승이용횟수"),
    "route_id": ("ROUTE_ID", "노선_ID", "노선ID"),
    "origin_stop_id": (
        "RIDE_BSST_ID",
        "승차_버스정류장_ID",
        "승차_정류장/역사_ID",
        "승차정류장_ID",
    ),
    "alight_dtm": ("ALGH_DTM", "하차_일시", "하차일시"),
    "destination_stop_id": (
        "ALGH_BSST_ID",
        "하차_버스정류장_ID",
        "하차_정류장/역사_ID",
        "하차정류장_ID",
    ),
    "passengers": ("PSNG_NUM", "승객_수", "승객수"),
    "service_date": ("STDR_DE", "기준_날짜", "기준날짜"),
}

REQUIRED_FIELDS = (
    "card_id",
    "ride_dtm",
    "transfer_id",
    "route_id",
    "origin_stop_id",
    "alight_dtm",
    "destination_stop_id",
    "passengers",
)

NETWORK_FILES = (
    "segments.csv",
    "segments_baseline.csv.gz",
    "stops.csv",
    "routes.csv",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "T-Data KSCC_DX_CARD의 환승 이용단위(TRNC_ID)를 이용해 버스-only 목적통행 OD를 "
            "복원하고 기존 model_input OD와 진단 비교합니다."
        )
    )
    p.add_argument("--raw-dir", required=True, help="KSCC_DX_CARD 일별 CSV/CSV.GZ/ZIP 폴더")
    p.add_argument("--output", required=True, help="목적통행 OD 결과 폴더")
    p.add_argument(
        "--model-input",
        help="기존 model_input 폴더. stops.csv ID 검증과 기존 od_hourly.csv 비교에 사용",
    )
    p.add_argument(
        "--dates",
        default=(
            "20260714,20260715,20260716,20260721,20260722,20260723,"
            "20260728,20260729,20260730,20260804,20260805,20260806"
        ),
        help="처리 날짜 YYYYMMDD 쉼표 목록",
    )
    p.add_argument("--hours", default="7,8,9,17,18,19", help="최초 승차시각 기준 분석 시간대")
    p.add_argument("--chunksize", type=int, default=250_000, help="원본 CSV chunk 크기")
    p.add_argument(
        "--stop-id-map",
        help="선택: card_stop_id,model_stop_id 두 열의 CSV. 카드 정류장 ID와 model_input ID가 다를 때 사용",
    )
    p.add_argument(
        "--allow-mixed-mode",
        action="store_true",
        help="버스+철도 혼합 목적통행도 포함. 현재 버스-only 최적화에는 기본값(미지정)을 권장",
    )
    p.add_argument(
        "--allow-unmapped-stops",
        action="store_true",
        help="model_input stops.csv에 없는 출도착 정류장도 OD에 유지. 기본은 제외",
    )
    p.add_argument(
        "--max-trip-minutes",
        type=float,
        default=240.0,
        help="최초 승차~최종 하차가 이 값보다 긴 chain 제외 (기본 240분)",
    )
    p.add_argument(
        "--export-model-input",
        help="선택: 기존 네트워크 파일을 복사하고 OD만 목적통행으로 교체한 새 model_input 폴더",
    )
    p.add_argument("--fresh", action="store_true", help="기존 작업 DB와 출력을 지우고 새로 시작")
    return p.parse_args()


def parse_set(text: str, width: Optional[int] = None) -> List[str]:
    out = sorted({x.strip().replace("-", "") for x in text.split(",") if x.strip()})
    if width and any(len(x) != width or not x.isdigit() for x in out):
        raise ValueError(f"형식이 잘못된 값: {out}")
    return out


def normalize_scalar(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    if text.endswith(".0") and text[:-2].replace("-", "").isdigit():
        text = text[:-2]
    return text


def clean_stop_id(value: object) -> str:
    text = normalize_scalar(value)
    return text


def clean_route_id(value: object) -> str:
    text = normalize_scalar(value)
    return "" if text in {"", "0", "X", "x"} else text


def to_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(str(value).replace(",", "").strip())
        return result if math.isfinite(result) else default
    except Exception:
        return default


def parse_dtm(value: object) -> Optional[datetime]:
    text = normalize_scalar(value)
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    for fmt, n in (("%Y%m%d%H%M%S", 14), ("%Y%m%d%H%M", 12), ("%Y%m%d%H", 10)):
        if len(digits) >= n:
            try:
                return datetime.strptime(digits[:n], fmt)
            except ValueError:
                pass
    return None


def detect_encoding(sample: bytes) -> str:
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "cp949"


def normalize_header(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = text.strip().strip("\"'")
    text = " ".join(text.split())
    return text


def resolve_columns(columns: Sequence[str]) -> Dict[str, str]:
    # T-Data files may use VW_KSCC_DX_CARD.csv as the filename even though
    # the underlying table/schema is KSCC_DX_CARD. Column matching therefore
    # relies on normalized headers, not the filename.
    normalized = {normalize_header(c): str(c) for c in columns}
    upper = {k.upper(): v for k, v in normalized.items()}
    mapping: Dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            alias_n = normalize_header(alias)
            if alias_n in normalized:
                mapping[canonical] = normalized[alias_n]
                break
            if alias_n.upper() in upper:
                mapping[canonical] = upper[alias_n.upper()]
                break
    missing = [f for f in REQUIRED_FIELDS if f not in mapping]
    if missing:
        raise ValueError(
            "KSCC_DX_CARD 필수 열을 찾지 못했습니다: " + ", ".join(missing) +
            "\n실제 헤더: " + ", ".join(map(str, columns[:80]))
        )
    return mapping


@dataclass(frozen=True)
class Source:
    path: Path
    zip_member: Optional[str] = None
    encoding: str = "utf-8-sig"

    @property
    def label(self) -> str:
        return f"{self.path.name}:{self.zip_member}" if self.zip_member else self.path.name


def csv_header_from_stream(stream: io.BufferedIOBase) -> Tuple[List[str], str]:
    sample = stream.read(256_000)
    enc = detect_encoding(sample)
    text = sample.decode(enc, errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return [], enc
    header = lines[0].lstrip("\ufeff")

    # Prefer common delimiters first; fall back to Sniffer on the whole sample.
    candidates = [",", "\t", ";", "|"]
    best = None
    best_score = -1
    for delim in candidates:
        row = next(csv.reader([header], delimiter=delim), [])
        score = len(row)
        if score > best_score:
            best = row
            best_score = score
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:10]), delimiters=",\t;|")
        sniffed = next(csv.reader([header], dialect))
        if len(sniffed) >= max(1, best_score):
            best = sniffed
    except csv.Error:
        pass
    return list(best or []), enc


def discover_sources(raw_dir: Path) -> List[Source]:
    sources: List[Source] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        low = path.name.lower()
        if low.endswith(".zip"):
            try:
                with zipfile.ZipFile(path) as zf:
                    for member in zf.namelist():
                        if member.endswith("/") or not member.lower().endswith((".csv", ".txt")):
                            continue
                        with zf.open(member) as raw:
                            cols, enc = csv_header_from_stream(raw)
                        try:
                            resolve_columns(cols)
                        except Exception:
                            continue
                        sources.append(Source(path=path, zip_member=member, encoding=enc))
            except zipfile.BadZipFile:
                print(f"[WARNING] 손상 ZIP 건너뜀: {path}")
        elif low.endswith((".csv", ".txt", ".csv.gz", ".txt.gz", ".gz")):
            opener = gzip.open if low.endswith(".gz") else open
            try:
                with opener(path, "rb") as raw:
                    cols, enc = csv_header_from_stream(raw)
                resolve_columns(cols)
                sources.append(Source(path=path, encoding=enc))
            except Exception:
                continue
    if not sources:
        # Give a useful diagnostic for the common T-Data naming convention
        # (VW_KSCC_DX_CARD.csv inside TAIMS_*.zip).
        discovered = []
        for path in sorted(raw_dir.rglob("*.zip")):
            try:
                with zipfile.ZipFile(path) as zf:
                    discovered.extend(m for m in zf.namelist() if m.lower().endswith(".csv"))
            except zipfile.BadZipFile:
                continue
        preview = ", ".join(discovered[:10]) if discovered else "(CSV member 없음)"
        raise RuntimeError(
            f"{raw_dir}에서 KSCC_DX_CARD 형태의 CSV/ZIP을 찾지 못했습니다.\n"
            f"ZIP 내부에서 확인된 CSV: {preview}\n"
            "TAIMS ZIP에서는 VW_KSCC_DX_CARD.csv 이름일 수 있습니다. "
            "다음 필수 열이 실제 CSV 헤더에 있어야 합니다: "
            "TRCR_ID, TRNC_ID, RIDE_DTM, ROUTE_ID, RIDE_BSST_ID, "
            "ALGH_DTM, ALGH_BSST_ID, PSNG_NUM"
        )
    return sources


def read_source_chunks(source: Source, chunksize: int) -> Iterator[pd.DataFrame]:
    if source.zip_member:
        with zipfile.ZipFile(source.path) as zf:
            with zf.open(source.zip_member) as raw:
                wrapper = io.TextIOWrapper(raw, encoding=source.encoding, errors="replace", newline="")
                yield from pd.read_csv(wrapper, dtype=str, chunksize=chunksize, low_memory=False)
    else:
        compression = "gzip" if source.path.name.lower().endswith(".gz") else None
        yield from pd.read_csv(
            source.path,
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
            encoding=source.encoding,
            encoding_errors="replace",
            compression=compression,
        )


def init_db(path: Path, fresh: bool) -> sqlite3.Connection:
    if fresh and path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=FILE")
    con.execute("PRAGMA cache_size=-200000")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS legs (
            service_date TEXT NOT NULL,
            card_id TEXT NOT NULL,
            transfer_id TEXT NOT NULL,
            ride_dtm TEXT NOT NULL,
            alight_dtm TEXT,
            serial_num TEXT,
            mode_code TEXT,
            transfer_count TEXT,
            route_id TEXT,
            origin_stop_id TEXT,
            destination_stop_id TEXT,
            passengers REAL NOT NULL,
            source_label TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS processed_sources (
            source_label TEXT PRIMARY KEY,
            rows_loaded INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS purpose_trips (
            trip_hash TEXT PRIMARY KEY,
            service_date TEXT NOT NULL,
            hour INTEGER NOT NULL,
            origin_stop_id TEXT NOT NULL,
            destination_stop_id TEXT NOT NULL,
            passengers REAL NOT NULL,
            legs INTEGER NOT NULL,
            transfers INTEGER NOT NULL,
            route_chain TEXT,
            mode_chain TEXT,
            transfer_stop_chain TEXT,
            first_ride_dtm TEXT,
            last_alight_dtm TEXT,
            duration_minutes REAL,
            all_bus INTEGER NOT NULL,
            passenger_count_consistent INTEGER NOT NULL,
            temporal_order_ok INTEGER NOT NULL,
            origin_mapped INTEGER NOT NULL,
            destination_mapped INTEGER NOT NULL
        );
        """
    )
    con.commit()
    return con


def ingest_sources(
    con: sqlite3.Connection,
    sources: Sequence[Source],
    target_dates: set[str],
    chunksize: int,
) -> Dict[str, int]:
    totals = Counter()
    for source in sources:
        done = con.execute(
            "SELECT rows_loaded FROM processed_sources WHERE source_label=?", (source.label,)
        ).fetchone()
        if done:
            print(f"[재사용] {source.label}: {done[0]:,}행")
            totals["rows_loaded"] += int(done[0])
            continue
        print(f"[읽기] {source.label}")
        source_rows = 0
        for chunk_no, chunk in enumerate(read_source_chunks(source, chunksize), start=1):
            mapping = resolve_columns(chunk.columns)
            rename = {actual: canon for canon, actual in mapping.items()}
            chunk = chunk.rename(columns=rename)
            for optional in ("serial_num", "mode_code", "transfer_count", "service_date"):
                if optional not in chunk.columns:
                    chunk[optional] = ""
            keep = list(dict.fromkeys(REQUIRED_FIELDS + ("serial_num", "mode_code", "transfer_count", "service_date")))
            chunk = chunk[keep].copy()

            # service_date 우선, 없으면 ride_dtm의 앞 8자리.
            service = chunk["service_date"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            ride_digits = chunk["ride_dtm"].fillna("").astype(str).str.replace(r"\D", "", regex=True)
            service = service.where(service.str.fullmatch(r"\d{8}"), ride_digits.str[:8])
            chunk["service_date"] = service
            chunk = chunk[chunk["service_date"].isin(target_dates)]
            if chunk.empty:
                continue

            for col in ("card_id", "transfer_id", "ride_dtm", "alight_dtm", "serial_num", "mode_code", "transfer_count"):
                chunk[col] = chunk[col].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            for col in ("route_id", "origin_stop_id", "destination_stop_id"):
                chunk[col] = chunk[col].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            chunk["route_id"] = chunk["route_id"].replace({"0": "", "X": "", "x": "", "nan": ""})
            chunk["passengers"] = pd.to_numeric(chunk["passengers"], errors="coerce")
            chunk = chunk[
                chunk["card_id"].ne("")
                & chunk["transfer_id"].ne("")
                & chunk["ride_dtm"].ne("")
                & chunk["origin_stop_id"].ne("")
                & chunk["destination_stop_id"].ne("")
                & chunk["passengers"].gt(0)
            ].copy()
            if chunk.empty:
                continue
            chunk["source_label"] = source.label
            chunk.to_sql("legs", con, if_exists="append", index=False, method="multi", chunksize=5000)
            source_rows += len(chunk)
            if chunk_no % 10 == 0:
                con.commit()
                print(f"  chunk {chunk_no:,}: 누적 {source_rows:,}행")
        con.execute(
            "INSERT OR REPLACE INTO processed_sources(source_label, rows_loaded) VALUES(?,?)",
            (source.label, source_rows),
        )
        con.commit()
        totals["rows_loaded"] += source_rows
        print(f"[완료] {source.label}: {source_rows:,}행")

    print("[DB] 인덱스 생성/확인")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_legs_chain ON legs(service_date, card_id, transfer_id, ride_dtm, serial_num)"
    )
    con.commit()
    return dict(totals)


def load_stop_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    frame = pd.read_csv(path, dtype=str)
    required = {"card_stop_id", "model_stop_id"}
    if not required.issubset(frame.columns):
        raise ValueError("stop-id-map에는 card_stop_id, model_stop_id 열이 필요합니다.")
    return {
        clean_stop_id(a): clean_stop_id(b)
        for a, b in zip(frame["card_stop_id"], frame["model_stop_id"])
        if clean_stop_id(a) and clean_stop_id(b)
    }


def load_model_stop_ids(model_input: Optional[Path]) -> set[str]:
    if not model_input:
        return set()
    # stops.csv가 있으면 우선. 없으면 segments에서 양 끝 정류장을 사용.
    stops = model_input / "stops.csv"
    if stops.exists():
        frame = pd.read_csv(stops, dtype=str, low_memory=False)
        candidates = [c for c in frame.columns if c.lower() in {"stop_id", "station_id", "node_id"}]
        if not candidates:
            candidates = [c for c in frame.columns if "stop" in c.lower() and "id" in c.lower()]
        if candidates:
            return {clean_stop_id(x) for x in frame[candidates[0]].dropna() if clean_stop_id(x)}
    seg = model_input / "segments_baseline.csv.gz"
    if seg.exists():
        frame = pd.read_csv(seg, dtype=str, usecols=lambda c: c in {"from_stop_id", "to_stop_id"}, low_memory=False)
        return {
            clean_stop_id(x)
            for c in ("from_stop_id", "to_stop_id") if c in frame.columns
            for x in frame[c].dropna()
            if clean_stop_id(x)
        }
    return set()


def map_stop(stop: str, stop_map: Mapping[str, str]) -> str:
    return stop_map.get(stop, stop)


def build_purpose_trips(
    con: sqlite3.Connection,
    target_dates: Sequence[str],
    stop_map: Mapping[str, str],
    valid_model_stops: set[str],
    allow_mixed_mode: bool,
    allow_unmapped_stops: bool,
    max_trip_minutes: float,
) -> Dict[str, int]:
    # 재실행 시 목적통행 테이블은 현재 옵션에 맞게 새로 계산한다.
    con.execute("DELETE FROM purpose_trips")
    con.commit()

    qa = Counter()
    insert_sql = """
        INSERT OR REPLACE INTO purpose_trips(
            trip_hash, service_date, hour, origin_stop_id, destination_stop_id, passengers,
            legs, transfers, route_chain, mode_chain, transfer_stop_chain,
            first_ride_dtm, last_alight_dtm, duration_minutes,
            all_bus, passenger_count_consistent, temporal_order_ok,
            origin_mapped, destination_mapped
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """

    for service_date in target_dates:
        print(f"[목적통행] {service_date}")
        cursor = con.execute(
            """
            SELECT card_id, transfer_id, ride_dtm, alight_dtm, serial_num, mode_code,
                   transfer_count, route_id, origin_stop_id, destination_stop_id, passengers
            FROM legs
            WHERE service_date=?
            ORDER BY card_id, transfer_id, ride_dtm, serial_num
            """,
            (service_date,),
        )

        current_key: Optional[Tuple[str, str]] = None
        group: List[Tuple] = []
        batch: List[Tuple] = []

        def flush_group(rows: List[Tuple]) -> None:
            if not rows:
                return
            qa["chains_total"] += 1
            card_id, transfer_id = rows[0][0], rows[0][1]
            ride_times = [parse_dtm(r[2]) for r in rows]
            alight_times = [parse_dtm(r[3]) for r in rows]
            if any(t is None for t in ride_times):
                qa["drop_bad_ride_time"] += 1
                return
            first_ride = min(t for t in ride_times if t is not None)
            last_alight_candidates = [t for t in alight_times if t is not None]
            last_alight = max(last_alight_candidates) if last_alight_candidates else None
            if last_alight is None:
                qa["drop_missing_final_alight"] += 1
                return
            duration = (last_alight - first_ride).total_seconds() / 60.0
            if duration < -1e-6 or duration > max_trip_minutes:
                qa["drop_bad_duration"] += 1
                return

            origin_raw = clean_stop_id(rows[0][8])
            destination_raw = clean_stop_id(rows[-1][9])
            origin = map_stop(origin_raw, stop_map)
            destination = map_stop(destination_raw, stop_map)
            if not origin or not destination or origin == destination:
                qa["drop_bad_od"] += 1
                return

            route_ids = [clean_route_id(r[7]) for r in rows]
            mode_codes = [normalize_scalar(r[5]) for r in rows]
            all_bus = all(bool(r) for r in route_ids)
            if not allow_mixed_mode and not all_bus:
                qa["drop_mixed_or_nonbus"] += 1
                return

            passenger_values = [to_float(r[10], 0.0) for r in rows if to_float(r[10], 0.0) > 0]
            if not passenger_values:
                qa["drop_no_passengers"] += 1
                return
            passengers = max(passenger_values)
            consistent = max(passenger_values) - min(passenger_values) <= 1e-9
            if not consistent:
                qa["warn_passenger_count_inconsistent"] += 1

            origin_mapped = (not valid_model_stops) or origin in valid_model_stops
            destination_mapped = (not valid_model_stops) or destination in valid_model_stops
            if valid_model_stops and not allow_unmapped_stops and not (origin_mapped and destination_mapped):
                qa["drop_unmapped_od"] += 1
                return

            temporal_ok = True
            for prev, nxt in zip(rows, rows[1:]):
                prev_alight = parse_dtm(prev[3])
                next_ride = parse_dtm(nxt[2])
                if prev_alight and next_ride and next_ride < prev_alight:
                    temporal_ok = False
                    break
            if not temporal_ok:
                qa["warn_temporal_overlap"] += 1

            transfer_parts = []
            for prev, nxt in zip(rows, rows[1:]):
                transfer_parts.append(
                    f"{map_stop(clean_stop_id(prev[9]), stop_map)}>{map_stop(clean_stop_id(nxt[8]), stop_map)}"
                )
            route_chain = ">".join(r or "RAIL/OTHER" for r in route_ids)
            mode_chain = ">".join(m or "?" for m in mode_codes)
            transfer_stop_chain = ";".join(transfer_parts)
            trip_hash = hashlib.sha256(
                f"{service_date}|{card_id}|{transfer_id}".encode("utf-8")
            ).hexdigest()[:24]
            batch.append(
                (
                    trip_hash,
                    service_date,
                    first_ride.hour,
                    origin,
                    destination,
                    passengers,
                    len(rows),
                    max(0, len(rows) - 1),
                    route_chain,
                    mode_chain,
                    transfer_stop_chain,
                    first_ride.strftime("%Y%m%d%H%M%S"),
                    last_alight.strftime("%Y%m%d%H%M%S"),
                    duration,
                    int(all_bus),
                    int(consistent),
                    int(temporal_ok),
                    int(origin_mapped),
                    int(destination_mapped),
                )
            )
            qa["chains_kept"] += 1
            qa["passengers_kept"] += int(round(passengers))

        for row in cursor:
            key = (str(row[0]), str(row[1]))
            if current_key is None:
                current_key = key
            if key != current_key:
                flush_group(group)
                group = []
                current_key = key
                if len(batch) >= 10_000:
                    con.executemany(insert_sql, batch)
                    con.commit()
                    batch.clear()
            group.append(row)
        flush_group(group)
        if batch:
            con.executemany(insert_sql, batch)
            con.commit()
        print(
            f"  누적 chain {qa['chains_total']:,}, 유지 {qa['chains_kept']:,}, "
            f"혼합/비버스 제외 {qa['drop_mixed_or_nonbus']:,}"
        )
    return dict(qa)


def export_outputs(
    con: sqlite3.Connection,
    output: Path,
    target_dates: Sequence[str],
    hours: Sequence[int],
    model_input: Optional[Path],
    qa: Mapping[str, int],
    ingest_qa: Mapping[str, int],
    sources: Sequence[Source],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    hours_sql = ",".join("?" for _ in hours)

    detail = pd.read_sql_query(
        f"""
        SELECT * FROM purpose_trips
        WHERE hour IN ({hours_sql})
        ORDER BY service_date, hour, origin_stop_id, destination_stop_id, first_ride_dtm
        """,
        con,
        params=list(hours),
    )
    detail.to_csv(output / "purpose_trip_detail.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

    if detail.empty:
        od = pd.DataFrame(columns=["date", "hour", "origin_stop_id", "destination_stop_id", "passengers"])
    else:
        od = (
            detail.groupby(["service_date", "hour", "origin_stop_id", "destination_stop_id"], as_index=False)["passengers"]
            .sum()
            .rename(columns={"service_date": "date"})
        )
    od.to_csv(output / "od_hourly.csv", index=False, encoding="utf-8-sig")

    # Optimizer-compatible baseline: 관측 날짜 전체(0수요일 포함 평균)가 아니라 기존 파이프라인과 맞춰
    # positive OD observations를 기본으로 만들되 observations는 전체 target date 수를 기록한다.
    if od.empty:
        baseline = pd.DataFrame(
            columns=["hour", "origin_stop_id", "destination_stop_id", "total_passengers", "avg_passengers", "std_passengers", "positive_days", "observations"]
        )
    else:
        baseline = (
            od.groupby(["hour", "origin_stop_id", "destination_stop_id"], as_index=False)
            .agg(
                total_passengers=("passengers", "sum"),
                avg_passengers=("passengers", "mean"),
                std_passengers=("passengers", "std"),
                positive_days=("date", "nunique"),
            )
        )
        baseline["std_passengers"] = baseline["std_passengers"].fillna(0.0)
        baseline["observations"] = len(target_dates)
    baseline.to_csv(output / "od_baseline.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

    # Chain 구조 요약
    if detail.empty:
        chain_summary = pd.DataFrame(columns=["date", "hour", "trips", "passengers", "mean_legs", "transfer_trip_share"])
    else:
        tmp = detail.copy()
        tmp["is_transfer"] = (tmp["transfers"] > 0).astype(int)
        chain_summary = (
            tmp.groupby(["service_date", "hour"], as_index=False)
            .agg(
                trips=("trip_hash", "nunique"),
                passengers=("passengers", "sum"),
                mean_legs=("legs", "mean"),
                transfer_trip_share=("is_transfer", "mean"),
            )
            .rename(columns={"service_date": "date"})
        )
    chain_summary.to_csv(output / "purpose_trip_summary_by_hour.csv", index=False, encoding="utf-8-sig")

    # 기존 OA-21222 기반 OD와 비교. 합치거나 보정하지 않고 진단만 한다.
    comparison_summary = pd.DataFrame()
    comparison_cells = pd.DataFrame()
    if model_input and (model_input / "od_hourly.csv").exists():
        old = pd.read_csv(model_input / "od_hourly.csv", dtype={"date": str, "origin_stop_id": str, "destination_stop_id": str}, low_memory=False)
        required = {"date", "hour", "origin_stop_id", "destination_stop_id", "passengers"}
        if required.issubset(old.columns):
            old = old[list(required)].copy()
            old["date"] = old["date"].astype(str).str.replace(r"\.0$", "", regex=True)
            old["hour"] = pd.to_numeric(old["hour"], errors="coerce")
            old["passengers"] = pd.to_numeric(old["passengers"], errors="coerce").fillna(0.0)
            old = old[old["date"].isin(target_dates) & old["hour"].isin(hours)]
            old_sum = old.groupby(["date", "hour"], as_index=False)["passengers"].sum().rename(columns={"passengers": "leg_od_passengers"})
            new_sum = od.groupby(["date", "hour"], as_index=False)["passengers"].sum().rename(columns={"passengers": "purpose_od_passengers"}) if not od.empty else pd.DataFrame(columns=["date", "hour", "purpose_od_passengers"])
            comparison_summary = old_sum.merge(new_sum, on=["date", "hour"], how="outer").fillna(0.0)
            comparison_summary["purpose_to_leg_ratio"] = comparison_summary["purpose_od_passengers"] / comparison_summary["leg_od_passengers"].replace(0, pd.NA)
            comparison_summary.to_csv(output / "od_comparison_by_hour.csv", index=False, encoding="utf-8-sig")

            old_cells = old.groupby(["date", "hour", "origin_stop_id", "destination_stop_id"], as_index=False)["passengers"].sum().rename(columns={"passengers": "leg_od_passengers"})
            new_cells = od.rename(columns={"passengers": "purpose_od_passengers"})
            comparison_cells = old_cells.merge(new_cells, on=["date", "hour", "origin_stop_id", "destination_stop_id"], how="outer").fillna(0.0)
            comparison_cells["difference"] = comparison_cells["purpose_od_passengers"] - comparison_cells["leg_od_passengers"]
            comparison_cells.to_csv(output / "od_comparison_cells.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose_trip_definition": "group by (service_date, TRCR_ID, TRNC_ID); OD = first boarding stop -> final alighting stop; hour = first boarding hour",
        "bus_only_rule": "all legs must have non-empty/nonzero ROUTE_ID unless --allow-mixed-mode",
        "passenger_weight_rule": "PSNG_NUM maximum within chain; mismatch is flagged",
        "target_dates": list(target_dates),
        "hours": list(hours),
        "sources": [s.label for s in sources],
        "ingest_qa": dict(ingest_qa),
        "purpose_trip_qa": dict(qa),
        "output_rows": {
            "purpose_trip_detail": int(len(detail)),
            "od_hourly": int(len(od)),
            "od_baseline": int(len(baseline)),
            "comparison_cells": int(len(comparison_cells)),
        },
        "important_note": (
            "기존 OA-21222 OD는 leg OD 비교/검증용으로만 사용하며 목적통행 OD에 더하지 않습니다. "
            "model_input 정류장 ID와 카드 정류장 ID가 다르면 stop_id_map.csv가 필요합니다."
        ),
    }
    (output / "purpose_trip_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def export_model_input(existing: Path, destination: Path, purpose_output: Path, target_dates: Sequence[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in NETWORK_FILES:
        src = existing / name
        if src.exists():
            shutil.copy2(src, destination / name)
    shutil.copy2(purpose_output / "od_hourly.csv", destination / "od_hourly.csv")
    shutil.copy2(purpose_output / "od_baseline.csv.gz", destination / "od_baseline.csv.gz")

    old_manifest = existing / "model_input_manifest.json"
    payload: Dict[str, object] = {}
    if old_manifest.exists():
        try:
            payload = json.loads(old_manifest.read_text(encoding="utf-8-sig"))
        except Exception:
            payload = {}
    payload["used_dates"] = list(target_dates)
    payload["od_source"] = "KSCC_DX_CARD purpose-trip chains grouped by TRCR_ID + TRNC_ID"
    payload["od_preprocessor"] = "build_purpose_trip_od.py"
    (destination / "model_input_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    output = Path(args.output).resolve()
    model_input = Path(args.model_input).resolve() if args.model_input else None
    target_dates = parse_set(args.dates, width=8)
    hours = [int(x) for x in parse_set(args.hours)]
    if any(h < 0 or h > 23 for h in hours):
        raise ValueError("hours는 0~23이어야 합니다.")

    if args.fresh and output.exists():
        for child in output.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(raw_dir)
    print("KSCC_DX_CARD 소스:", len(sources), "개")
    for source in sources[:20]:
        print(" -", source.label)
    if len(sources) > 20:
        print(f" - ... {len(sources)-20}개 추가")

    con = init_db(output / "_work" / "purpose_trip.sqlite", args.fresh)
    try:
        ingest_qa = ingest_sources(con, sources, set(target_dates), args.chunksize)
        stop_map = load_stop_map(args.stop_id_map)
        valid_stops = load_model_stop_ids(model_input)
        print(f"model_input 정류장 ID: {len(valid_stops):,}개")
        if stop_map:
            print(f"정류장 ID 매핑: {len(stop_map):,}개")
        qa = build_purpose_trips(
            con,
            target_dates,
            stop_map,
            valid_stops,
            args.allow_mixed_mode,
            args.allow_unmapped_stops,
            args.max_trip_minutes,
        )
        export_outputs(con, output, target_dates, hours, model_input, qa, ingest_qa, sources)
    finally:
        con.close()

    if args.export_model_input:
        if not model_input:
            raise ValueError("--export-model-input을 쓰려면 --model-input이 필요합니다.")
        export_model_input(model_input, Path(args.export_model_input).resolve(), output, target_dates)
        print("새 model_input:", Path(args.export_model_input).resolve())

    print("완료:", output)
    print("핵심 출력:")
    print(" - od_hourly.csv                 (optimizer 호환 목적통행 OD)")
    print(" - od_baseline.csv.gz            (optimizer 호환 baseline OD)")
    print(" - purpose_trip_detail.csv.gz    (익명 trip chain 상세)")
    print(" - purpose_trip_summary_by_hour.csv")
    print(" - od_comparison_by_hour.csv     (기존 leg OD와 진단 비교)")
    print(" - purpose_trip_manifest.json")


if __name__ == "__main__":
    main()