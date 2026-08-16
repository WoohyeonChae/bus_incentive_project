from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


# -----------------------------------------------------------------------------
# 기본 설정
# -----------------------------------------------------------------------------

BASE_URL = "http://openapi.seoul.go.kr:8088"
DEFAULT_HOURS = ("07", "08", "09", "17", "18", "19")
PAGE_SIZE = 1000  # 서울 열린데이터광장 Open API의 일반적인 최대 행 수
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Dataset:
    key_name: str
    oa_id: str
    services: tuple[str, ...]
    required: frozenset[str]
    kind: str  # dated_parameter | dated_scan | master
    value_prefix: str | None = None
    date_field: str = "CRTR_DD"


DATASETS = (
    Dataset(
        "route_section_passengers",
        "OA-21218",
        ("tpssRouteSectionUser",),
        frozenset(
            {
                "CRTR_DD",
                "RTE_ID",
                "DPTRE_STOPS_ID",
                "ARVL_STOPS_ID",
                "RDS_TNOPE_07",
            }
        ),
        "dated_parameter",
        "RDS_TNOPE_",
    ),
    Dataset(
        "station_route_operations",
        "OA-21220",
        ("tpssStationRouteTurn",),
        frozenset({"CRTR_DD", "RTE_ID", "STOPS_ID", "BUS_OPR_07"}),
        "dated_scan",
        "BUS_OPR_",
    ),
    Dataset(
        "route_section_time",
        "OA-21217",
        ("tpssRouteSectionTime",),
        frozenset(
            {
                "CRTR_DD",
                "RTE_ID",
                "DPTRE_STOPS_ID",
                "ARVL_STOPS_ID",
                "OPR_HR_07",
            }
        ),
        "dated_scan",
        "OPR_HR_",
    ),
    Dataset(
        "route_master",
        "OA-21230",
        ("tbisMasterRoute",),
        frozenset({"RTE_ID"}),
        "master",
    ),
    Dataset(
        "route_stop_master",
        "OA-21233",
        ("masterRouteNode",),
        frozenset({"RTE_ID", "CRTR_ID"}),
        "master",
    ),
    Dataset(
        "stop_master",
        "OA-21231",
        ("tbisMasterStation",),
        frozenset({"CRTR_ID"}),
        "master",
    ),
)


# -----------------------------------------------------------------------------
# 명령행 인자
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "서울 버스 6개 Open API 통합 수집기. "
            "날짜 파라미터가 없는 대용량 API는 날짜 인덱스를 먼저 탐색해 "
            "필요한 페이지만 수집하며, 실패 시 전체 스캔으로 보완할 수 있습니다."
        )
    )
    parser.add_argument(
        "--end-date",
        help="수집 기준일 YYYYMMDD (기본: 오늘-5일)",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=4,
        help="몇 주를 수집할지 (기본: 4주)",
    )
    parser.add_argument(
        "--weekdays",
        default="1,2,3",
        help="수집 요일. 월=0 ... 일=6 (기본: 1,2,3 = 화수목)",
    )
    parser.add_argument(
        "--hours",
        default=",".join(DEFAULT_HOURS),
        help="분석 시간대, 쉼표 구분 (기본: 07,08,09,17,18,19)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data"),
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "api_config.json"),
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.10,
        help="요청 직후 대기시간(초). 스레드별 적용 (기본: 0.10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="동시 다운로드 스레드 수 (기본: 4, 권장 2~4)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="서비스명과 스키마만 검사하고 종료",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="기존 체크포인트를 모두 삭제하고 다시 받기",
    )
    parser.add_argument(
        "--no-full-fallback",
        action="store_true",
        help=(
            "날짜 인덱스 방식으로 일부 목표일을 찾지 못해도 전체 API를 "
            "추가 스캔하지 않음"
        ),
    )
    parser.add_argument(
        "--strict-dates",
        action="store_true",
        help="목표 날짜가 하나라도 누락되면 오류로 종료",
    )
    parser.add_argument(
        "--skip-model-build",
        action="store_true",
        help="6개 원천/정리 CSV만 만들고 model_segments.csv는 만들지 않음",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# 날짜 및 설정
# -----------------------------------------------------------------------------


def parse_csv_ints(value: str, minimum: int, maximum: int) -> tuple[int, ...]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        number = int(token)
        if number < minimum or number > maximum:
            raise ValueError(f"범위를 벗어난 값: {number}")
        result.append(number)
    if not result:
        raise ValueError("하나 이상의 값이 필요합니다.")
    return tuple(dict.fromkeys(result))


def target_dates(end: date, weeks: int, weekdays: tuple[int, ...]) -> list[str]:
    if weeks < 1:
        raise ValueError("--weeks는 1 이상이어야 합니다.")

    monday = end - timedelta(days=end.weekday())
    result: list[str] = []
    for weeks_back in range(weeks - 1, -1, -1):
        week = monday - timedelta(weeks=weeks_back)
        for weekday in weekdays:
            current = week + timedelta(days=weekday)
            if current <= end:
                result.append(current.strftime("%Y%m%d"))
    return sorted(dict.fromkeys(result))


def load_keys(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(
            f"설정 파일이 없습니다: {path}\n"
            "api_config.example.json을 복사해 api_config.json을 만드세요."
        )

    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    default_key = str(raw.get("default", raw.get("api_key", ""))).strip()

    keys: dict[str, str] = {}
    for dataset in DATASETS:
        value = str(raw.get(dataset.key_name, default_key)).strip()
        keys[dataset.key_name] = value

    missing = [
        dataset.key_name
        for dataset in DATASETS
        if not keys.get(dataset.key_name) or "여기에" in keys[dataset.key_name]
    ]
    if missing:
        raise SystemExit("인증키가 비어 있습니다: " + ", ".join(missing))
    return keys


# -----------------------------------------------------------------------------
# API 클라이언트
# -----------------------------------------------------------------------------


class ApiResponseError(RuntimeError):
    def __init__(self, service: str, code: str, message: str):
        super().__init__(f"{service}: {code} - {message}")
        self.service = service
        self.code = code
        self.message = message


@dataclass(frozen=True)
class PageResult:
    rows: list[dict[str, Any]]
    total: int


class SeoulAPI:
    def __init__(self, pause: float):
        self.pause = max(0.0, pause)
        self.local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers["User-Agent"] = "seoul-bus-six-api-collector/2.0"
            self.local.session = session
        return session

    def page(
        self,
        key: str,
        service: str,
        start: int,
        end: int,
        suffix: str | None = None,
        max_retries: int = 5,
    ) -> PageResult:
        parts = [BASE_URL, key, "json", service, str(start), str(end)]
        if suffix:
            parts.append(suffix)
        url = "/".join(part.strip("/") for part in parts)

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._session().get(url, timeout=90)

                if response.status_code == 404:
                    raise RuntimeError(
                        f"HTTP 404: 서비스명 또는 URL 형식 확인 필요 ({service})"
                    )

                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f"일시적 HTTP 오류 {response.status_code}",
                        response=response,
                    )

                response.raise_for_status()
                payload = response.json()
                block = payload.get(service)

                # INFO-200은 정상적인 '해당 데이터 없음'으로 취급
                if not isinstance(block, dict):
                    result = payload.get("RESULT", {})
                    code = str(result.get("CODE", "UNKNOWN"))
                    message = str(result.get("MESSAGE", str(payload)[:300]))
                    if code == "INFO-200":
                        time.sleep(self.pause)
                        return PageResult([], 0)
                    raise ApiResponseError(service, code, message)

                result = block.get("RESULT", {})
                code = result.get("CODE")
                if code not in (None, "INFO-000"):
                    if code == "INFO-200":
                        time.sleep(self.pause)
                        return PageResult([], 0)
                    raise ApiResponseError(
                        service,
                        str(code),
                        str(result.get("MESSAGE", "알 수 없는 오류")),
                    )

                rows = block.get("row", []) or []
                total = int(block.get("list_total_count", len(rows)))
                time.sleep(self.pause)
                return PageResult(rows, total)

            except ApiResponseError:
                # 서비스명/인증/스키마 오류는 재시도로 해결되지 않음
                raise
            except (requests.RequestException, ValueError, RuntimeError) as error:
                last_error = error
                if attempt == max_retries - 1:
                    break
                wait_seconds = min(30, 2**attempt)
                time.sleep(wait_seconds)

        raise RuntimeError(f"API 요청 실패: {service}: {last_error}")


# -----------------------------------------------------------------------------
# 공통 파일 처리
# -----------------------------------------------------------------------------


def atomic_csv(df: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
        compression=compression,
    )
    try:
        os.replace(tmp, path)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise PermissionError(
            f"{path} 파일이 Excel 등에서 열려 있습니다. 파일을 닫고 다시 실행하세요."
        )


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.DataFrame(rows), path)


def normalize_date(value: Any) -> str:
    text = str(value or "").strip().replace("-", "").replace("/", "")
    return text[:8] if len(text) >= 8 and text[:8].isdigit() else ""


def part_path(directory: Path, page_number: int) -> Path:
    return directory / f"part_{page_number:06d}.csv"


def marker_path(directory: Path, page_number: int) -> Path:
    return directory / f"page_{page_number:06d}.done"


def page_bounds(page_number: int, total: int) -> tuple[int, int]:
    start = (page_number - 1) * PAGE_SIZE + 1
    end = min(page_number * PAGE_SIZE, total)
    return start, end


def read_parts(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("part_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat(
        (pd.read_csv(file, dtype=str, low_memory=False) for file in files),
        ignore_index=True,
    )


def found_dates_in_parts(path: Path, date_field: str) -> set[str]:
    found: set[str] = set()
    for file in sorted(path.rglob("part_*.csv")):
        try:
            frame = pd.read_csv(
                file,
                usecols=lambda column: column == date_field,
                dtype=str,
                low_memory=False,
            )
        except (ValueError, pd.errors.EmptyDataError):
            continue
        if date_field in frame.columns:
            found.update(normalize_date(value) for value in frame[date_field])
    found.discard("")
    return found


# -----------------------------------------------------------------------------
# 서비스 확인
# -----------------------------------------------------------------------------


def choose_service(
    api: SeoulAPI,
    dataset: Dataset,
    key: str,
    probe_dates: list[str],
) -> str:
    errors: list[str] = []

    for service in dataset.services:
        try:
            if dataset.kind == "dated_parameter":
                # 최신 목표일이 적재 지연ㆍ공휴일 등으로 비어 있을 수 있으므로
                # 최근 목표일들을 역순으로 확인한다.
                responses: list[PageResult] = []
                for probe_date in reversed(probe_dates):
                    response = api.page(key, service, 1, 5, probe_date)
                    responses.append(response)
                    if response.rows:
                        break
                response = next(
                    (candidate for candidate in responses if candidate.rows),
                    responses[-1],
                )
            else:
                response = api.page(key, service, 1, 5)

            fields = (
                set().union(*(row.keys() for row in response.rows))
                if response.rows
                else set()
            )
            if response.rows and dataset.required.issubset(fields):
                return service
            errors.append(f"{service}: 필드 불일치 또는 빈 응답 ({sorted(fields)})")
        except Exception as error:
            errors.append(str(error))

    raise RuntimeError(
        f"{dataset.oa_id}({dataset.key_name}) 서비스 확인 실패\n  "
        + "\n  ".join(errors)
        + "\n해당 사이트의 [Open API] 탭에 표시되는 샘플 URL을 확인해 주세요."
    )


# -----------------------------------------------------------------------------
# 병렬 페이지 수집기
# -----------------------------------------------------------------------------


def fetch_selected_pages(
    api: SeoulAPI,
    key: str,
    service: str,
    total: int,
    page_numbers: Iterable[int],
    checkpoint: Path,
    workers: int,
    wanted_dates: set[str] | None = None,
    date_field: str = "CRTR_DD",
    suffix: str | None = None,
) -> set[str]:
    """
    지정한 페이지를 병렬 수집한다.

    wanted_dates가 있으면 해당 날짜의 행만 part CSV에 저장한다.
    행이 없더라도 .done 마커를 기록해 재실행 시 중복 호출을 막는다.
    """

    checkpoint.mkdir(parents=True, exist_ok=True)
    pages = sorted(set(page_numbers))
    if not pages:
        return set()

    pending = [
        page_number
        for page_number in pages
        if not marker_path(checkpoint, page_number).exists()
    ]

    print(
        f"  페이지 {len(pages):,}개 대상, "
        f"신규 요청 {len(pending):,}개, workers={workers}"
    )

    found_dates: set[str] = set()
    progress_lock = threading.Lock()
    completed = 0

    def task(page_number: int) -> tuple[int, list[dict[str, Any]], set[str]]:
        start, end = page_bounds(page_number, total)
        result = api.page(key, service, start, end, suffix)

        selected: list[dict[str, Any]] = []
        local_dates: set[str] = set()
        for row in result.rows:
            if wanted_dates is None:
                selected.append(dict(row))
                current_date = normalize_date(row.get(date_field))
                if current_date:
                    local_dates.add(current_date)
                continue

            current_date = normalize_date(row.get(date_field))
            if current_date in wanted_dates:
                copied = dict(row)
                copied[date_field] = current_date
                selected.append(copied)
                local_dates.add(current_date)

        return page_number, selected, local_dates

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(task, page): page for page in pending}
            for future in as_completed(futures):
                page_number = futures[future]
                try:
                    returned_page, rows, local_dates = future.result()
                except Exception as error:
                    raise RuntimeError(
                        f"{service} 페이지 {page_number} 다운로드 실패: {error}"
                    ) from error

                if rows:
                    write_rows(rows, part_path(checkpoint, returned_page))
                marker_path(checkpoint, returned_page).write_text("ok", encoding="ascii")
                found_dates.update(local_dates)

                with progress_lock:
                    completed += 1
                    if completed == 1 or completed % 20 == 0 or completed == len(pending):
                        print(f"    {completed:,} / {len(pending):,} 페이지 완료")

    found_dates.update(found_dates_in_parts(checkpoint, date_field))
    return found_dates


# -----------------------------------------------------------------------------
# 날짜 파라미터형 API
# -----------------------------------------------------------------------------


def collect_parameter_dates(
    api: SeoulAPI,
    dataset: Dataset,
    key: str,
    service: str,
    dates: list[str],
    checkpoint: Path,
    workers: int,
) -> set[str]:
    found: set[str] = set()

    for day in dates:
        day_dir = checkpoint / day
        done = day_dir / "DONE"
        if done.exists():
            print(f"  {day}: 완료된 체크포인트 사용")
            found.add(day)
            continue

        first = api.page(key, service, 1, PAGE_SIZE, day)
        if not first.rows or first.total == 0:
            print(f"  {day}: 데이터 없음")
            continue

        total_pages = math.ceil(first.total / PAGE_SIZE)
        write_rows(first.rows, part_path(day_dir, 1))
        marker_path(day_dir, 1).parent.mkdir(parents=True, exist_ok=True)
        marker_path(day_dir, 1).write_text("ok", encoding="ascii")

        remaining_pages = range(2, total_pages + 1)
        fetch_selected_pages(
            api=api,
            key=key,
            service=service,
            total=first.total,
            page_numbers=remaining_pages,
            checkpoint=day_dir,
            workers=workers,
            suffix=day,
        )

        done.write_text("ok", encoding="ascii")
        found.add(day)
        print(f"  {day}: {first.total:,}행 완료")

    return found


# -----------------------------------------------------------------------------
# 날짜 파라미터가 없는 대용량 API: 날짜 페이지 인덱스
# -----------------------------------------------------------------------------


def load_page_index(path: Path) -> dict[int, tuple[str, str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(page): (str(value[0]), str(value[1]))
        for page, value in raw.items()
    }


def save_page_index(path: Path, index: dict[int, tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        str(page): [minimum, maximum]
        for page, (minimum, maximum) in sorted(index.items())
    }
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_page_date_range(
    api: SeoulAPI,
    key: str,
    service: str,
    total: int,
    page_number: int,
    cache: dict[int, tuple[str, str]],
    date_field: str,
    cache_path: Path,
) -> tuple[str, str]:
    if page_number in cache:
        return cache[page_number]

    start, end = page_bounds(page_number, total)
    result = api.page(key, service, start, end)
    dates = sorted(
        date_value
        for date_value in (
            normalize_date(row.get(date_field)) for row in result.rows
        )
        if date_value
    )
    value = (dates[0], dates[-1]) if dates else ("", "")
    cache[page_number] = value
    save_page_index(cache_path, cache)
    return value


def detect_date_order(
    api: SeoulAPI,
    key: str,
    service: str,
    total: int,
    date_field: str,
    cache: dict[int, tuple[str, str]],
    cache_path: Path,
) -> str | None:
    page_count = math.ceil(total / PAGE_SIZE)
    sample_count = min(11, page_count)
    sample_pages = sorted(
        {
            1 + round((page_count - 1) * index / max(1, sample_count - 1))
            for index in range(sample_count)
        }
    )

    representative: list[str] = []
    for page in sample_pages:
        minimum, maximum = get_page_date_range(
            api,
            key,
            service,
            total,
            page,
            cache,
            date_field,
            cache_path,
        )
        representative.append(maximum or minimum)

    valid = [value for value in representative if value]
    if len(valid) < 2:
        return None

    ascending = all(left <= right for left, right in zip(valid, valid[1:]))
    descending = all(left >= right for left, right in zip(valid, valid[1:]))

    if ascending and not descending:
        return "ascending"
    if descending and not ascending:
        return "descending"
    if ascending and descending:
        return "constant"
    return None


def lower_bound_page(
    page_count: int,
    predicate: Any,
) -> int | None:
    low, high = 1, page_count
    answer: int | None = None
    while low <= high:
        middle = (low + high) // 2
        if predicate(middle):
            answer = middle
            high = middle - 1
        else:
            low = middle + 1
    return answer


def upper_bound_page(
    page_count: int,
    predicate: Any,
) -> int | None:
    low, high = 1, page_count
    answer: int | None = None
    while low <= high:
        middle = (low + high) // 2
        if predicate(middle):
            answer = middle
            low = middle + 1
        else:
            high = middle - 1
    return answer


def pages_for_target_dates(
    api: SeoulAPI,
    dataset: Dataset,
    key: str,
    service: str,
    total: int,
    dates: list[str],
    checkpoint: Path,
) -> tuple[set[int] | None, str | None]:
    """
    API가 날짜 순으로 정렬되어 있으면 각 목표 날짜가 들어 있는 페이지 구간을
    이진 탐색해 필요한 페이지만 반환한다. 정렬을 확인할 수 없으면 None을 반환한다.
    """

    page_count = math.ceil(total / PAGE_SIZE)
    cache_path = checkpoint / "page_date_index.json"
    cache = load_page_index(cache_path)

    order = detect_date_order(
        api,
        key,
        service,
        total,
        dataset.date_field,
        cache,
        cache_path,
    )
    if order not in {"ascending", "descending", "constant"}:
        return None, None

    def date_range(page: int) -> tuple[str, str]:
        return get_page_date_range(
            api,
            key,
            service,
            total,
            page,
            cache,
            dataset.date_field,
            cache_path,
        )

    selected_pages: set[int] = set()

    for target in dates:
        if order == "descending":
            first = lower_bound_page(
                page_count,
                lambda page: (date_range(page)[0] or "99999999") <= target,
            )
            last = upper_bound_page(
                page_count,
                lambda page: (date_range(page)[1] or "00000000") >= target,
            )
        elif order == "ascending":
            first = lower_bound_page(
                page_count,
                lambda page: (date_range(page)[1] or "00000000") >= target,
            )
            last = upper_bound_page(
                page_count,
                lambda page: (date_range(page)[0] or "99999999") <= target,
            )
        else:
            first, last = 1, page_count

        if first is None or last is None or first > last:
            continue

        # 경계에서 같은 날짜가 다음/이전 페이지에 걸칠 수 있으므로 한 페이지 여유
        start_page = max(1, first - 1)
        end_page = min(page_count, last + 1)
        selected_pages.update(range(start_page, end_page + 1))

    return selected_pages, order


def collect_scan_dates(
    api: SeoulAPI,
    dataset: Dataset,
    key: str,
    service: str,
    dates: list[str],
    checkpoint: Path,
    workers: int,
    full_fallback: bool,
    strict_dates: bool,
) -> set[str]:
    done = checkpoint / "DONE"
    wanted = set(dates)

    if done.exists():
        found = found_dates_in_parts(checkpoint, dataset.date_field)
        missing = wanted - found
        if not missing:
            print("  완료된 체크포인트 사용")
            return found
        print(
            "  기존 체크포인트에 누락 날짜가 있어 이어서 수집합니다:",
            ", ".join(sorted(missing)),
        )
        done.unlink(missing_ok=True)

    first = api.page(key, service, 1, PAGE_SIZE)
    if not first.rows or first.total == 0:
        raise RuntimeError(f"{service}: API 전체 데이터가 비어 있습니다.")

    total = first.total
    page_count = math.ceil(total / PAGE_SIZE)
    print(f"  API 전체 규모: {total:,}행, {page_count:,}페이지")

    pages, order = pages_for_target_dates(
        api,
        dataset,
        key,
        service,
        total,
        dates,
        checkpoint,
    )

    if pages is None:
        print("  날짜 정렬을 확인하지 못했습니다. 전체 페이지 스캔으로 전환합니다.")
        pages = set(range(1, page_count + 1))
    else:
        print(
            f"  날짜 정렬: {order}; 목표 날짜용 페이지 {len(pages):,}개만 우선 수집"
        )

    # 1페이지는 이미 받았지만 fetch_selected_pages가 체크포인트를 관리하므로
    # 다시 호출해도 한 번만 저장된다.
    found = fetch_selected_pages(
        api=api,
        key=key,
        service=service,
        total=total,
        page_numbers=pages,
        checkpoint=checkpoint,
        workers=workers,
        wanted_dates=wanted,
        date_field=dataset.date_field,
    )

    missing = wanted - found
    if missing and full_fallback:
        print(
            "  목표 날짜 일부를 찾지 못했습니다. "
            f"전체 스캔으로 보완: {', '.join(sorted(missing))}"
        )
        remaining_pages = set(range(1, page_count + 1)) - set(pages)
        found = fetch_selected_pages(
            api=api,
            key=key,
            service=service,
            total=total,
            page_numbers=remaining_pages,
            checkpoint=checkpoint,
            workers=workers,
            wanted_dates=wanted,
            date_field=dataset.date_field,
        )
        found.update(found_dates_in_parts(checkpoint, dataset.date_field))
        missing = wanted - found

    status = {
        "service": service,
        "total_rows": total,
        "total_pages": page_count,
        "target_dates": sorted(wanted),
        "found_dates": sorted(found),
        "missing_dates": sorted(missing),
        "date_order": order,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    (checkpoint / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if missing:
        message = f"{dataset.key_name}: 누락된 날짜 {sorted(missing)}"
        if strict_dates:
            raise RuntimeError(message)
        print("  경고:", message)

    if not missing:
        done.write_text("ok", encoding="ascii")
    else:
        done.unlink(missing_ok=True)
    return found


# -----------------------------------------------------------------------------
# 마스터 API
# -----------------------------------------------------------------------------


def collect_master(
    api: SeoulAPI,
    key: str,
    service: str,
    checkpoint: Path,
    workers: int,
) -> None:
    done = checkpoint / "DONE"
    if done.exists():
        print("  완료된 체크포인트 사용")
        return

    first = api.page(key, service, 1, PAGE_SIZE)
    if first.total == 0:
        raise RuntimeError(f"{service}: 마스터 데이터가 비어 있습니다.")

    total_pages = math.ceil(first.total / PAGE_SIZE)
    write_rows(first.rows, part_path(checkpoint, 1))
    marker_path(checkpoint, 1).parent.mkdir(parents=True, exist_ok=True)
    marker_path(checkpoint, 1).write_text("ok", encoding="ascii")

    fetch_selected_pages(
        api=api,
        key=key,
        service=service,
        total=first.total,
        page_numbers=range(2, total_pages + 1),
        checkpoint=checkpoint,
        workers=workers,
    )

    done.write_text("ok", encoding="ascii")
    print(f"  {first.total:,}행 완료")


# -----------------------------------------------------------------------------
# 정리 및 모델용 테이블
# -----------------------------------------------------------------------------


def temporal_long(df: pd.DataFrame, dataset: Dataset, hours: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        return df
    assert dataset.value_prefix

    value_columns = [dataset.value_prefix + hour for hour in hours]
    missing = [column for column in value_columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"{dataset.key_name}: 시간 필드 누락: {missing}")

    id_columns = [
        column for column in df.columns if not column.startswith(dataset.value_prefix)
    ]
    output = df[id_columns + value_columns].melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="HOUR",
        value_name="VALUE",
    )
    output["HOUR"] = output["HOUR"].str[-2:]
    output["VALUE"] = pd.to_numeric(output["VALUE"], errors="coerce")
    return output


def unique_keys_with_optional_sequence(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> list[str]:
    keys = ["CRTR_DD", "RTE_ID", "DPTRE_STOPS_ID", "ARVL_STOPS_ID"]
    if "STOPS_SEQ" in left.columns and "STOPS_SEQ" in right.columns:
        keys.append("STOPS_SEQ")
    keys.append("HOUR")
    return keys


def build_model(
    time_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    passengers = time_frames["route_section_passengers"].rename(
        columns={"VALUE": "PASSENGERS"}
    )
    times = time_frames["route_section_time"].rename(
        columns={"VALUE": "AVG_OPERATION_TIME"}
    )
    operations = time_frames["station_route_operations"].rename(
        columns={"VALUE": "BUS_OPERATIONS"}
    )

    section_keys = unique_keys_with_optional_sequence(passengers, times)

    passengers["PASSENGERS"] = pd.to_numeric(
        passengers["PASSENGERS"], errors="coerce"
    )
    times["AVG_OPERATION_TIME"] = pd.to_numeric(
        times["AVG_OPERATION_TIME"], errors="coerce"
    )
    operations["BUS_OPERATIONS"] = pd.to_numeric(
        operations["BUS_OPERATIONS"], errors="coerce"
    )

    # 중복 키가 있으면 명시적으로 집계해 다대다 merge를 방지
    passengers_clean = (
        passengers[section_keys + ["PASSENGERS"]]
        .groupby(section_keys, as_index=False, dropna=False)["PASSENGERS"]
        .sum(min_count=1)
    )
    times_clean = (
        times[section_keys + ["AVG_OPERATION_TIME"]]
        .groupby(section_keys, as_index=False, dropna=False)["AVG_OPERATION_TIME"]
        .mean()
    )

    # 품질진단용 outer join
    qa = passengers_clean.merge(
        times_clean,
        on=section_keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    # 실제 모델용은 필수값이 모두 있는 inner join
    model = passengers_clean.merge(
        times_clean,
        on=section_keys,
        how="inner",
        validate="one_to_one",
    )

    operation_keys = ["CRTR_DD", "RTE_ID", "STOPS_ID", "HOUR"]
    if set(operation_keys).issubset(operations.columns):
        operation_clean = (
            operations[operation_keys + ["BUS_OPERATIONS"]]
            .groupby(operation_keys, as_index=False, dropna=False)["BUS_OPERATIONS"]
            .max()
        )
        model = model.merge(
            operation_clean,
            left_on=["CRTR_DD", "RTE_ID", "DPTRE_STOPS_ID", "HOUR"],
            right_on=["CRTR_DD", "RTE_ID", "STOPS_ID", "HOUR"],
            how="left",
            validate="many_to_one",
        )
    else:
        model["BUS_OPERATIONS"] = pd.NA

    model["PASSENGERS_PER_OPERATION"] = model["PASSENGERS"] / model[
        "BUS_OPERATIONS"
    ].replace(0, pd.NA)

    model = model.dropna(
        subset=["PASSENGERS", "AVG_OPERATION_TIME", "BUS_OPERATIONS"]
    ).copy()
    model = model[model["BUS_OPERATIONS"] > 0].copy()

    return model, qa


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    end = (
        datetime.strptime(args.end_date, "%Y%m%d").date()
        if args.end_date
        else date.today() - timedelta(days=5)
    )
    weekdays = parse_csv_ints(args.weekdays, 0, 6)
    hour_numbers = parse_csv_ints(args.hours, 0, 23)
    hours = tuple(f"{hour:02d}" for hour in hour_numbers)
    dates = target_dates(end, args.weeks, weekdays)

    if args.workers < 1:
        raise ValueError("--workers는 1 이상이어야 합니다.")

    if not dates:
        raise RuntimeError("수집 대상 날짜가 없습니다.")

    output = Path(args.output).resolve()
    checkpoint_root = output / "checkpoints"
    if args.fresh and checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)

    keys = load_keys(Path(args.config))
    api = SeoulAPI(args.pause)

    print("대상 날짜:", ", ".join(dates))
    print("대상 시간:", ", ".join(hours))
    print("동시 요청 수:", args.workers)

    services: dict[str, str] = {}
    print("\n[1/3] API 서비스와 스키마 검사")
    for dataset in DATASETS:
        service = choose_service(api, dataset, keys[dataset.key_name], dates)
        services[dataset.key_name] = service
        print(f"  OK {dataset.oa_id} {dataset.key_name}: {service}")

    if args.check_only:
        print("\n검사 완료")
        return

    print("\n[2/3] 다운로드")
    date_status: dict[str, dict[str, list[str]]] = {}

    for index, dataset in enumerate(DATASETS, start=1):
        print(f"\n({index}/{len(DATASETS)}) {dataset.oa_id} {dataset.key_name}")
        checkpoint = checkpoint_root / dataset.key_name

        if dataset.kind == "dated_parameter":
            found = collect_parameter_dates(
                api,
                dataset,
                keys[dataset.key_name],
                services[dataset.key_name],
                dates,
                checkpoint,
                args.workers,
            )
            date_status[dataset.key_name] = {
                "found": sorted(found),
                "missing": sorted(set(dates) - found),
            }
        elif dataset.kind == "dated_scan":
            found = collect_scan_dates(
                api,
                dataset,
                keys[dataset.key_name],
                services[dataset.key_name],
                dates,
                checkpoint,
                args.workers,
                full_fallback=not args.no_full_fallback,
                strict_dates=args.strict_dates,
            )
            date_status[dataset.key_name] = {
                "found": sorted(found),
                "missing": sorted(set(dates) - found),
            }
        else:
            collect_master(
                api,
                keys[dataset.key_name],
                services[dataset.key_name],
                checkpoint,
                args.workers,
            )

    if args.strict_dates:
        all_missing = {
            key: status["missing"]
            for key, status in date_status.items()
            if status["missing"]
        }
        if all_missing:
            raise RuntimeError(
                "목표 날짜 누락: "
                + json.dumps(all_missing, ensure_ascii=False)
            )

    print("\n[3/3] CSV 정리 및 결합")
    counts: dict[str, int] = {}
    time_frames: dict[str, pd.DataFrame] = {}

    for dataset in DATASETS:
        raw = read_parts(checkpoint_root / dataset.key_name)

        if dataset.kind != "master":
            cleaned = temporal_long(raw, dataset, hours)
            destination = output / "time" / f"{dataset.key_name}.csv"
            time_frames[dataset.key_name] = cleaned
        else:
            cleaned = raw
            destination = output / "master" / f"{dataset.key_name}.csv"

        atomic_csv(cleaned, destination)
        counts[dataset.key_name] = len(cleaned)
        print(f"  {destination}: {len(cleaned):,}행")

    if not args.skip_model_build:
        model, qa = build_model(time_frames)

        model_path = output / "model" / "model_segments.csv"
        qa_path = output / "model" / "model_segments_join_qa.csv"
        atomic_csv(model, model_path)
        atomic_csv(qa, qa_path)
        counts["model_segments"] = len(model)
        counts["model_segments_join_qa"] = len(qa)
        print(f"  {model_path}: {len(model):,}행")
        print(f"  {qa_path}: {len(qa):,}행")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "end_date": end.strftime("%Y%m%d"),
        "weeks": args.weeks,
        "target_dates": dates,
        "weekdays": list(weekdays),
        "hours": list(hours),
        "workers": args.workers,
        "services": services,
        "date_status": date_status,
        "row_counts": counts,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n완료: {output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다. 다음 실행 때 체크포인트에서 이어받습니다.")
        sys.exit(130)
    except Exception as exception:
        print(f"\n오류: {exception}", file=sys.stderr)
        sys.exit(1)