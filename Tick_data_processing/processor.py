from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

from tqdm import tqdm


DATE_FORMATS: Tuple[str, ...] = ("%m/%d/%Y",)
TIME_FORMATS: Tuple[str, ...] = ("%H:%M:%S.%f", "%H:%M:%S")


def _normalise_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace(".", "")


def _resolve_volume_column(headers: Iterable[str]) -> str:
    for header in headers:
        if _normalise_header(header) in {"volume", "vol", "vol_"}:
            return header
    raise ValueError("Could not identify a volume column in the provided CSV headers.")


def _resolve_price_column(headers: Iterable[str]) -> str:
    for header in headers:
        if _normalise_header(header) in {"price", "last", "close"}:
            return header
    raise ValueError("Could not identify a price column in the provided CSV headers.")


def _resolve_datetime_columns(headers: Iterable[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    date_key: Optional[str] = None
    time_key: Optional[str] = None
    timestamp_key: Optional[str] = None

    for header in headers:
        normalised = _normalise_header(header)
        if normalised in {"date"}:
            date_key = header
        elif normalised in {"time"}:
            time_key = header
        elif normalised in {"timestamp", "datetime"}:
            timestamp_key = header

    if timestamp_key is None and (date_key is None or time_key is None):
        raise ValueError("Expected either a combined timestamp column or both date and time columns.")

    return date_key, time_key, timestamp_key


def _parse_volume(raw_value: str) -> float:
    cleaned = raw_value.replace(",", "").strip()
    if not cleaned:
        raise ValueError("Empty volume value")
    return float(cleaned)


def _parse_datetime(date_value: str, time_value: str) -> datetime:
    for date_fmt in DATE_FORMATS:
        for time_fmt in TIME_FORMATS:
            try:
                return datetime.strptime(f"{date_value} {time_value}", f"{date_fmt} {time_fmt}")
            except ValueError:
                continue
    raise ValueError(f"Unable to parse datetime '{date_value} {time_value}'")


def _parse_timestamp(raw_value: str) -> datetime:
    cleaned = raw_value.strip()
    if not cleaned:
        raise ValueError("Empty timestamp value")

    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        pass

    for date_fmt in DATE_FORMATS:
        for time_fmt in TIME_FORMATS:
            try:
                return datetime.strptime(cleaned, f"{date_fmt} {time_fmt}")
            except ValueError:
                continue

    raise ValueError(f"Unable to parse timestamp '{raw_value}'")


def compute_volume_stats(csv_path: Path) -> Tuple[float, Dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing headers.")

        volume_key = _resolve_volume_column(reader.fieldnames)
        date_key, time_key, timestamp_key = _resolve_datetime_columns(reader.fieldnames)

        total_volume = 0.0
        monthly_totals: Dict[str, float] = defaultdict(float)

        for index, row in tqdm(enumerate(reader, start=1), desc="Aggregating volume", unit="rows"):
            raw_volume = row.get(volume_key, "")
            if not raw_volume:
                continue

            try:
                volume = _parse_volume(raw_volume)
            except ValueError as exc:
                raise ValueError(f"Row {index}: unable to parse volume value '{raw_volume}'") from exc

            if timestamp_key is not None:
                raw_timestamp = row.get(timestamp_key, "")
                if not raw_timestamp:
                    continue
                try:
                    timestamp = _parse_timestamp(raw_timestamp)
                except ValueError as exc:
                    raise ValueError(f"Row {index}: unable to parse timestamp '{raw_timestamp}'") from exc
            else:
                raw_date = row.get(date_key or "", "")
                raw_time = row.get(time_key or "", "")
                if not raw_date or not raw_time:
                    continue
                try:
                    timestamp = _parse_datetime(raw_date, raw_time)
                except ValueError as exc:
                    raise ValueError(
                        f"Row {index}: unable to parse date/time '{raw_date} {raw_time}'"
                    ) from exc

            month_key = f"{timestamp.year:04d}-{timestamp.month:02d}"
            monthly_totals[month_key] += volume
            total_volume += volume

    ordered_monthly = dict(sorted(monthly_totals.items()))
    return total_volume, ordered_monthly


def _floor_timestamp(dt: datetime, window_minutes: int) -> datetime:
    delta = timedelta(
        minutes=dt.minute % window_minutes,
        seconds=dt.second,
        microseconds=dt.microsecond,
    )
    return dt - delta


def _iter_ticks(
    csv_path: Path,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Iterator[Tuple[datetime, float, float]]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing headers.")

        volume_key = _resolve_volume_column(reader.fieldnames)
        price_key = _resolve_price_column(reader.fieldnames)
        date_key, time_key, timestamp_key = _resolve_datetime_columns(reader.fieldnames)

        for index, row in tqdm(enumerate(reader, start=1), desc="Reading ticks", unit="rows"):
            raw_volume = row.get(volume_key, "")
            raw_price = row.get(price_key, "")
            if not raw_volume or not raw_price:
                continue

            try:
                volume = _parse_volume(raw_volume)
                price = float(raw_price)
            except ValueError as exc:
                raise ValueError(
                    f"Row {index}: unable to parse price/volume '{raw_price}' / '{raw_volume}'"
                ) from exc

            if timestamp_key is not None:
                raw_timestamp = row.get(timestamp_key, "")
                if not raw_timestamp:
                    continue
                try:
                    timestamp = _parse_timestamp(raw_timestamp)
                except ValueError as exc:
                    raise ValueError(f"Row {index}: unable to parse timestamp '{raw_timestamp}'") from exc
            else:
                raw_date = row.get(date_key or "", "")
                raw_time = row.get(time_key or "", "")
                if not raw_date or not raw_time:
                    continue
                try:
                    timestamp = _parse_datetime(raw_date, raw_time)
                except ValueError as exc:
                    raise ValueError(
                        f"Row {index}: unable to parse date/time '{raw_date} {raw_time}'"
                    ) from exc

            if start is not None and timestamp < start:
                continue
            if end is not None and timestamp > end:
                break

            yield timestamp, price, volume


def aggregate_ticks_to_ohlcv(
    csv_path: Path,
    output_path: Path,
    *,
    window_minutes: int = 5,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        

        current_bucket_start: Optional[datetime] = None
        open_price = high_price = low_price = close_price = None
        cumulative_volume = 0.0

        for timestamp, price, volume in _iter_ticks(csv_path, start=start, end=end):
            bucket_start = _floor_timestamp(timestamp, window_minutes)

            if current_bucket_start is None:
                current_bucket_start = bucket_start
                open_price = high_price = low_price = close_price = price
                cumulative_volume = volume
                continue

            if bucket_start != current_bucket_start:
                writer.writerow(
                    [
                        current_bucket_start.strftime("%Y-%m-%d %H:%M:%S"),
                        f"{open_price:.6f}",
                        f"{high_price:.6f}",
                        f"{low_price:.6f}",
                        f"{close_price:.6f}",
                        f"{cumulative_volume:.6f}",
                    ]
                )

                current_bucket_start = bucket_start
                open_price = high_price = low_price = close_price = price
                cumulative_volume = volume
                continue

            high_price = max(high_price, price) if high_price is not None else price
            low_price = min(low_price, price) if low_price is not None else price
            close_price = price
            cumulative_volume += volume

        if current_bucket_start is not None and open_price is not None:
            writer.writerow(
                [
                    current_bucket_start.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{open_price:.6f}",
                    f"{high_price:.6f}",
                    f"{low_price:.6f}",
                    f"{close_price:.6f}",
                    f"{cumulative_volume:.6f}",
                ]
            )


def main() -> None:
    csv_path = Path(__file__).resolve().parent / "SP.csv"
    output_path = Path(__file__).resolve().parent / "SP_5min.csv"
    start = datetime(2003, 7, 1)
    end = datetime(2014, 1, 31, 23, 59, 59)
    aggregate_ticks_to_ohlcv(csv_path, output_path, window_minutes=5, start=start, end=end)
    print(f"Wrote aggregated OHLCV data to {output_path}")


if __name__ == "__main__":
    main()

