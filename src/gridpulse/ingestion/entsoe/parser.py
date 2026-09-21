"""ENTSO-E XML parsing into normalised UTC-aware time series.

The REST API returns CIM/XML (and ZIPs of XML for imbalance prices). This
parser is namespace-agnostic: it matches element *local names* (XML element
names may contain dots, e.g. ``price.amount``).

Reference behaviour (positions, resolution, curve types) mirrors the widely
used entsoe-py parsers.
"""

from __future__ import annotations

import io
import logging
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

from ..common.errors import EmptyResponseError, MalformedResponseError, NoDataError
from ..common.models import DataPoint, TimeSeries
from .mappings import RESOLUTION_MINUTES

LOGGER = logging.getLogger("gridpulse.ingestion.entsoe.parser")

_NO_MATCHING = "No matching data found"


def _local(tag: str) -> str:
    """Element local name, stripping any XML namespace."""
    return tag.rsplit("}", 1)[-1]


def _parse_datetime(text: str) -> datetime:
    """Parse an ENTSO-E datetime (typically ``...T00:00Z``) as aware UTC."""
    value = text.strip().rstrip("Z")
    if value.endswith(("+00:00", "-00:00")):
        value = value[:-6]
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise MalformedResponseError(f"unrecognised timestamp {text!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _resolution_to_minutes(resolution: str) -> int | None:
    return RESOLUTION_MINUTES.get(resolution.strip())


def iter_timeseries(root: ET.Element) -> Iterator[ET.Element]:
    """Yield ``TimeSeries`` elements regardless of namespace."""
    for elem in root.iter():
        if _local(elem.tag) == "TimeSeries":
            yield elem


_KNOWN_METADATA_ELEMENTS = {
    "in_Domain.mRID",
    "out_Domain.mRID",
    "inBiddingZone_Domain.mRID",
    "outBiddingZone_Domain.mRID",
    "controlArea_Domain.mRID",
    "MktPSRType.psrType",
    "businessType",
    "curveType",
    "quantity_measure_unit.name",
    "price_measure_unit.name",
    "currency",
}


def _series_metadata(timeseries: ET.Element) -> dict[str, str]:
    """Capture identifiers carried beside the points."""
    meta: dict[str, str] = {}
    for elem in timeseries.iter():
        name = _local(elem.tag)
        if name == "psrType":
            # Nested under <MktPSRType>; key it as the PSR type for generation.
            meta["MktPSRType.psrType"] = (elem.text or "").strip()
        elif name in _KNOWN_METADATA_ELEMENTS:
            meta[name] = (elem.text or "").strip()
    return meta


def _points_from_period(period: ET.Element, value_label: str, curve_type: str) -> list[DataPoint]:
    start_elem = end_elem = res_elem = None
    for elem in period.iter():
        name = _local(elem.tag)
        if name == "start" and start_elem is None:
            start_elem = elem
        elif name == "end" and end_elem is None:
            end_elem = elem
        elif name == "resolution" and res_elem is None:
            res_elem = elem
    if start_elem is None or end_elem is None or res_elem is None:
        raise MalformedResponseError("Period missing timeInterval or resolution")

    start = _parse_datetime(start_elem.text or "")
    end = _parse_datetime(end_elem.text or "")
    resolution = _resolution_to_minutes(res_elem.text or "")
    if resolution is None:
        raise MalformedResponseError(f"unsupported resolution {res_elem.text!r}")
    step = timedelta(minutes=resolution)

    points_by_pos: dict[int, float] = {}
    for point in period:
        if _local(point.tag) != "Point":
            continue
        position: int | None = None
        value: float | None = None
        for child in point:
            n = _local(child.tag)
            if n == "position":
                position = int(child.text or "0")
            elif n == value_label:
                value = float((child.text or "").replace(",", ""))
        if position is not None and value is not None:
            points_by_pos[position] = value

    if not points_by_pos:
        return []

    points: list[DataPoint] = []
    if curve_type == "A03":
        # Missing positions repeat the last known value (forward fill).
        position = 1
        last: float | None = None
        while True:
            ts = start + (position - 1) * step
            if ts >= end:
                break
            if position in points_by_pos:
                last = points_by_pos[position]
            if last is not None:
                points.append(DataPoint(ts, last))
            position += 1
    else:
        for position in sorted(points_by_pos):
            ts = start + (position - 1) * step
            if ts < end:
                points.append(DataPoint(ts, points_by_pos[position]))

    return points


def _build_series(
    timeseries: ET.Element,
    *,
    source: str,
    entity: str,
    value_label: str,
    unit: str,
) -> TimeSeries:
    meta = _series_metadata(timeseries)
    points: list[DataPoint] = []
    curve_type = meta.get("curveType") or "A01"
    for period in timeseries.iter():
        if _local(period.tag) != "Period":
            continue
        points.extend(_points_from_period(period, value_label, curve_type))

    points.sort(key=lambda p: p.timestamp)
    return TimeSeries(
        source=source,
        entity=entity,
        unit=unit,
        points=tuple(points),
        resolution_minutes=_infer_resolution(points),
        tz="UTC",
        metadata=meta,
    )


def _infer_resolution(points: list[DataPoint]) -> int | None:
    """Infer the median step between consecutive points as minutes."""
    if len(points) < 2:
        return None
    deltas = sorted(
        int((b.timestamp - a.timestamp).total_seconds() / 60)
        for a, b in zip(points, points[1:])
        if b.timestamp > a.timestamp
    )
    return deltas[len(deltas) // 2] if deltas else None


def parse_xml_to_series(
    payload: bytes,
    *,
    source: str = "entsoe",
    entity: str,
    value_label: str,
    unit: str,
) -> list[TimeSeries]:
    """Core parser used by all ENTSO-E document types."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise MalformedResponseError(f"malformed XML: {exc}") from exc

    text = payload.decode("utf-8", errors="replace")
    if _NO_MATCHING in text:
        raise NoDataError(f"{entity}: ENTSO-E reported 'No matching data found'")

    series = [
        _build_series(ts, source=source, entity=entity, value_label=value_label, unit=unit)
        for ts in iter_timeseries(root)
    ]
    points = [p for s in series for p in s.points]
    if not series or not points:
        raise EmptyResponseError(f"{entity}: no <TimeSeries> data in response")
    return series


def _merge(series_list: list[TimeSeries], entity: str) -> TimeSeries:
    points = [p for s in series_list for p in s.points]
    points.sort(key=lambda p: p.timestamp)
    if not points:
        raise EmptyResponseError(f"{entity}: no points across parsed time series")
    merged_meta: dict[str, str] = {}
    for s in series_list:
        merged_meta.update(s.metadata)
    return TimeSeries(
        source=series_list[0].source,
        entity=entity,
        unit=series_list[0].unit,
        points=tuple(points),
        resolution_minutes=_infer_resolution(points),
        tz="UTC",
        metadata=merged_meta,
    )


def parse_load(payload: bytes, *, entity: str = "actual-total-load") -> TimeSeries:
    """Actual / forecast total load (documentType A65), unit MW."""
    return _merge(parse_xml_to_series(payload, entity=entity, value_label="quantity", unit="MW"), entity)


def parse_day_ahead_prices(payload: bytes, *, entity: str = "dayahead-prices") -> TimeSeries:
    """Day-ahead prices (documentType A44), unit EUR/MWh."""
    return _merge(parse_xml_to_series(payload, entity=entity, value_label="price.amount", unit="EUR/MWh"), entity)


def parse_generation(payload: bytes, *, entity: str = "actual-generation-by-type") -> dict[str, TimeSeries]:
    """Generation per production type (documentType A75), unit MW.

    Returns a mapping of PSR type code (e.g. ``B16``) to its merged time
    series (multiple TimeSeries blocks can share one PSR type).
    """
    series_list = parse_xml_to_series(payload, entity=entity, value_label="quantity", unit="MW")
    merged: dict[str, list[TimeSeries]] = {}
    for s in series_list:
        psr = s.metadata.get("MktPSRType.psrType") or "unknown"
        merged.setdefault(psr, []).append(s)
    return {psr: _merge(blocks, entity) for psr, blocks in merged.items()}


def parse_crossborder_flows(payload: bytes, *, entity: str = "cross-border-physical-flows") -> TimeSeries:
    """Cross-border physical flows (documentType A11), unit MW."""
    return _merge(parse_xml_to_series(payload, entity=entity, value_label="quantity", unit="MW"), entity)


def parse_imbalance_prices(payload: bytes, *, entity: str = "imbalance-prices") -> TimeSeries:
    """Imbalance prices (documentType A85).

    The ENTSO-E endpoint returns a ZIP archive containing per-TSO XML files;
    a single XML document is also accepted. Unit: EUR/MWh.
    """
    if zipfile.is_zipfile(io.BytesIO(payload)):
        series_list: list[TimeSeries] = []
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            xml_names = [n for n in archive.namelist() if n.lower().endswith(".xml")]
            if not xml_names:
                raise EmptyResponseError(f"{entity}: ZIP contained no XML files")
            for name in xml_names:
                series_list.extend(
                    parse_xml_to_series(
                        archive.read(name),
                        entity=f"{entity}:{name}",
                        value_label="imbalance_price.amount",
                        unit="EUR/MWh",
                    )
                )
        return _merge(series_list, entity)

    return _merge(parse_xml_to_series(payload, entity=entity, value_label="imbalance_price.amount", unit="EUR/MWh"), entity)