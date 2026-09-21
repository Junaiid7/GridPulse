"""Deterministic synthetic electricity + weather payloads for OFFLINE TESTS.

IMPORTANT (Phase 4B data-integrity rule): everything here is a **hand-crafted
synthetic fixture** that exists only to exercise the real pipeline / forecast
code paths offline. It is NEVER real, downloaded, or representative data, and
it must never be combined with live data in a way that could be mistaken for a
real historical dataset. Any benchmark produced from these payloads is
labelled ``FIXTURE-VERIFIED``.

The generation/load/prices XML parse through the real ENTSO-E parsers and the
weather JSON through the real Open-Meteo parser, so the offline run exercises
the identical Bronze -> Silver -> Gold -> Features code path as a live run.

Design (deterministic, no random):
- Window defaults to 2024-01-15T00:00Z .. 2024-03-15T00:00Z (~59 full days,
  no DST inside: the Europe/Amsterdam transition is 2024-03-31, outside).
- The **residual** signal is chosen first (strong 24h diurnal + a 7-day
  component) so the seasonal-naive 24h baseline is a meaningful challenger.
- wind = B18 (offshore) + B19 (onshore), solar = B16; load is then built as
  ``load = residual + wind + solar`` so ``residual_load`` matches the repo's
  definition (``residual_load = load - wind - solar``).
- All series are hourly (PT60M), every hour of the window, no gaps/duplicates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from gridpulse.config import Settings
from gridpulse.ingestion.common.errors import NoDataError
from gridpulse.ingestion.common.models import FetchResult
from gridpulse.pipeline.runner import PipelineRun, run_pipeline

UTC = UTC
DEFAULT_START = datetime(2024, 1, 15, tzinfo=UTC)
DEFAULT_END = datetime(2024, 3, 15, tzinfo=UTC)

_NS = "urn:iec62325.351:tc57wg16:451-1:glmarketdocument:7:0"


# ----------------------------------------------------------------- signals
def _local_hour(utc_hour: int) -> int:
    """Amsterdam local hour our window (UTC+1 winter, no DST crossed)."""
    return (utc_hour + 1) % 24


def _signals(t: datetime) -> dict:
    """Deterministic hourly signals for timestamp *t* (all floats)."""
    h = int(t.timestamp() // 3600)  # hour index since epoch
    utc_hr = t.hour
    ll = _local_hour(utc_hr)

    residual = (
        2800.0
        + 900.0 * math.cos(2 * math.pi * (ll - 19.5) / 24)
        + 300.0 * math.sin(2 * math.pi * (h % 168) / 168 + 1.3)
    )
    solar = 0.0 if ll < 7 or ll >= 18 else 1500.0 * math.sin(math.pi * (ll - 7) / 11)
    wind_offshore = (
        700.0
        + 250.0 * math.sin(2 * math.pi * h / 72)
        + 100.0 * math.sin(2 * math.pi * ll / 24)
    )
    wind_onshore = (
        900.0
        + 300.0 * math.sin(2 * math.pi * h / 96 + 0.7)
        + 120.0 * math.sin(2 * math.pi * ll / 48)
    )
    price = (
        38.0
        + 22.0 * math.cos(2 * math.pi * (ll - 19) / 24)
        + 6.0 * math.sin(2 * math.pi * h / 168)
    )

    temp = (
        5.5
        + 4.0 * math.cos(2 * math.pi * (ll - 15) / 24)
        + 2.0 * math.sin(2 * math.pi * h / (24 * 14))
    )
    humidity = 72.0 + 15.0 * math.cos(2 * math.pi * (ll - 5) / 24)
    ws10 = (
        4.8
        + 2.0 * math.sin(2 * math.pi * ll / 24)
        + 1.0 * math.sin(2 * math.pi * h / 72)
    )
    ws100 = (
        8.2
        + 2.5 * math.sin(2 * math.pi * ll / 24)
        + 1.5 * math.sin(2 * math.pi * h / 96)
    )
    wdir = 210.0 + 60.0 * math.sin(2 * math.pi * h / 96)
    shw = 0.0 if ll < 6 or ll >= 18 else 420.0 * math.sin(math.pi * (ll - 6) / 12)
    dni = (
        0.0 if ll < 6 or ll >= 18 else 520.0 * math.sin(math.pi * (ll - 6) / 12) ** 1.2
    )
    dif = 0.0 if ll < 6 or ll >= 18 else 90.0 + 60.0 * math.sin(math.pi * (ll - 6) / 12)
    cloud = 45.0 + 35.0 * math.sin(2 * math.pi * (h + 2000) / 240)
    precip = max(0.0, 1.2 * math.sin(2 * math.pi * (h % 43) / 43) - 0.6)

    return {
        "residual_load_mw": residual,
        "solar_mw": solar,
        "wind_offshore_mw": wind_offshore,
        "wind_onshore_mw": wind_onshore,
        "load_mw": residual + solar + wind_offshore + wind_onshore,
        "day_ahead_price_eur_mwh": price,
        "temperature_2m": temp,
        "relative_humidity_2m": humidity,
        "wind_speed_10m": ws10,
        "wind_speed_100m": ws100,
        "wind_direction_100m": wdir,
        "shortwave_radiation": shw,
        "direct_normal_irradiance": dni,
        "diffuse_radiation": dif,
        "cloud_cover": cloud,
        "precipitation": precip,
    }


def _hourly_timestamps(start: datetime, end: datetime) -> list[datetime]:
    out = []
    cursor = start
    while cursor < end:
        out.append(cursor)
        cursor += timedelta(hours=1)
    return out


# ------------------------------------------------------------------ XML
def _points_xml(values: list[float], value_tag: str) -> str:
    lines = [
        f"<Point><position>{i}</position><{value_tag}>{v:.1f}</{value_tag}></Point>"
        for i, v in enumerate(values, start=1)
    ]
    return "\n          ".join(lines)


def _time_series_block(
    ts_id: str, start: datetime, end: datetime, values: list[float], value_tag: str
) -> str:
    return (
        "    <TimeSeries>\n"
        f"      <mRID>{ts_id}</mRID>\n"
        "      <businessType>B22</businessType>\n"
        "      <curveType>A01</curveType>\n"
        '      <outBiddingZone_Domain.mRID codingScheme="A01">10YNL----------L</outBiddingZone_Domain.mRID>\n'
        f"      <quantity_measure_unit.name>MAW</quantity_measure_unit.name>\n"
        "      <Period>\n"
        "        <timeInterval>\n"
        f"          <start>{start:%Y-%m-%dT%H:%M}Z</start>\n"
        f"          <end>{end:%Y-%m-%dT%H:%M}Z</end>\n"
        "        </timeInterval>\n"
        "        <resolution>PT60M</resolution>\n"
        f"          {_points_xml(values, value_tag)}\n"
        "      </Period>\n"
        "    </TimeSeries>"
    )


def _load_xml(start: datetime, end: datetime, per_hour: list[dict]) -> bytes:
    values = [s["load_mw"] for s in per_hour]
    body = _time_series_block("ts-load", start, end, values, "quantity")
    return _wrap_xml(body, doc_type="A65")


def _generation_xml(start: datetime, end: datetime, per_hour: list[dict]) -> bytes:
    blocks = [
        ("ts-solar", [s["solar_mw"] for s in per_hour]),
        ("ts-wind-offshore", [s["wind_offshore_mw"] for s in per_hour]),
        ("ts-wind-onshore", [s["wind_onshore_mw"] for s in per_hour]),
    ]
    built = []
    for ts_id, values in blocks:
        raw = _time_series_block(ts_id, start, end, values, "quantity")
        # Inject the PSR element right after the curveType line, mirroring fixture.
        raw = raw.replace(
            "      <curveType>A01</curveType>\n",
            "      <curveType>A01</curveType>\n      <MktPSRType>\n        <psrType>"
            + _PSR_FOR_TS[ts_id]
            + "</psrType>\n      </MktPSRType>\n",
            1,
        )
        built.append(raw)
    return _wrap_xml("\n".join(built), doc_type="A75")


_PSR_FOR_TS = {"ts-solar": "B16", "ts-wind-offshore": "B18", "ts-wind-onshore": "B19"}


def _prices_xml(start: datetime, end: datetime, per_hour: list[dict]) -> bytes:
    values = [s["day_ahead_price_eur_mwh"] for s in per_hour]
    body = _time_series_block("ts-price", start, end, values, "price.amount")
    return _wrap_xml(body, doc_type="A44")


def _wrap_xml(body: str, *, doc_type: str) -> bytes:
    return (
        "<!-- SYNTHETIC FIXTURE (offline tests only; never real ENTSO-E data) -->\n"
        f'<GL_MarketDocument xmlns="{_NS}">\n'
        "  <mRID>synthetic-fixture-0003</mRID>\n"
        "  <revisionNumber>1</revisionNumber>\n"
        f"  <type>{doc_type}</type>\n"
        "  <process.processType>A16</process.processType>\n"
        f"{body}\n"
        "</GL_MarketDocument>"
    ).encode()


# ------------------------------------------------------------------ weather
def _weather_json(start: datetime, end: datetime, per_hour: list[dict]) -> bytes:
    variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_100m",
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "cloud_cover",
        "precipitation",
    ]
    units = {
        "temperature_2m": "°C",
        "relative_humidity_2m": "%",
        "wind_speed_10m": "m/s",
        "wind_speed_100m": "m/s",
        "wind_direction_100m": "°",
        "shortwave_radiation": "W/m²",
        "direct_normal_irradiance": "W/m²",
        "diffuse_radiation": "W/m²",
        "cloud_cover": "%",
        "precipitation": "mm",
    }
    payload = {
        "latitude": 52.21,
        "longitude": 5.29,
        "elevation": 5.0,
        "generationtime_ms": 0.05,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "timezone_abbreviation": "GMT",
        "hourly_units": {"time": "iso8601", **units},
        "hourly": {
            "time": [f"{t:%Y-%m-%dT%H:%M}" for t in _hourly_timestamps(start, end)],
            **{var: [round(s[var], 2) for s in per_hour] for var in variables},
        },
    }
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


# -------------------------------------------------------------------- bundle
@dataclass(frozen=True)
class SyntheticFixtures:
    start: datetime
    end: datetime
    payloads: dict  # bytes per kind
    per_hour: list[dict]  # one dict of signals per hourly timestamp
    timestamps: list[datetime]

    def signals_at(self, t: datetime) -> dict:
        return self.per_hour[self.timestamps.index(t)]


def build_synthetic_window(
    window_start: datetime = DEFAULT_START,
    window_end: datetime = DEFAULT_END,
) -> SyntheticFixtures:
    """Build the deterministic synthetic fixture bundle for ``[start, end)``."""
    per_hour = [_signals(t) for t in _hourly_timestamps(window_start, window_end)]
    return SyntheticFixtures(
        start=window_start,
        end=window_end,
        payloads={
            "load": _load_xml(window_start, window_end, per_hour),
            "generation": _generation_xml(window_start, window_end, per_hour),
            "prices": _prices_xml(window_start, window_end, per_hour),
            "weather": _weather_json(window_start, window_end, per_hour),
        },
        per_hour=per_hour,
        timestamps=_hourly_timestamps(window_start, window_end),
    )


# ------------------------------------------------------------------ clients
class SyntheticEntsoeClient:
    """Fake ENTSO-E client serving the synthetic bundle via the real interface."""

    def __init__(self, fixtures: SyntheticFixtures) -> None:
        self._f = fixtures

    def _result(self, entity: str, payload: bytes) -> FetchResult:
        return FetchResult(
            source="entsoe",
            entity=entity,
            start=self._f.start,
            end=self._f.end,
            retrieved_at=self._f.end + timedelta(hours=6),
            payload=payload,
            content_type="application/xml",
            encoding="utf-8",
            url="https://web-api.tp.entsoe.eu/api",
            timezone="UTC",
        )

    def fetch_load(self, area, start, end) -> list[FetchResult]:
        return [self._result("actual-total-load", self._f.payloads["load"])]

    def fetch_generation(self, area, start, end) -> list[FetchResult]:
        return [
            self._result("actual-generation-by-type", self._f.payloads["generation"])
        ]

    def fetch_day_ahead_prices(self, area, start, end) -> list[FetchResult]:
        return [self._result("dayahead-prices", self._f.payloads["prices"])]

    def fetch_imbalance_prices(self, area, start, end) -> list[FetchResult]:
        raise NoDataError(
            "synthetic: NL imbalance intentionally UNAVAILABLE in fixtures"
        )


class SyntheticOpenMeteoClient:
    """Fake Open-Meteo client: same synthetic weather payload for every location."""

    def __init__(self, fixtures: SyntheticFixtures) -> None:
        self._f = fixtures

    def fetch_historical(self, location, start_date, end_date) -> list[FetchResult]:
        start = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
        end = datetime.combine(end_date, datetime.min.time(), tzinfo=UTC) + timedelta(
            days=1
        )
        return [
            FetchResult(
                source="open-meteo",
                entity="historical-weather",
                start=start,
                end=end,
                retrieved_at=self._f.end + timedelta(hours=6),
                payload=self._f.payloads["weather"],
                content_type="application/json",
                encoding="utf-8",
                url="https://archive-api.open-meteo.com/v1/archive",
                identifiers=(f"location:{location.name}",),
                timezone="UTC",
                metadata={"location": location.name, "variables": "hourly"},
            )
        ]


class SyntheticImbalanceSource:
    """Honest UNAVAILABLE imbalance provider (real NL feed often returns no data)."""

    availability: dict = {"verified": "fixture", "available": False}
    name = "synthetic-imbalance-unavailable"

    def fetch(self, area, start, end) -> list[FetchResult]:
        raise NoDataError("synthetic imbalance: recorded unavailable by design")


# ------------------------------------------------------------------ pipeline
def run_synthetic_pipeline(
    data_dir,
    *,
    window_start: datetime = DEFAULT_START,
    window_end: datetime = DEFAULT_END,
) -> PipelineRun:
    """Run the REAL pipeline end-to-end on synthetic fixtures into ``data_dir``.

    ``data_dir`` is a scratch directory (tests pass tmp_path; the demo script
    uses a temporary dir). Returns the live ``PipelineRun`` so callers can read
    ``run.features.csv_path`` etc. The imbalance source is the honest
    UNAVAILABLE provider; everything electricity is FIXTURE-VERIFIED.
    """
    fixtures = build_synthetic_window(window_start, window_end)
    return run_pipeline(
        start=fixtures.start,
        end=fixtures.end,
        settings=Settings(project_root=data_dir, data_root=data_dir),
        open_meteo_client=SyntheticOpenMeteoClient(fixtures),
        entsoe_client=SyntheticEntsoeClient(fixtures),
        imbalance_source=SyntheticImbalanceSource(),
        report_dir=data_dir / "reports",
    )


def features_rows(run: PipelineRun):
    """Read the pipeline's feature-table rows (as ``read_table`` dicts)."""
    from pathlib import Path

    from gridpulse.transformation.csvio import read_table

    return read_table(Path(run.features.csv_path))


__all__ = [
    "SyntheticFixtures",
    "build_synthetic_window",
    "SyntheticEntsoeClient",
    "SyntheticOpenMeteoClient",
    "SyntheticImbalanceSource",
    "run_synthetic_pipeline",
    "features_rows",
    "DEFAULT_START",
    "DEFAULT_END",
]
