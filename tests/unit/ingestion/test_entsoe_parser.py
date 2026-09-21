"""Tests for the ENTSO-E XML parser: timestamps, resolution, units, curve
type A03 forward-fill, malformed/empty responses, and ZIP imbalance parsing.
All inputs are inline synthetic XML strings.
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC

import pytest

from gridpulse.ingestion.common.errors import (
    EmptyResponseError,
    MalformedResponseError,
    NoDataError,
)
from gridpulse.ingestion.entsoe.parser import (
    parse_crossborder_flows,
    parse_day_ahead_prices,
    parse_generation,
    parse_imbalance_prices,
    parse_load,
)

UTC = UTC

LOAD_XML = b"""<GL_MarketDocument>
  <TimeSeries>
    <mRID>1</mRID>
    <businessType>B73</businessType>
    <curveType>A01</curveType>
    <outBiddingZone_Domain.mRID codingScheme="A01">10YNL----------L</outBiddingZone_Domain.mRID>
    <quantity_measure_unit.name>MAW</quantity_measure_unit.name>
    <Period>
      <timeInterval><start>2024-01-01T00:00Z</start><end>2024-01-01T01:00Z</end></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>4500</quantity></Point>
      <Point><position>2</position><quantity>4550</quantity></Point>
      <Point><position>3</position><quantity>4600</quantity></Point>
      <Point><position>4</position><quantity>4700</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""

PRICE_XML = b"""<GL_MarketDocument>
  <TimeSeries>
    <curveType>A01</curveType>
    <Period>
      <timeInterval><start>2024-01-01T00:00Z</start><end>2024-01-01T03:00Z</end></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>45.2</price.amount></Point>
      <Point><position>2</position><price.amount>48.7</price.amount></Point>
      <Point><position>3</position><price.amount>55.0</price.amount></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""

FLOW_XML = b"""<GL_MarketDocument>
  <TimeSeries>
    <curveType>A01</curveType>
    <out_Domain.mRID codingScheme="A01">10YNL----------L</out_Domain.mRID>
    <in_Domain.mRID codingScheme="A01">10Y1001A1001A82H</in_Domain.mRID>
    <Period>
      <timeInterval><start>2024-01-01T00:00Z</start><end>2024-01-01T02:00Z</end></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>-250.0</quantity></Point>
      <Point><position>2</position><quantity>-310.5</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""

A03_XML = b"""<GL_MarketDocument>
  <TimeSeries>
    <curveType>A03</curveType>
    <Period>
      <timeInterval><start>2024-01-01T00:00Z</start><end>2024-01-01T03:00Z</end></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>1.0</quantity></Point>
      <Point><position>3</position><quantity>3.0</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""

GENERATION_XML = b"""<GL_MarketDocument>
  <TimeSeries>
    <curveType>A01</curveType>
    <MktPSRType><psrType>B16</psrType></MktPSRType>
    <Period>
      <timeInterval><start>2024-01-01T00:00Z</start><end>2024-01-01T02:00Z</end></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>2500.0</quantity></Point>
      <Point><position>2</position><quantity>2700.0</quantity></Point>
    </Period>
  </TimeSeries>
  <TimeSeries>
    <curveType>A01</curveType>
    <MktPSRType><psrType>B04</psrType></MktPSRType>
    <Period>
      <timeInterval><start>2024-01-01T00:00Z</start><end>2024-01-01T02:00Z</end></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>300.0</quantity></Point>
      <Point><position>2</position><quantity>310.0</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""


def test_parse_load_timestamps_are_utc_aware_and_resolution_15min() -> None:
    series = parse_load(LOAD_XML)
    assert series.unit == "MW"
    assert series.resolution_minutes == 15
    assert len(series.points) == 4
    assert series.points[0].timestamp.isoformat() == "2024-01-01T00:00:00+00:00"
    assert series.points[1].timestamp.isoformat() == "2024-01-01T00:15:00+00:00"
    assert all(p.timestamp.tzinfo is UTC for p in series.points)
    assert [p.value for p in series.points] == [4500.0, 4550.0, 4600.0, 4700.0]


def test_parse_prices_positions_map_to_hours() -> None:
    series = parse_day_ahead_prices(PRICE_XML)
    assert series.unit == "EUR/MWh"
    assert [p.value for p in series.points] == [45.2, 48.7, 55.0]
    assert series.points[2].timestamp.isoformat() == "2024-01-01T02:00:00+00:00"


def test_parse_flows_keeps_sign() -> None:
    series = parse_crossborder_flows(FLOW_XML)
    assert series.unit == "MW"
    assert series.values == [-250.0, -310.5]  # export is signed negative consistently


def test_curve_type_a03_forward_fills_missing_positions() -> None:
    series = parse_load(A03_XML)
    assert len(series.points) == 3
    assert [p.value for p in series.points] == [1.0, 1.0, 3.0]  # position 2 got 1.0


def test_parse_generation_splits_by_psr_type() -> None:
    by_type = parse_generation(GENERATION_XML)
    assert set(by_type) == {"B16", "B04"}
    assert by_type["B16"].values == [2500.0, 2700.0]
    assert by_type["B04"].values == [300.0, 310.0]
    assert by_type["B16"].metadata["MktPSRType.psrType"] == "B16"


def test_malformed_xml_raises() -> None:
    with pytest.raises(MalformedResponseError):
        parse_load(b"<GL_MarketDocument><unclosed>")


def test_no_timeseries_raises_empty() -> None:
    with pytest.raises(EmptyResponseError):
        parse_load(b"<GL_MarketDocument></GL_MarketDocument>")


def test_no_matching_data_text_raises_no_data() -> None:
    with pytest.raises(NoDataError):
        parse_load(b"<ack><text>No matching data found</text></ack>")


def test_parse_imbalance_prices_from_zip() -> None:
    inner = b"""<GL_MarketDocument>
      <TimeSeries>
        <curveType>A01</curveType>
        <Period>
          <timeInterval><start>2024-01-01T00:00Z</start><end>2024-01-01T02:00Z</end></timeInterval>
          <resolution>PT60M</resolution>
          <Point><position>1</position><imbalance_price.amount>-10.5</imbalance_price.amount></Point>
          <Point><position>2</position><imbalance_price.amount>-4.0</imbalance_price.amount></Point>
        </Period>
      </TimeSeries>
    </GL_MarketDocument>"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("tennet_imbalance.xml", inner)
    series = parse_imbalance_prices(buffer.getvalue())
    assert series.unit == "EUR/MWh"
    assert series.values == [-10.5, -4.0]


def test_parse_imbalance_prices_from_single_xml() -> None:
    single = b"""<GL_MarketDocument>
      <TimeSeries>
        <curveType>A01</curveType>
        <Period>
          <timeInterval><start>2024-01-01T00:00Z</start><end>2024-01-01T01:00Z</end></timeInterval>
          <resolution>PT60M</resolution>
          <Point><position>1</position><imbalance_price.amount>3.0</imbalance_price.amount></Point>
        </Period>
      </TimeSeries>
    </GL_MarketDocument>"""
    assert parse_imbalance_prices(single).values == [3.0]


def test_parse_zip_with_no_xml_raises_empty() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", b"hello")
    with pytest.raises(EmptyResponseError):
        parse_imbalance_prices(buffer.getvalue())
