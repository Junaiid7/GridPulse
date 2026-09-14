# GridPulse Data Sources

Detailed reference for every external data source used by the ingestion
layer. This document records what was **verified against live APIs** and what
remains **uncertain or unverified**.

## ENTSO-E Transparency Platform

**Base URL:** `https://web-api.tp.entsoe.eu/api`

**Authentication:** `securityToken` query parameter. The token is read from
the `ENTSOE_API_KEY` environment variable and must never be hardcoded.

**Response format:** CIM/XML (with ZIP archives for some document types like
imbalance prices A85).

**Verified behaviour (live testing, September 2026):**

- Omitting the token returns HTTP 401 with an XML `<Ack>` containing
  "Authentication failed".
- Requesting more elements than the server limit returns HTTP 400 with
  `"amount of requested data exceeds allowed limit"`. The client
  automatically halves the time window on this error (up to 10 levels deep).
- A valid request with no data for the requested period returns HTTP 200 with
  body text `"No matching data found"` — treated as an empty result, not an
  error.
- `periodStart` / `periodEnd` format: `YYYYMMDDHH00` (UTC, hour-granularity,
  minute always `00`).

**Rate limits:** No published rate limit was found. The client includes
configurable retries with exponential backoff for transient HTTP errors
(408, 425, 429, 500, 502, 503, 504).

### Document Types and Parameters

| Dataset | docType | processType | Parameters | Resolution |
|---------|---------|-------------|------------|------------|
| Actual total load | A65 | A16 | `out_Domain`, `outBiddingZone_Domain` | PT15M or PT60M |
| Day-ahead load forecast | A65 | A01 | `out_Domain`, `outBiddingZone_Domain` | PT60M |
| Actual generation by type | A75 | A16 | `in_Domain`, `inBiddingZone_Domain`, `psrType` | PT15M or PT60M |
| Day-ahead prices (SDAC) | A44 | A01 | `in_Domain`, `contract_MarketAgreement.type=A01` | PT60M |
| Cross-border physical flows | A11 | A16 | `in_Domain`, `out_Domain` | PT15M or PT60M |
| Imbalance prices | A85 | A16 | `controlArea_Domain` | varies |
| Imbalance volumes | A86 | A16 | `controlArea_Domain` | varies |

### EIC Area Codes

| Area | EIC Code | Timezone |
|------|----------|----------|
| Netherlands (NL) | `10YNL----------L` | Europe/Amsterdam |
| Germany-Luxembourg (DE_LU) | `10Y1001A1001A82H` | Europe/Berlin |
| Belgium (BE) | `10YBE----------2` | Europe/Brussels |

### PSR Production Types (used with A75)

| Code | Description |
|------|-------------|
| B01 | Biomass |
| B02 | Fossil gas |
| B03 | Fossil hard coal |
| B04 | Fossil oil |
| B05 | Geothermal |
| B07 | Hydro pumped storage |
| B08 | Hydro run-of-river |
| B09 | Hydro water reservoir |
| B10 | Marine |
| B11 | Nuclear |
| B12 | Other renewable |
| B13 | Solar |
| B14 | Waste |
| B15 | Wind offshore |
| B16 | Wind onshore |
| B17 | Fossil peat |
| B18 | Fossil sub-bituminous coal |
| B19 | Fossil lignite |
| B20 | Fossil coal derived gas |
| B21 | Hydro pumped storage (reclassified) |
| B22 | Solar thermal |
| B23 | Oceanic |
| B24 | Fossil fuel gas |
| B25 | Other (non-renewable) |

**Residual-load relevant PSRs:** B16 (wind onshore), B18 (sub-bituminous),
B19 (lignite).

### Chunking

The ENTSO-E API limits the number of elements per response. The client
pre-chunks long time ranges into conservative windows:

| Dataset | Max chunk |
|---------|-----------|
| Load / forecast | 31 days |
| Generation / flows | 92 days |
| Prices / imbalance | 366 days |

If a chunk still exceeds the element limit, the client automatically halves
it recursively (up to 10 levels).

### Day-Ahead Price Sequences

Germany-Luxembourg (DE_LU) and Austria (AT) publish multiple classification
sequences for day-ahead prices. The client defaults to sequence 1 (single
market clearing price), matching the behaviour of the widely-used `entsoe-py`
library. NL uses no sequence parameter.

---

## Open-Meteo Historical Archive

**Base URL:** `https://archive-api.open-meteo.com/v1/archive`

**Authentication:** None required for non-commercial use. Commercial use
requires a paid customer API key.

**Response format:** JSON with `hourly` data arrays, `hourly_units`, and
metadata fields.

**Verified behaviour (live testing, September 2026):**

- Multi-year single requests work (verified: 3-year hourly request returns
  ~1 MB). No chunking required by default.
- Free tier limits: 10,000 requests/day, 5,000/hour, 5,000/minute (from
  pricing page).
- Setting `timezone=UTC` returns ISO 8601 timestamps in UTC.
- Setting `wind_speed_unit=ms` returns wind speeds in m/s (SI).
- Null values in hourly arrays represent missing data points.

### Request Parameters

| Parameter | Value |
|-----------|-------|
| `latitude` | Location latitude |
| `longitude` | Location longitude |
| `start_date` | `YYYY-MM-DD` |
| `end_date` | `YYYY-MM-DD` (inclusive) |
| `hourly` | Comma-separated variable names |
| `timezone` | `UTC` (for consistent UTC timestamps) |
| `wind_speed_unit` | `ms` (meters per second) |
| `model` | Optional: specific reanalysis model |

### Hourly Variables

| Variable | Unit | Description |
|----------|------|-------------|
| `temperature_2m` | °C | Air temperature at 2m |
| `relative_humidity_2m` | % | Relative humidity at 2m |
| `wind_speed_10m` | m/s | Wind speed at 10m |
| `wind_speed_100m` | m/s | Wind speed at 100m |
| `wind_direction_100m` | ° | Wind direction at 100m |
| `shortwave_radiation` | W/m² | Total shortwave radiation |
| `direct_normal_irradiance` | W/m² | Direct normal irradiance |
| `diffuse_radiation` | W/m² | Diffuse radiation |
| `cloud_cover` | % | Total cloud cover |
| `precipitation` | mm | Total precipitation |

### Netherlands Weather Locations

| Name | Latitude | Longitude |
|------|----------|-----------|
|_nl-central_ | 52.21 | 5.29 |
| _nl-north_ | 53.22 | 6.57 |
| _nl-south_ | 51.44 | 5.47 |
| _nl-east_ | 52.05 | 6.89 |
| _nl-coast_ | 52.10 | 4.30 |
| _nl-offshore_ | 52.70 | 3.80 |

---

## NL Imbalance Prices — Status

**The availability of NL imbalance prices through the ENTSO-E A85 endpoint
is UNVERIFIED.**

- The A85 endpoint exists and is documented.
- A live API key is required to test whether NL TSO data is actually returned.
- Some ENTSO-E member TSOs publish imbalance data; others do not.
- If the endpoint returns "No matching data found" for NL, the data simply
  does not exist in ENTSO-E for this area.

**Fallback architecture:**

A deliberately-unimplemented synthetic imbalance penalty interface exists at
`src/gridpulse/ingestion/fallback/imbalance.py`. This provides:

- `ImbalancePenaltyProvider` — Protocol defining the interface.
- `SyntheticImbalancePenalty` — Concrete implementation that raises
  `NotImplementedError` with a clear message.

The synthetic penalty is **NOT implemented** and **NOT a real data source**.
It exists solely as a documented placeholder so the system can gracefully
handle the absence of real imbalance prices in later phases (e.g., using a
fixed penalty multiplier as a proxy when dispatching storage).

Tests enforce two invariants:
1. The ENTSO-E imbalance source always declares `nl_coverage: UNVERIFIED`.
2. The synthetic fallback always raises `NotImplementedError`.

---

## Verified vs. Uncertain

### Verified (live-tested against real APIs)

- ENTSO-E authentication mechanism (401 XML Ack)
- ENTSO-E XML response format and CIM element structure
- ENTSO-E "No matching data found" behaviour (200 + text)
- ENTSO-E element limit error message
- ENTSO-E ZIP response for A85 (imbalance prices)
- EIC codes for NL, DE_LU, BE
- Open-Meteo JSON response structure
- Open-Meteo multi-year single-request capability
- Open-Meteo free-tier rate limits (from published pricing page)
- Open-Meteo `timezone=UTC` and `wind_speed_unit=ms` behaviour

### Uncertain / Needs Live Key

- **NL imbalance price availability** — requires a live ENTSO-E API key to
  confirm whether NL TSO data is actually published through A85
- **Exact element counts per response** — the precise number of elements that
  triggers the limit error depends on server configuration and may change
- **ENTSO-E rate limits** — no published limits were found; the client uses
  conservative retries as a precaution
- **Open-Meteo commercial vs. non-commercial thresholds** — the pricing page
  distinguishes tiers but the boundary is not precisely defined for our use
  case
- **Generation data completeness** — not all areas publish all PSR types; some
  may return empty results for certain categories
