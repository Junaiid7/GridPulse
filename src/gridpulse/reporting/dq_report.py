"""Reproducible data-quality report for a completed pipeline run.

Takes the structured :class:`~gridpulse.pipeline.runner.PipelineRun` and
produces:

- a nested-dict representation suitable for JSON serialization;
- a Markdown document following the Step 8 specification;
- deterministic file output under ``data/reports/``.

Sections mirror the Step 8 DQ specification:

1. SOURCE — per dataset: source, entity, time range, retrieval info.
2. COVERAGE — row counts, expected, missing, duplicates.
3. VALUES — nulls, invalid, min/max/mean.
4. TIME — UTC policy, Amsterdam local, DST observations.
5. ENERGY — load / wind / solar / residual / price coverage.
6. WEATHER — per-location, per-variable, missingness, locations, variables.
7. IMBALANCE — NL imbalance-price availability status + evidence.
8. PROVENANCE — Bronze artifact locations, hashes, manifests.

No secrets are stored; API keys are never written to reports.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _collect_datasets(run) -> list[dict[str, Any]]:
    """Build the SOURCE + COVERAGE + VALUES sections from run.datasets."""
    rows: list[dict[str, Any]] = []
    for ds in run.datasets:
        rows.append({
            "source": ds.source,
            "entity": ds.entity,
            "status": ds.status,
            "requested_window": {
                "start_utc": ds.requested_start.isoformat() if ds.requested_start else None,
                "end_utc": ds.requested_end.isoformat() if ds.requested_end else None,
            },
            "actual_window": {
                "start_utc": ds.actual_start.isoformat() if ds.actual_start else None,
                "end_utc": ds.actual_end.isoformat() if ds.actual_end else None,
            },
            "fetched_at_utc": ds.fetched_at.isoformat() if ds.fetched_at else None,
            "resolution_minutes": ds.resolution_minutes,
            "unit": ds.unit,
            "row_count": ds.row_count,
            "expected_count": ds.expected_count,
            "missing_timestamps": ds.missing_timestamps,
            "duplicate_dropped": ds.duplicate_dropped,
            "null_values": ds.null_values,
            "min_value": ds.min_value,
            "max_value": ds.max_value,
            "mean_value": ds.mean_value,
            "note": ds.note,
            "local_note": ds.local_note,
            "bronze_dir": ds.bronze_dir,
            "bronze_sha256": ds.bronze_sha256,
            "silver_csv": ds.silver_csv,
        })
    return rows


def _dostats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def _energy_section(run) -> dict[str, Any]:
    gold = run.gold
    if not gold.produced:
        return {
            "gold_dataset": "NOT PRODUCED",
            "reason": gold.reason,
        }
    return {
        "gold_dataset": "produced",
        "row_count": gold.row_count,
        "actual_window": {
            "start_utc": gold.actual_start.isoformat() if gold.actual_start else None,
            "end_utc": gold.actual_end.isoformat() if gold.actual_end else None,
        },
        "residual_definition": gold.residual_definition,
        "wind_psr": gold.wind_psr,
        "solar_psr": gold.solar_psr,
        "missing_counts": gold.missing_counts,
        "duplicate_count": gold.duplicates,
        "residual_stats": gold.residual_stats,
        "negative_residual_fraction": gold.negative_residual_fraction,
    }


def _weather_section(run) -> dict[str, Any]:
    # Aggregate per-variable stats from the NL-hourly weather series.
    nl = run.weather_nl_hourly
    variables = sorted(nl.keys()) or list(run.weather_variables)
    by_var: dict[str, dict[str, Any]] = {}
    for var in variables:
        series = nl.get(var)
        if series is not None:
            by_var[var] = {
                "points": len(series.points),
                "unit": series.unit,
                "locations_aggregated": series.metadata.get("locations_total", 0),
                "hours_with_partial_coverage": series.metadata.get("hours_with_partial_coverage", 0),
            }
        else:
            # Hand-built or partial runs where weather_nl_hourly is not populated.
            by_var[var] = {"points": 0, "unit": "", "locations_aggregated": 0, "hours_with_partial_coverage": 0}
    # Location coverage summary: per (location, variable) status counts.
    status_counts: dict[str, int] = {}
    for ds in run.datasets:
        if ds.source != "open-meteo":
            continue
        key = f"{ds.status}"
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "locations": [loc.name for loc in run.weather_locations],
        "variables": variables,
        "by_variable": by_var,
        "per_location_variable_status_counts": status_counts,
    }


def _dostats_dict(run) -> dict[str, Any]:
    rows = _collect_datasets(run)
    per_source: dict[str, list[dict]] = {}
    for r in rows:
        per_source.setdefault(r["source"], []).append(r)
    return {src: _aggregate(src, ds_list) for src, ds_list in per_source.items()}


def _aggregate(source: str, datasets: list[dict]) -> dict[str, Any]:
    statuses = [d["status"] for d in datasets]
    return {
        "total_datasets": len(datasets),
        "statuses": {s: statuses.count(s) for s in sorted(set(statuses))},
    }


def _time_policy_section(run) -> dict[str, Any]:
    offsets = run.dst_offsets_observed  # {offset_minutes: count}
    n_offsets = len(offsets)
    transitions = run.dst_transitions_observed  # int
    return {
        "utc_policy": "All timestamps stored as aware UTC.",
        "local_time_policy": "Europe/Amsterdam (CET/CEST); local fields derived from UTC.",
        "dst_offsets_observed": offsets,
        "number_of_distinct_local_offsets": n_offsets,
        "dst_transition_count": transitions,
    }


class DataQualityReport:
    """Build and write the DQ report from a PipelineRun-like object."""

    def __init__(self, run: Any) -> None:
        self._run = run

    def to_dict(self) -> dict[str, Any]:
        run = self._run
        return {
            "report": {
                "schema_version": "1.0",
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "gridpulse_version": run.gridpulse_version,
                "python_version": run.python_version,
                "requested_window": {
                    "start_utc": run.requested_start.isoformat(),
                    "end_utc": run.requested_end.isoformat(),
                },
                "total_runtime_s": (
                    run.finished_at - run.started_at
                ).total_seconds() if run.finished_at and run.started_at else None,
            },
            "summary": {
                "open_meteo_datasets_verified": sum(
                    1 for d in run.datasets
                    if d.source == "open-meteo" and d.status == "verified_live"
                ),
                "entsoe_datasets_verified": sum(
                    1 for d in run.datasets
                    if d.source == "entsoe" and d.status == "verified_live"
                ),
                "entsoe_datasets_unavailable": sum(
                    1 for d in run.datasets
                    if d.source == "entsoe" and d.status != "verified_live"
                ),
                "imbalance_status": run.imbalance_note,
                "gold_produced": run.gold.produced,
                "features_produced": run.features.produced,
            },
            "time_policy": _time_policy_section(run),
            "source_coverage_values": _collect_datasets(run),
            "energy": _energy_section(run),
            "weather": _weather_section(run),
            "provenance": {
                "bronze_root": str(run.bronze_root),
                "silver_root": str(run.silver_root),
                "gold_root": str(run.gold_root),
                "hashes_where_available": [
                    {"entity": d.entity, "sha256": d.bronze_sha256}
                    for d in run.datasets
                    if d.bronze_sha256
                ],
            },
            "errors": run.errors,
        }

    def to_markdown(self) -> str:
        d = self.to_dict()
        lines: list[str] = []
        h = lines.append
        h("# GridPulse Data-Quality Report (Phase 4A)\n")
        meta = d["report"]
        h(f"- **Generated:** {meta['generated_at_utc']}")
        h(f"- **GridPulse:** {meta['gridpulse_version']} | Python {meta['python_version']}")
        rw = meta["requested_window"]
        h(f"- **Requested window:** {rw['start_utc']} → {rw['end_utc']}")
        if meta["total_runtime_s"] is not None:
            h(f"- **Runtime:** {meta['total_runtime_s']:.1f}s\n")
        h("---\n")

        # Time policy
        tp = d["time_policy"]
        h("## TIME POLICY\n")
        h(f"- {tp['utc_policy']}")
        h(f"- {tp['local_time_policy']}")
        h(f"- Distinct local offsets observed: {tp['number_of_distinct_local_offsets']}")
        h(f"- DST transitions in window: {tp['dst_transition_count']}\n")

        # Summary
        s = d["summary"]
        h("## SUMMARY\n")
        h("| Metric | Value |")
        h("|--------|-------|")
        h(f"| Open-Meteo datasets verified (live) | {s['open_meteo_datasets_verified']} |")
        h(f"| ENTSO-E datasets verified (live) | {s['entsoe_datasets_verified']} |")
        h(f"| ENTSO-E datasets unavailable | {s['entsoe_datasets_unavailable']} |")
        h(f"| Imbalance status | {s['imbalance_status']} |")
        h(f"| Gold produced | {s['gold_produced']} |")
        h(f"| Features produced | {s['features_produced']} |")

        # SOURCE/COVERAGE/VALUES
        h("\n## SOURCE / COVERAGE / VALUES\n")
        for ds in d["source_coverage_values"]:
            h(f"### {ds['source']}: {ds['entity']}")
            h(f"- **Status:** {ds['status']}")
            h(f"- **Requested:** {ds['requested_window']['start_utc']} → {ds['requested_window']['end_utc']}")
            if ds["actual_window"]["start_utc"]:
                h(f"- **Actual:** {ds['actual_window']['start_utc']} → {ds['actual_window']['end_utc']}")
            if ds["resolution_minutes"] is not None:
                h(f"- **Resolution:** {ds['resolution_minutes']} min")
            if ds["unit"]:
                h(f"- **Unit:** {ds['unit']}")
            h(f"- **Row count:** {ds['row_count']}")
            h(f"- **Expected count:** {ds['expected_count']}")
            h(f"- **Missing timestamps:** {ds['missing_timestamps']}")
            h(f"- **Duplicate dropped:** {ds['duplicate_dropped']}")
            if ds["null_values"] is not None:
                h(f"- **Null values:** {ds['null_values']}")
            if ds["min_value"] is not None:
                h(f"- **Min/Max/Mean:** {ds['min_value']:.3f} / {ds['max_value']:.3f} / {ds['mean_value']:.3f}")
            if ds["note"]:
                h(f"- **Note:** {ds['note']}")
            h("")

        # ENERGY
        h("## ENERGY\n")
        energy = d["energy"]
        if energy.get("gold_dataset") == "NOT PRODUCED":
            h(f"**Gold dataset NOT produced.** Reason: {energy['reason']}\n")
        else:
            h(f"- **Rows:** {energy['row_count']}")
            h(f"- **Residual definition:** {energy['residual_definition']}")
            h(f"- **Wind PSR:** {', '.join(energy['wind_psr'])}")
            h(f"- **Solar PSR:** {', '.join(energy['solar_psr'])}")
            mc = energy["missing_counts"]
            h("- **Missing counts:**")
            for k, v in mc.items():
                h(f"  - {k}: {v}")
            h(f"- **Duplicate rows:** {energy['duplicate_count']}")
            rs = energy["residual_stats"]
            if rs.get("min") is not None:
                h(f"- **Residual MW min/max/mean:** {rs['min']:.2f} / {rs['max']:.2f} / {rs['mean']:.2f}")
                h(f"- **Negative residual fraction:** {energy['negative_residual_fraction']*100:.1f}%")

        # WEATHER
        wx = d["weather"]
        h("\n## WEATHER\n")
        h(f"- **Locations:** {', '.join(wx['locations'])}")
        h(f"- **Variables:** {', '.join(wx['variables'])}")
        h("\n### Per-variable summary\n")
        for var, info in wx["by_variable"].items():
            h(f"- `{var}`: {info['points']} hours | unit={info['unit']} | locations_agg={info['locations_aggregated']} | partial_coverage_hrs={info['hours_with_partial_coverage']}")
        if wx["per_location_variable_status_counts"]:
            h("\n### Per-location-variable fetch status")
            for st, cnt in wx["per_location_variable_status_counts"].items():
                h(f"- {st}: {cnt}")

        # IMBALANCE
        h("\n## IMBALANCE\n")
        h(f"- **NL imbalance-price status:** {d['summary']['imbalance_status']}")

        # PROVENANCE
        prov = d["provenance"]
        h("\n## PROVENANCE\n")
        h(f"- Bronze root: `{prov['bronze_root']}`")
        h(f"- Silver root: `{prov['silver_root']}`")
        h(f"- Gold root: `{prov['gold_root']}`")
        if prov["hashes_where_available"]:
            h("\n### Bronze hashes\n")
            for h_entry in prov["hashes_where_available"]:
                h(f"- `{h_entry['entity']}`: {h_entry['sha256']}")

        if d["errors"]:
            h("\n## ERRORS / WARNINGS\n")
            for err in d["errors"]:
                h(f"- {err}")

        return "\n".join(lines) + "\n"

    def write(self, out_dir: Path) -> tuple[Path, Path]:
        """Write JSON + Markdown report; returns ``(json_path, md_path)``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        run = self._run
        slug = f"{run.requested_start:%Y%m%d}__{run.requested_end:%Y%m%d}"
        json_path = out_dir / f"dq-report_{slug}.json"
        md_path = out_dir / f"dq-report_{slug}.md"

        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path


__all__ = ["DataQualityReport"]
