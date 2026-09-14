"""Unit tests for provenance records attached to Silver/Gold outputs."""

from __future__ import annotations

from datetime import datetime, timezone

from gridpulse.transformation.provenance import Provenance

UTC = timezone.utc


def test_provenance_default_generated_at_is_utc_now() -> None:
    before = datetime.now(UTC)
    p = Provenance("silver", "entsoe", "actual-total-load")
    after = datetime.now(UTC)
    assert p.generated_at is not None
    assert before <= p.generated_at <= after
    assert p.generated_at.tzinfo is not None


def test_provenance_to_dict_serializes() -> None:
    p = Provenance(
        tier="gold",
        source="entsoe+open-meteo",
        entity="hourly-nl-dataset",
        unit="MW",
        generated_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        notes=("residual=load-wind-solar",),
    )
    d = p.to_dict()
    assert d["tier"] == "gold"
    assert d["unit"] == "MW"
    assert d["generated_at"] == "2024-01-01T12:00:00+00:00"
    assert d["notes"] == ["residual=load-wind-solar"]
    assert d["timezone_policy"] == "canonical_utc; local derived as Europe/Amsterdam"


def test_provenance_default_timezone_policy() -> None:
    assert Provenance("silver", "s", "e").timezone_policy == "canonical_utc; local derived as Europe/Amsterdam"


def test_provenance_is_frozen() -> None:
    p = Provenance("silver", "s", "e")
    try:
        p.tier = "gold"  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001 - frozen dataclass raises FrozenInstanceError
        assert "FrozenInstanceError" in type(exc).__name__ or "frozen" in str(exc).lower()
    else:
        raise AssertionError("Provenance should be immutable")