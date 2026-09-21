"""Phase 5: End-to-End Orchestration.

This module provides the top-level E2E orchestration that ties together:
- Phase 4A: Data pipeline (run_pipeline)
- Phase 4C: Probabilistic forecasting (run_probabilistic_benchmark)
- Phase 4D-B: Dispatch backtesting (run_dispatch_backtest)

The orchestration manages data-status propagation: fixtures use
``FIXTURE-VERIFIED``, real ENTSO-E data uses ``verified_live``.
"""

from gridpulse.orchestration.runner import run_e2e_orchestration  # noqa: E402

__all__ = ["run_e2e_orchestration"]
