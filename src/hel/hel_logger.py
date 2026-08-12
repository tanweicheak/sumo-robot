"""
src.hel.hel_logger

Phase: Phase 3
Purpose: HEL logging (report Section 3.3.3.4). Records per-cycle inference timing and
    deadline compliance from the orchestration's timing_ms dict, and computes the
    latency-compliance-rate and mean-latency metrics used as KPIs (measured across all
    of 5a/5b/5c; enforced-under-throttling only in 5c).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HELLogRecord:
    frame_index: int
    actual_ms: float
    deadline_compliant: bool


class HELLogger:
    def __init__(self, decision_window_ms: float = 50.0) -> None:
        self.decision_window_ms = decision_window_ms
        self.records: list[HELLogRecord] = []

    def log_cycle(self, frame_index: int, timing_ms: dict[str, float], injected_ms: float | None = None) -> HELLogRecord:
        actual = sum(timing_ms.values()) if injected_ms is None else injected_ms
        rec = HELLogRecord(frame_index, actual, actual <= self.decision_window_ms)
        self.records.append(rec)
        return rec

    def latency_compliance_rate(self) -> float:
        """Fraction of cycles completing within the decision window (Phase 5c KPI)."""
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.deadline_compliant) / len(self.records)

    def mean_latency_ms(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.actual_ms for r in self.records) / len(self.records)

    def reset(self) -> None:
        self.records.clear()