"""Pure prefix fold of source-backed stable combo facts, separate from score."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StableComboFact:
    time_ms: float
    kind: Literal['increment', 'reset']
    source: Literal['tap', 'miss', 'ln-hold', 'ln-break', 'ln-final']
    column: int


@dataclass(frozen=True)
class StableComboState:
    combo: int = 0
    running_max: int = 0
    last_increment_ms: float = 0
    break_previous_value: int = 0
    last_break_ms: float = -99999
    last_large_break_ms: float = -99999


@dataclass(frozen=True)
class StableComboTimeline:
    facts: tuple[StableComboFact, ...]
    times: tuple[float, ...]
    states: tuple[StableComboState, ...]

    @classmethod
    def build(cls, facts: tuple[StableComboFact, ...]):
        # Preserve same-time source insertion order; never reorder by column
        # or reconcile to the header maximum. Queries include the whole prefix.
        state = StableComboState()
        states = []
        for fact in facts:
            if fact.kind == 'increment':
                combo = state.combo + 1
                state = StableComboState(combo, max(state.running_max, combo), fact.time_ms,
                                         state.break_previous_value, state.last_break_ms,
                                         state.last_large_break_ms)
            else:
                state = StableComboState(0, state.running_max, state.last_increment_ms,
                                         state.combo if state.combo else state.break_previous_value,
                                         fact.time_ms if state.combo else state.last_break_ms,
                                         fact.time_ms if state.combo > 20 else state.last_large_break_ms)
            states.append(state)
        return cls(facts, tuple(fact.time_ms for fact in facts), tuple(states))

    @property
    def reconstructed_max_combo(self):
        return self.states[-1].running_max if self.states else 0

    def at(self, t_ms: float):
        index = bisect_right(self.times, t_ms) - 1
        return self.states[index] if index >= 0 else StableComboState()
