"""Attack exposure and strength curriculum for adversarial training.

Mixes clean and attacked episodes (``p_attack``) so the replay retains viable-state experience,
and ramps the attack budget from ``budget_min`` toward ``budget_max`` only while the defender
stays competent, meaning windowed delivery over attacked episodes at or above
``competence_floor``. Deterministic given ``seed``, and free of framework dependencies.
"""

from __future__ import annotations

import random
from collections import deque


class AttackCurriculum:
    def __init__(
        self,
        *,
        budget_min: float,
        budget_max: float,
        n_levels: int = 4,
        p_attack: float = 0.75,
        competence_floor: float = 0.4,
        window: int = 20,
        seed: int = 0,
    ) -> None:
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        if not 0.0 <= p_attack <= 1.0:
            raise ValueError("p_attack must be in [0, 1]")
        self.budget_min = float(budget_min)
        self.budget_max = float(budget_max)
        self.n_levels = int(n_levels)
        self.p_attack = float(p_attack)
        self.competence_floor = float(competence_floor)
        self.window = int(window)
        self._rng = random.Random(seed)
        self._level = 0  # 0 .. n_levels-1
        self._recent: deque[float] = deque(maxlen=self.window)
        self._last_attacked = False

    @property
    def level(self) -> int:
        return self._level

    def _budget_at(self, level: int) -> float:
        if self.n_levels == 1:
            return self.budget_max
        frac = level / (self.n_levels - 1)
        return self.budget_min + frac * (self.budget_max - self.budget_min)

    @property
    def current_budget(self) -> float:
        return self._budget_at(self._level)

    def decide(self) -> tuple[bool, float]:
        """Draw whether the next episode is attacked, and at what budget.

        Clean episodes report a zero budget, leaving the antagonist inert.
        """
        attacked = self._rng.random() < self.p_attack
        self._last_attacked = attacked
        return attacked, (self.current_budget if attacked else 0.0)

    def record(self, delivery_rate: float) -> None:
        """Feed back an episode's delivery rate; only attacked episodes drive the ramp.

        A full window of attacked episodes averaging at or above the floor advances one
        difficulty level and resets the window, so each level must be re-earned.
        """
        if not self._last_attacked:
            return
        self._recent.append(float(delivery_rate))
        if len(self._recent) >= self.window and self._level < self.n_levels - 1:
            if sum(self._recent) / len(self._recent) >= self.competence_floor:
                self._level += 1
                self._recent.clear()

    def state(self) -> dict:
        """Return a loggable snapshot of the curriculum state."""
        return {
            "level": self._level,
            "budget": self.current_budget,
            "window_fill": len(self._recent),
            "window_mean_delivery": (sum(self._recent) / len(self._recent)) if self._recent else 0.0,
        }
