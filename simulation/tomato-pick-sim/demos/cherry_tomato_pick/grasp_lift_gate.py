"""Optional right-hand lift interlock, independent of physics and policy transport."""
from __future__ import annotations

import numpy as np


class GraspLiftGate:
    """Gate the first grasp's upward TCP targets until physical support is ready.

    The caller supplies the backend's existing support confirmation, including
    contact geometry, finger wrap and hold duration. No closure-only shortcut.
    """

    PHASES = ("disabled", "approach", "waiting", "confirmed", "released", "timed_out")

    def __init__(self, enabled=False, timeout_s=8.):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("grasp wait timeout must be finite and positive")
        self.enabled = bool(enabled)
        self.timeout_s = float(timeout_s)
        self.trigger_closure = None
        self.phase = "approach" if enabled else "disabled"
        self.started_at = None
        self.confirmed_at = None
        self.released_at = None
        self.ceiling_m = None
        self.limited_targets = 0
        self.events = []

    def configure(self, trigger_closure):
        if not np.isfinite(trigger_closure) or not 0 < trigger_closure <= 1:
            raise ValueError("minimum_support_grip must be finite and in (0, 1]")
        self.trigger_closure = float(trigger_closure)

    def _transition(self, phase, time_s):
        self.phase = phase
        self.events.append({"phase": phase, "time_s": float(time_s), "ceiling_m": self.ceiling_m})

    def observe(self, time_s, support_ready):
        """Return True when cached model actions must be discarded.

        Called every physics tick. Confirmation just allows a *fresh* request;
        the capped target remains held until that request succeeds.
        """
        if self.phase not in {"waiting", "confirmed"}:
            return False
        if time_s - self.started_at >= self.timeout_s - 1e-9:
            self._transition("timed_out", time_s)
            return True
        if self.phase == "waiting" and support_ready:
            self.confirmed_at = float(time_s)
            self._transition("confirmed", time_s)
            return True
        if self.phase == "confirmed" and not support_ready:
            # Do not reset the deadline on transient confirmation/contact loss.
            self._transition("waiting", time_s)
            return True
        return False

    def constrain(self, action, measured_right_z, time_s, *, fresh_inference=False):
        """Modify only right TCP Z; preserve hand commands and other pose axes."""
        result = action.copy()
        if self.phase == "approach" and result[15] >= self.trigger_closure:
            self.started_at = float(time_s)
            self.ceiling_m = float(measured_right_z)
            self._transition("waiting", time_s)
        if self.phase == "confirmed" and fresh_inference:
            if result[15] >= self.trigger_closure:
                self.released_at = float(time_s)
                self._transition("released", time_s)
            else:
                # Do not lift on the same action that commands the confirmed
                # support hand to open. Keep the original wait deadline.
                self._transition("waiting", time_s)
        if self.phase in {"waiting", "confirmed", "timed_out"}:
            if result[9] > self.ceiling_m:
                self.limited_targets += 1
            # Ratchet downward to allow the model to finish approaching, but
            # never accumulate upward drift via successive small targets.
            self.ceiling_m = min(self.ceiling_m, float(result[9]))
            result[9] = self.ceiling_m
        return result

    @property
    def phase_index(self):
        return self.PHASES.index(self.phase)

    def summary(self):
        return {"enabled": self.enabled, "phase": self.phase,
                "timeout_s": self.timeout_s, "trigger_closure": self.trigger_closure,
                "started_at_s": self.started_at, "confirmed_at_s": self.confirmed_at,
                "released_at_s": self.released_at, "height_ceiling_m": self.ceiling_m,
                "limited_targets": self.limited_targets, "events": self.events}
