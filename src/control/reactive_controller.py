"""Reactive and Oracle single-agent controllers."""

import numpy as np


class ReactiveController:
    """Proportional controller that tracks the *current* hotspot position.

    At each step the controller computes:

        u = k * (target - pos)

    where ``target = field.get_hotspot_position(t)``.  The output is clipped to
    *max_velocity* before being returned (the UAV's ``step()`` method will also
    clip, but we clip here so the recorded velocity matches).

    Parameters
    ----------
    gain : float
        Proportional gain (1/s).
    max_velocity : float
        Saturation speed (m/s).
    """

    def __init__(self, gain: float = 1.0, max_velocity: float = 5.0):
        self.gain = float(gain)
        self.max_velocity = float(max_velocity)

    def compute(self, pos: np.ndarray, field, t: float) -> np.ndarray:
        """Return the velocity command.

        Parameters
        ----------
        pos   : array-like of shape (2,) — current agent position
        field : DynamicSensitivityField
        t     : float — current simulation time (s)

        Returns
        -------
        np.ndarray of shape (2,) — velocity command (m/s)
        """
        pos = np.asarray(pos, dtype=float)
        target = field.get_hotspot_position(t)
        u = self.gain * (target - pos)
        speed = np.linalg.norm(u)
        if speed > self.max_velocity:
            u = u * self.max_velocity / speed
        return u


class OracleController:
    """Proportional controller that tracks the *future* hotspot position.

    At each step the controller computes:

        u = k * (target - pos)

    where ``target = field.get_hotspot_position_at_time(t + horizon)``.

    This represents a *perfect predictor* and acts as a theoretical upper bound
    on single-agent performance.

    Parameters
    ----------
    horizon : float
        Look-ahead time (s).
    gain : float
        Proportional gain (1/s).
    max_velocity : float
        Saturation speed (m/s).
    """

    def __init__(self, horizon: float = 0.5, gain: float = 1.0, max_velocity: float = 5.0):
        self.horizon = float(horizon)
        self.gain = float(gain)
        self.max_velocity = float(max_velocity)

    def compute(self, pos: np.ndarray, field, t: float) -> np.ndarray:
        """Return the velocity command.

        Parameters
        ----------
        pos   : array-like of shape (2,) — current agent position
        field : DynamicSensitivityField
        t     : float — current simulation time (s)

        Returns
        -------
        np.ndarray of shape (2,) — velocity command (m/s)
        """
        pos = np.asarray(pos, dtype=float)
        target = field.get_hotspot_position_at_time(t + self.horizon)
        u = self.gain * (target - pos)
        speed = np.linalg.norm(u)
        if speed > self.max_velocity:
            u = u * self.max_velocity / speed
        return u
