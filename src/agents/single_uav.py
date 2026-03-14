"""Single UAV agent with first-order kinematics."""

import numpy as np


class SingleUAV:
    """A single UAV modelled with first-order (velocity-controlled) dynamics.

    Parameters
    ----------
    init_pos : array-like of shape (2,)
        Initial position of the agent (metres).
    max_velocity : float
        Maximum speed (m / s).
    """

    def __init__(self, init_pos, max_velocity: float = 5.0):
        self.pos = np.array(init_pos, dtype=float)
        self.max_velocity = float(max_velocity)

        # History lists (populated by calling record())
        self.position_history: list[np.ndarray] = [self.pos.copy()]
        self.sensed_value_history: list[float] = []
        self.speed_history: list[float] = []

    def step(self, u: np.ndarray, dt: float) -> None:
        """Apply a velocity command *u* for one time step.

        The command is clipped to *max_velocity* before integration.

        Parameters
        ----------
        u  : array-like of shape (2,)  — desired velocity (m/s)
        dt : float                     — time step (s)
        """
        u = np.asarray(u, dtype=float)
        speed = np.linalg.norm(u)
        if speed > self.max_velocity:
            u = u * self.max_velocity / speed
        self.pos = self.pos + u * dt

    def sense(self, field, t: float) -> float:
        """Return the sensitivity field value at the agent's current position.

        Parameters
        ----------
        field : DynamicSensitivityField
        t     : float — current time (s)

        Returns
        -------
        float
        """
        return field.get_value(self.pos, t)

    def record(self, field, t: float, u: np.ndarray) -> None:
        """Append current state to the history buffers.

        Parameters
        ----------
        field : DynamicSensitivityField
        t     : float — current simulation time (s)
        u     : array-like of shape (2,) — velocity applied this step
        """
        self.position_history.append(self.pos.copy())
        self.sensed_value_history.append(self.sense(field, t))
        u = np.asarray(u, dtype=float)
        self.speed_history.append(float(np.linalg.norm(u)))

    @property
    def trajectory(self) -> np.ndarray:
        """Full trajectory as (T, 2) array."""
        return np.array(self.position_history)
