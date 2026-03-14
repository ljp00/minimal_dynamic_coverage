"""Dynamic sensitivity field with a single Gaussian hotspot on a circular trajectory."""

import numpy as np


class DynamicSensitivityField:
    """A 2-D sensitivity field whose single Gaussian hotspot moves in a circle.

    Parameters
    ----------
    domain : tuple of float
        (x_min, x_max, y_min, y_max) — spatial extent of the field.
    center : array-like of shape (2,), optional
        Centre of the circular hotspot trajectory.  Defaults to the domain
        centroid.
    radius : float
        Radius of the circular hotspot trajectory (metres).
    angular_speed : float
        Angular speed of the hotspot (rad / s).  Positive → counter-clockwise.
    phase : float
        Initial phase of the hotspot (rad).
    sigma : float
        Standard deviation of the Gaussian kernel (metres).
    amplitude : float
        Peak value of the sensitivity field.
    grid_resolution : int
        Number of grid points along each axis used for coverage integration.
    """

    def __init__(
        self,
        domain=(0.0, 100.0, 0.0, 100.0),
        center=None,
        radius=30.0,
        angular_speed=0.2,
        phase=0.0,
        sigma=10.0,
        amplitude=1.0,
        grid_resolution=50,
    ):
        self.domain = domain  # (x_min, x_max, y_min, y_max)
        x_mid = 0.5 * (domain[0] + domain[1])
        y_mid = 0.5 * (domain[2] + domain[3])
        self.center = np.array(center, dtype=float) if center is not None else np.array([x_mid, y_mid])
        self.radius = float(radius)
        self.omega = float(angular_speed)
        self.phase = float(phase)
        self.sigma = float(sigma)
        self.amplitude = float(amplitude)
        self.grid_resolution = int(grid_resolution)

        # Pre-build the evaluation grid (for coverage integration)
        xs = np.linspace(domain[0], domain[1], grid_resolution)
        ys = np.linspace(domain[2], domain[3], grid_resolution)
        self._XX, self._YY = np.meshgrid(xs, ys)
        self._grid_pts = np.stack([self._XX.ravel(), self._YY.ravel()], axis=1)  # (N, 2)
        self._cell_area = (
            (domain[1] - domain[0]) / (grid_resolution - 1)
            * (domain[3] - domain[2]) / (grid_resolution - 1)
        )

    # ------------------------------------------------------------------
    # Hotspot position
    # ------------------------------------------------------------------

    def get_hotspot_position(self, t: float) -> np.ndarray:
        """Return the hotspot centre at time *t* (seconds).

        Parameters
        ----------
        t : float
            Current simulation time (s).

        Returns
        -------
        np.ndarray of shape (2,)
        """
        angle = self.omega * t + self.phase
        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)
        return np.array([x, y])

    def get_hotspot_position_at_time(self, t: float) -> np.ndarray:
        """Return the ground-truth hotspot position at an *arbitrary* time *t*.

        This is a named interface required by the Oracle controller so that
        callers can explicitly signal intent (future look-up) without relying
        on the positional semantics of ``get_hotspot_position``.  Keeping it
        as a distinct method also makes it easy to override in subclasses that
        may compute future positions differently (e.g. with prediction noise).

        Parameters
        ----------
        t : float
            Query time (s).  May be in the past or the future.

        Returns
        -------
        np.ndarray of shape (2,)
        """
        return self.get_hotspot_position(t)

    # ------------------------------------------------------------------
    # Field value and gradient
    # ------------------------------------------------------------------

    def get_value(self, pos: np.ndarray, t: float) -> float:
        """Sensitivity field value at position *pos* and time *t*.

        Parameters
        ----------
        pos : array-like of shape (2,)
        t   : float

        Returns
        -------
        float in [0, amplitude]
        """
        hotspot = self.get_hotspot_position(t)
        diff = np.asarray(pos) - hotspot
        return float(self.amplitude * np.exp(-0.5 * np.dot(diff, diff) / self.sigma**2))

    def get_values_on_grid(self, t: float) -> np.ndarray:
        """Evaluate the field on the pre-built grid at time *t*.

        Returns
        -------
        np.ndarray of shape (grid_resolution, grid_resolution)
        """
        hotspot = self.get_hotspot_position(t)
        diff = self._grid_pts - hotspot  # (N, 2)
        vals = self.amplitude * np.exp(-0.5 * np.sum(diff**2, axis=1) / self.sigma**2)
        return vals.reshape(self.grid_resolution, self.grid_resolution)

    def get_gradient(self, pos: np.ndarray, t: float) -> np.ndarray:
        """Analytical gradient of the sensitivity field w.r.t. position.

        Parameters
        ----------
        pos : array-like of shape (2,)
        t   : float

        Returns
        -------
        np.ndarray of shape (2,)
        """
        pos = np.asarray(pos, dtype=float)
        hotspot = self.get_hotspot_position(t)
        diff = pos - hotspot
        val = self.amplitude * np.exp(-0.5 * np.dot(diff, diff) / self.sigma**2)
        return -val * diff / self.sigma**2

    # ------------------------------------------------------------------
    # Grid / domain helpers
    # ------------------------------------------------------------------

    @property
    def grid_X(self) -> np.ndarray:
        return self._XX

    @property
    def grid_Y(self) -> np.ndarray:
        return self._YY
