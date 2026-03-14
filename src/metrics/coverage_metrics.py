"""Single-agent coverage metrics."""

import numpy as np


class CoverageMetrics:
    """Compute scalar coverage quality metrics for a single agent.

    All metrics are evaluated at a single (position, time) query unless noted.

    Parameters
    ----------
    field : DynamicSensitivityField
        The environment field used for integration.
    coverage_radius : float
        The sensing / coverage radius of the agent (metres).  Used to compute
        the weighted coverage rate by counting grid cells within this radius.
    """

    def __init__(self, field, coverage_radius: float = 10.0):
        self.field = field
        self.coverage_radius = float(coverage_radius)

    # ------------------------------------------------------------------
    # Per-step metrics
    # ------------------------------------------------------------------

    def weighted_coverage(self, pos: np.ndarray, t: float) -> float:
        """Fraction of total field weight covered by the agent's footprint.

        The agent is modelled as a disc of radius *coverage_radius*.  The
        coverage fraction is computed as:

            C = sum_i phi_i * in_radius_i / sum_i phi_i

        where phi_i is the field value at grid cell i and ``in_radius_i`` is 1
        if cell i is within *coverage_radius* of *pos*.

        Parameters
        ----------
        pos : array-like of shape (2,)
        t   : float

        Returns
        -------
        float in [0, 1]
        """
        pos = np.asarray(pos, dtype=float)
        field_vals = self.field.get_values_on_grid(t).ravel()  # (N,)
        grid_pts = self.field._grid_pts  # (N, 2)
        dists = np.linalg.norm(grid_pts - pos, axis=1)
        covered = (dists <= self.coverage_radius).astype(float)
        total_weight = field_vals.sum()
        if total_weight < 1e-12:
            return 0.0
        return float((field_vals * covered).sum() / total_weight)

    def coverage_cost(self, pos: np.ndarray, t: float) -> float:
        """Coverage cost — weighted sum of *uncovered* field.

        A lower value is better.

            cost = sum_i phi_i * (1 - in_radius_i)

        Normalised by the total field weight so the result is in [0, 1].

        Parameters
        ----------
        pos : array-like of shape (2,)
        t   : float

        Returns
        -------
        float in [0, 1]
        """
        return 1.0 - self.weighted_coverage(pos, t)

    def hotspot_distance(self, pos: np.ndarray, t: float) -> float:
        """Euclidean distance from *pos* to the current hotspot centre.

        Parameters
        ----------
        pos : array-like of shape (2,)
        t   : float

        Returns
        -------
        float >= 0
        """
        pos = np.asarray(pos, dtype=float)
        hotspot = self.field.get_hotspot_position(t)
        return float(np.linalg.norm(pos - hotspot))

    def sensed_value(self, pos: np.ndarray, t: float) -> float:
        """Field value at the agent's current position.

        Parameters
        ----------
        pos : array-like of shape (2,)
        t   : float

        Returns
        -------
        float >= 0
        """
        return self.field.get_value(pos, t)
