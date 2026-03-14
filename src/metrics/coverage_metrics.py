"""
覆盖率评估指标模块
"""
import numpy as np


class CoverageMetrics:
    """覆盖控制指标计算：加权覆盖率、覆盖代价、热点追踪距离"""

    def __init__(self, domain=(0, 100, 0, 100), resolution=50, sensing_radius=15.0):
        """
        参数:
            domain: 工作域 (x_min, x_max, y_min, y_max)
            resolution: 网格分辨率
            sensing_radius: 智能体感知半径 (m)
        """
        self.sensing_radius = sensing_radius

        x_min, x_max, y_min, y_max = domain
        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        xx, yy = np.meshgrid(xs, ys)
        # 展平为 (N, 2) 的网格点集合
        self._grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        self._n_grid = self._grid_points.shape[0]

    def compute_weighted_coverage(self, position, field):
        """
        计算加权覆盖率：被感知范围覆盖的敏感度占总敏感度的比例

        参数:
            position: (2,) 智能体位置
            field: 具有 get_density 方法的敏感度场对象

        返回:
            coverage: 覆盖率 [0, 1]
        """
        densities = field.get_density(self._grid_points)  # (N,)
        total_density = densities.sum()

        if total_density < 1e-10:
            return 0.0

        # 计算网格点到智能体的距离
        dists = np.linalg.norm(self._grid_points - position, axis=1)
        covered_mask = dists <= self.sensing_radius

        covered_density = densities[covered_mask].sum()
        return float(covered_density / total_density)

    def compute_coverage_cost(self, position, field):
        """
        计算覆盖代价（单智能体版本）：
        H = Σ_q s(q) * ||q - p||^2

        参数:
            position: (2,) 智能体位置
            field: 具有 get_density 方法的敏感度场对象

        返回:
            cost: 覆盖代价（越小越好）
        """
        densities = field.get_density(self._grid_points)  # (N,)
        dist_sq = np.sum((self._grid_points - position) ** 2, axis=1)  # (N,)
        cost = np.sum(densities * dist_sq)
        return float(cost)

    def compute_hotspot_distance(self, position, field):
        """
        计算智能体到热点中心的欧几里得距离

        参数:
            position: (2,) 智能体位置
            field: 具有 get_hotspot_position 方法的敏感度场对象

        返回:
            distance: 到热点的距离 (m)
        """
        hotspot_pos = field.get_hotspot_position()
        return float(np.linalg.norm(np.array(position) - hotspot_pos))
