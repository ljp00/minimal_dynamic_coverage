"""
动态敏感度场模块
单个高斯热点做圆周运动
"""
import numpy as np


class DynamicSensitivityField:
    """动态敏感度场：单个高斯热点沿圆形轨迹运动"""

    def __init__(self, domain=(0, 100, 0, 100), resolution=50, background_density=0.1):
        # 工作域
        self.x_min, self.x_max, self.y_min, self.y_max = domain
        self.resolution = resolution
        self.background_density = background_density  # 背景基础密度

        # 热点运动参数
        self.origin = np.array([50.0, 50.0])   # 圆周运动中心
        self.amplitude = 25.0                   # 运动半径
        self.omega = 0.2                        # 角速度 (rad/s)
        self.intensity = 2.0                    # 热点强度
        self.spread = 15.0                      # 热点扩散范围（高斯标准差）

        self.time = 0.0
        # 初始热点位置（t=0 时刻）
        self.center = self.origin + self.amplitude * np.array([np.cos(0.0), np.sin(0.0)])

        # 预计算网格坐标
        xs = np.linspace(self.x_min, self.x_max, self.resolution)
        ys = np.linspace(self.y_min, self.y_max, self.resolution)
        self._xx, self._yy = np.meshgrid(xs, ys)  # shape (resolution, resolution)

    def update(self, dt):
        """更新时间步，移动热点"""
        self.time += dt
        angle = self.omega * self.time
        self.center = self.origin + self.amplitude * np.array([np.cos(angle), np.sin(angle)])

    def get_density(self, positions):
        """
        计算给定位置处的敏感度密度值

        参数:
            positions: (N, 2) 或 (2,) 的位置数组

        返回:
            density: (N,) 的密度值数组
        """
        positions = np.atleast_2d(positions)
        dist_sq = np.sum((positions - self.center) ** 2, axis=1)
        density = self.intensity * np.exp(-dist_sq / (2 * self.spread ** 2))
        density += self.background_density  # 背景密度
        return density

    def get_field_grid(self):
        """
        返回整个工作域上的敏感度场网格

        返回:
            field: (resolution, resolution) 的密度场数组
            xx, yy: 对应的网格坐标
        """
        grid_points = np.column_stack([self._xx.ravel(), self._yy.ravel()])
        field = self.get_density(grid_points).reshape(self.resolution, self.resolution)
        return field, self._xx, self._yy

    def get_hotspot_position(self):
        """返回当前热点中心位置"""
        return self.center.copy()
