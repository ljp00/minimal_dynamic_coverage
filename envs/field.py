"""
envs/field.py
定义敏感度场/热点场，是整个实验的核心"环境信号源"。
支持静态热点和动态热点（圆周运动、线性运动）。
"""

import numpy as np
from typing import List, Tuple


class SensitivityField:
    """
    敏感度场类，管理热点分布与随时间变化的场值。

    热点公式（高斯叠加）：
        phi(x, y) = sum_i A_i * exp(-((x - cx_i)^2 + (y - cy_i)^2) / (2 * sigma^2))
    """

    def __init__(self, config, grid_resolution: int = 100):
        """
        参数：
            config: BaseConfig 实例
            grid_resolution: 网格分辨率（每个维度的点数）
        """
        self.config = config
        self.area_size = config.area_size
        self.sigma = config.sigma
        self.hotspot_speed = config.hotspot_speed
        self.hotspot_trajectory = config.hotspot_trajectory
        # 初始热点中心位置列表，每个元素为 [x, y]
        self.initial_centers = [np.array(c, dtype=float) for c in config.hotspot_centers]
        self.n_hotspots = len(self.initial_centers)

        # 建立离散网格
        self.grid_resolution = grid_resolution
        xs = np.linspace(0.0, self.area_size, grid_resolution)
        ys = np.linspace(0.0, self.area_size, grid_resolution)
        # meshgrid：xx[i,j] 对应第 i 行第 j 列的 x 坐标
        self.xx, self.yy = np.meshgrid(xs, ys)
        # 网格范围，用于可视化
        self.grid_extent = [0.0, self.area_size, 0.0, self.area_size]

    def get_hotspot_centers(self, t: float) -> List[np.ndarray]:
        """
        返回时刻 t 的热点中心位置列表。

        参数：
            t: 当前时刻（秒）
        返回：
            热点中心位置列表，每个元素为形如 [x, y] 的 numpy 数组
        """
        centers = []
        for i, init_c in enumerate(self.initial_centers):
            if self.hotspot_trajectory == "static" or self.hotspot_speed == 0.0:
                # 静态热点：位置不变
                centers.append(init_c.copy())
            elif self.hotspot_trajectory == "circle":
                # 圆周运动：以初始位置为圆心（相对区域中心）进行圆周运动
                # 圆半径取区域大小的 1/4
                radius = self.area_size / 4.0
                cx0, cy0 = self.area_size / 2.0, self.area_size / 2.0
                angle = self.hotspot_speed * t + i * (2 * np.pi / self.n_hotspots)
                x = cx0 + radius * np.cos(angle)
                y = cy0 + radius * np.sin(angle)
                centers.append(np.array([x, y]))
            elif self.hotspot_trajectory == "linear":
                # 线性运动：沿 x 轴方向匀速移动，到达边界后反弹
                dx = self.hotspot_speed * t
                period = 2.0 * self.area_size
                # 利用三角波计算边界反弹
                phase = dx % period
                if phase <= self.area_size:
                    x = init_c[0] + phase
                else:
                    x = init_c[0] + period - phase
                # 限制在区域内
                x = np.clip(x, 0.0, self.area_size)
                centers.append(np.array([x, init_c[1]]))
            else:
                # 默认静态
                centers.append(init_c.copy())
        return centers

    def get_future_hotspot_centers(self, t: float, horizon: float) -> List[np.ndarray]:
        """
        返回未来真实热点位置（供 oracle 预测器使用）。

        参数：
            t: 当前时刻
            horizon: 预测时域（秒）
        返回：
            未来时刻 (t + horizon) 的热点中心位置列表
        """
        return self.get_hotspot_centers(t + horizon)

    def get_value(self, x: float, y: float, t: float) -> float:
        """
        返回某个点 (x, y) 在时刻 t 的敏感度值（高斯叠加）。

        参数：
            x, y: 查询点坐标
            t: 当前时刻
        返回：
            该点的敏感度值（标量）
        """
        centers = self.get_hotspot_centers(t)
        value = 0.0
        for c in centers:
            dist_sq = (x - c[0]) ** 2 + (y - c[1]) ** 2
            value += np.exp(-dist_sq / (2.0 * self.sigma ** 2))
        return value

    def get_field(self, t: float) -> np.ndarray:
        """
        返回整张场的二维 numpy 数组 phi(x, y, t)，在网格上离散化。

        参数：
            t: 当前时刻
        返回：
            形状为 (grid_resolution, grid_resolution) 的二维数组
        """
        centers = self.get_hotspot_centers(t)
        field = np.zeros_like(self.xx)
        for c in centers:
            dist_sq = (self.xx - c[0]) ** 2 + (self.yy - c[1]) ** 2
            field += np.exp(-dist_sq / (2.0 * self.sigma ** 2))
        return field
