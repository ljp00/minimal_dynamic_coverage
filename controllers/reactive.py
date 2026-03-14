"""
controllers/reactive.py
定义无预测的反应式控制器（baseline）。
控制逻辑：每个智能体朝当前最近的热点中心移动。
"""

import numpy as np
from typing import List


class ReactiveController:
    """
    反应式控制器：根据当前热点位置驱动智能体运动，不依赖任何未来信息。

    控制策略：
        - 计算每个智能体到各热点中心的距离
        - 朝最近热点方向移动（以最大速度行进）
    """

    def __init__(self, config):
        """
        参数：
            config: BaseConfig 实例
        """
        self.config = config
        self.max_speed = config.max_speed

    def compute(
        self,
        agent_positions: List[np.ndarray],
        field,
        t: float,
    ) -> List[np.ndarray]:
        """
        计算每个智能体的控制输入。

        参数：
            agent_positions: 智能体位置列表，每个元素为 [x, y] 数组
            field: SensitivityField 实例（用于获取当前热点中心）
            t: 当前时刻
        返回：
            控制输入列表，每个元素为 [ux, uy] 数组
        """
        # 获取当前时刻的热点中心
        hotspot_centers = field.get_hotspot_centers(t)
        controls = []

        for pos in agent_positions:
            pos = np.array(pos, dtype=float)

            if len(hotspot_centers) == 0:
                # 无热点时，控制输入为零
                controls.append(np.zeros(2))
                continue

            # 找到距离当前智能体最近的热点中心
            distances = [np.linalg.norm(pos - c) for c in hotspot_centers]
            nearest_idx = int(np.argmin(distances))
            target = hotspot_centers[nearest_idx]

            # 计算方向向量并归一化后乘以最大速度
            direction = target - pos
            dist = np.linalg.norm(direction)

            if dist < 1e-6:
                # 智能体已在热点处，停止运动
                u = np.zeros(2)
            else:
                u = direction / dist * self.max_speed

            controls.append(u)

        return controls
