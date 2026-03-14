"""
agents/dynamics.py
定义智能体的运动学模型（一阶积分器）。
支持单智能体和多智能体，包含速度限幅和边界裁剪。
"""

import numpy as np
from typing import List, Optional


class AgentDynamics:
    """
    智能体运动学类，实现一阶积分器模型：
        p_{t+1} = p_t + clip(u, max_speed) * dt

    支持多智能体，用列表管理各智能体位置。
    """

    def __init__(self, config, init_positions: Optional[List] = None):
        """
        参数：
            config: BaseConfig 实例
            init_positions: 初始位置列表，每个元素为 [x, y]；
                            若为 None，则在区域内随机生成
        """
        self.config = config
        self.n_agents = config.n_agents
        self.dt = config.dt
        self.max_speed = config.max_speed
        self.area_size = config.area_size

        rng = np.random.default_rng(config.seed)

        if init_positions is not None:
            # 使用指定初始位置
            self.positions = [np.array(p, dtype=float) for p in init_positions]
        else:
            # 随机生成初始位置（在区域内均匀分布）
            self.positions = [
                rng.uniform(0.0, self.area_size, size=2)
                for _ in range(self.n_agents)
            ]

        # 记录初始位置，便于 reset
        self._init_positions = [p.copy() for p in self.positions]

    def _clip_speed(self, u: np.ndarray) -> np.ndarray:
        """速度限幅：若控制输入超过最大速度，则归一化到最大速度"""
        speed = np.linalg.norm(u)
        if speed > self.max_speed:
            u = u / speed * self.max_speed
        return u

    def _clip_boundary(self, pos: np.ndarray) -> np.ndarray:
        """边界裁剪：保持智能体在仿真区域 [0, area_size] 内"""
        return np.clip(pos, 0.0, self.area_size)

    def step(self, controls: List[np.ndarray]) -> List[np.ndarray]:
        """
        接收控制输入列表，更新所有智能体位置并返回新位置列表。

        参数：
            controls: 控制输入列表，每个元素为形如 [ux, uy] 的 numpy 数组
        返回：
            更新后的智能体位置列表
        """
        for i, u in enumerate(controls):
            u = np.array(u, dtype=float)
            # 速度限幅
            u = self._clip_speed(u)
            # 一阶积分器更新
            self.positions[i] = self._clip_boundary(self.positions[i] + u * self.dt)
        return [p.copy() for p in self.positions]

    def get_positions(self) -> List[np.ndarray]:
        """返回所有智能体当前位置（副本列表）"""
        return [p.copy() for p in self.positions]

    def get_position(self, agent_idx: int = 0) -> np.ndarray:
        """返回指定智能体当前位置（副本）"""
        return self.positions[agent_idx].copy()

    def reset(self, positions: Optional[List] = None) -> List[np.ndarray]:
        """
        重置智能体位置。

        参数：
            positions: 新的初始位置列表；若为 None，则恢复到构造时的初始位置
        返回：
            重置后的位置列表
        """
        if positions is not None:
            self.positions = [np.array(p, dtype=float) for p in positions]
        else:
            self.positions = [p.copy() for p in self._init_positions]
        return self.get_positions()
