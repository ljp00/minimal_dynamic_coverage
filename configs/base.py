"""
configs/base.py
基础实验参数配置，使用 dataclass 定义，提供合理默认值。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BaseConfig:
    """整个最小实验通用的基础参数"""

    # 仿真区域大小（正方形边长）
    area_size: float = 10.0

    # 时间步长
    dt: float = 0.1

    # 总仿真步数
    steps: int = 200

    # 智能体数量
    n_agents: int = 1

    # 智能体最大速度
    max_speed: float = 1.0

    # 热点高斯宽度（控制热点扩散范围）
    sigma: float = 1.5

    # 随机种子（保证实验可复现）
    seed: int = 42

    # 智能体覆盖半径
    coverage_radius: float = 1.0

    # 热点中心位置列表（每个元素为 [x, y]）
    hotspot_centers: List[List[float]] = field(
        default_factory=lambda: [[7.0, 7.0]]
    )

    # 热点移动速度（0.0 表示静态热点）
    hotspot_speed: float = 0.0

    # 热点运动类型："static"（静态）、"circle"（圆周）、"linear"（线性）
    hotspot_trajectory: str = "static"


def get_default_config() -> BaseConfig:
    """返回一个默认配置实例"""
    return BaseConfig()
