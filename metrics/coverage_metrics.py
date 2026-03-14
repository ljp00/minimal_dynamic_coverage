"""
metrics/coverage_metrics.py
统一计算实验指标的纯函数集合，以及逐步记录指标的 CoverageTracker 类。
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def compute_coverage(
    agent_positions: List[np.ndarray],
    grid: Tuple[np.ndarray, np.ndarray],
    coverage_radius: float,
) -> float:
    """
    计算总覆盖率：被至少一个智能体覆盖的网格单元占总面积的比例。

    参数：
        agent_positions: 智能体位置列表，每个元素为 [x, y]
        grid: (xx, yy) 网格坐标数组（由 np.meshgrid 生成）
        coverage_radius: 覆盖半径
    返回：
        覆盖率（0.0 ~ 1.0）
    """
    xx, yy = grid
    total_cells = xx.size
    covered = np.zeros(xx.shape, dtype=bool)

    for pos in agent_positions:
        dist_sq = (xx - pos[0]) ** 2 + (yy - pos[1]) ** 2
        covered |= dist_sq <= coverage_radius ** 2

    return float(np.sum(covered)) / total_cells


def compute_weighted_coverage(
    agent_positions: List[np.ndarray],
    field_values: np.ndarray,
    grid: Tuple[np.ndarray, np.ndarray],
    coverage_radius: float,
) -> float:
    """
    计算高敏感区加权覆盖率：覆盖区域的场值之和占总场值之和的比例。

    参数：
        agent_positions: 智能体位置列表
        field_values: 场值二维数组（与网格对应）
        grid: (xx, yy) 网格坐标数组
        coverage_radius: 覆盖半径
    返回：
        加权覆盖率（0.0 ~ 1.0）
    """
    xx, yy = grid
    covered = np.zeros(xx.shape, dtype=bool)

    for pos in agent_positions:
        dist_sq = (xx - pos[0]) ** 2 + (yy - pos[1]) ** 2
        covered |= dist_sq <= coverage_radius ** 2

    total_field = np.sum(field_values)
    if total_field < 1e-10:
        return 0.0

    covered_field = np.sum(field_values[covered])
    return float(covered_field) / total_field


def compute_hotspot_distance(
    agent_positions: List[np.ndarray],
    hotspot_centers: List[np.ndarray],
) -> float:
    """
    计算智能体与热点中心的平均最近距离。

    参数：
        agent_positions: 智能体位置列表
        hotspot_centers: 热点中心位置列表
    返回：
        平均距离（标量）
    """
    if not agent_positions or not hotspot_centers:
        return float("nan")

    total_dist = 0.0
    for pos in agent_positions:
        # 找到最近热点的距离
        dists = [np.linalg.norm(np.array(pos) - np.array(c)) for c in hotspot_centers]
        total_dist += min(dists)

    return total_dist / len(agent_positions)


def compute_overlap(
    agent_positions: List[np.ndarray],
    coverage_radius: float,
) -> float:
    """
    计算多智能体之间的覆盖重叠率。
    重叠率定义为：有重叠的智能体对数 / 总智能体对数。

    参数：
        agent_positions: 智能体位置列表
        coverage_radius: 覆盖半径
    返回：
        重叠率（0.0 ~ 1.0），单智能体时返回 0.0
    """
    n = len(agent_positions)
    if n <= 1:
        return 0.0

    total_pairs = n * (n - 1) / 2
    overlap_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(
                np.array(agent_positions[i]) - np.array(agent_positions[j])
            )
            if dist < 2.0 * coverage_radius:
                overlap_pairs += 1

    return overlap_pairs / total_pairs


class CoverageTracker:
    """
    逐步记录实验指标的追踪器，并提供汇总统计。
    """

    def __init__(self):
        # 历史记录字典，键为指标名称，值为列表
        self.history: Dict[str, List[float]] = {
            "coverage": [],
            "weighted_coverage": [],
            "hotspot_distance": [],
            "overlap": [],
        }

    def record(
        self,
        agent_positions: List[np.ndarray],
        field_values: np.ndarray,
        grid: Tuple[np.ndarray, np.ndarray],
        hotspot_centers: List[np.ndarray],
        coverage_radius: float,
    ) -> Dict[str, float]:
        """
        记录当前步的所有指标。

        参数：
            agent_positions: 智能体位置列表
            field_values: 当前场值二维数组
            grid: (xx, yy) 网格坐标数组
            hotspot_centers: 热点中心位置列表
            coverage_radius: 覆盖半径
        返回：
            当前步的指标字典
        """
        cov = compute_coverage(agent_positions, grid, coverage_radius)
        wcov = compute_weighted_coverage(
            agent_positions, field_values, grid, coverage_radius
        )
        hdist = compute_hotspot_distance(agent_positions, hotspot_centers)
        overlap = compute_overlap(agent_positions, coverage_radius)

        self.history["coverage"].append(cov)
        self.history["weighted_coverage"].append(wcov)
        self.history["hotspot_distance"].append(hdist)
        self.history["overlap"].append(overlap)

        return {
            "coverage": cov,
            "weighted_coverage": wcov,
            "hotspot_distance": hdist,
            "overlap": overlap,
        }

    def summary(self) -> Dict[str, float]:
        """
        返回各指标的均值汇总。
        """
        result = {}
        for key, values in self.history.items():
            if values:
                result[f"mean_{key}"] = float(np.mean(values))
                result[f"final_{key}"] = float(values[-1])
        return result
