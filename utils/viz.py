"""
utils/viz.py
统一可视化函数集合，使用 matplotlib 实现。
支持场热力图、轨迹图、指标曲线和仿真动画。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，支持无显示器环境保存图片
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional, Dict, Tuple


def plot_field(
    field_values: np.ndarray,
    grid_extent: List[float],
    agent_positions: Optional[List[np.ndarray]] = None,
    hotspot_centers: Optional[List[np.ndarray]] = None,
    coverage_radius: float = 1.0,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制敏感度场热力图，并可叠加智能体位置。

    参数：
        field_values: 二维场值数组
        grid_extent: [xmin, xmax, ymin, ymax]
        agent_positions: 智能体位置列表（可选）
        hotspot_centers: 热点中心列表（可选，用星号标注）
        coverage_radius: 覆盖半径（绘制覆盖圆时使用）
        title: 图标题
        save_path: 若指定则保存到该路径
    返回：
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        field_values,
        extent=grid_extent,
        origin="lower",
        cmap="hot",
        interpolation="bilinear",
        vmin=0.0,
    )
    plt.colorbar(im, ax=ax, label="敏感度")

    # 绘制智能体位置（蓝色圆点）及覆盖圆
    if agent_positions:
        for pos in agent_positions:
            ax.scatter(pos[0], pos[1], c="blue", s=80, zorder=5, label="智能体")
            circle = plt.Circle(
                (pos[0], pos[1]),
                coverage_radius,
                color="blue",
                fill=False,
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.add_patch(circle)

    # 绘制热点中心（红色星号）
    if hotspot_centers:
        for c in hotspot_centers:
            ax.scatter(c[0], c[1], c="red", s=200, marker="*", zorder=6, label="热点中心")

    ax.set_xlim(grid_extent[0], grid_extent[1])
    ax.set_ylim(grid_extent[2], grid_extent[3])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or "敏感度场")

    # 去除重复图例项
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_trajectory(
    trajectory: List[np.ndarray],
    hotspot_trajectory: Optional[List[np.ndarray]] = None,
    area_size: Optional[float] = None,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制智能体轨迹图，可叠加热点真实轨迹。

    参数：
        trajectory: 智能体位置序列，每个元素为 [x, y]
        hotspot_trajectory: 热点真实轨迹序列（可选），每个元素为 [x, y] 或热点列表
        area_size: 区域大小（用于设置坐标轴范围）
        title: 图标题
        save_path: 若指定则保存到该路径
    返回：
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    if trajectory:
        traj = np.array(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1.5, label="智能体轨迹", alpha=0.8)
        # 标注起点和终点
        ax.scatter(traj[0, 0], traj[0, 1], c="green", s=100, zorder=5, label="起点")
        ax.scatter(traj[-1, 0], traj[-1, 1], c="blue", s=100, marker="^", zorder=5, label="终点")

    if hotspot_trajectory:
        # 若热点轨迹每个元素为列表（多热点），取第一个热点
        ht = []
        for entry in hotspot_trajectory:
            if isinstance(entry, (list, tuple)) and len(entry) > 0:
                if isinstance(entry[0], np.ndarray):
                    ht.append(entry[0])
                else:
                    ht.append(np.array(entry[0]))
            elif isinstance(entry, np.ndarray) and entry.ndim == 1:
                ht.append(entry)
        if ht:
            ht_arr = np.array(ht)
            ax.plot(ht_arr[:, 0], ht_arr[:, 1], "r--", linewidth=1.5, label="热点轨迹", alpha=0.8)
            ax.scatter(ht_arr[0, 0], ht_arr[0, 1], c="red", s=100, marker="*", zorder=5)

    if area_size:
        ax.set_xlim(0, area_size)
        ax.set_ylim(0, area_size)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or "智能体轨迹")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metrics(
    metrics_history: Dict[str, List[float]],
    metric_names: Optional[List[str]] = None,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制指标随时间变化的曲线。

    参数：
        metrics_history: 指标历史字典，键为指标名，值为数值列表
        metric_names: 要绘制的指标名称列表；若为 None 则绘制全部
        title: 图标题
        save_path: 若指定则保存到该路径
    返回：
        matplotlib Figure 对象
    """
    if metric_names is None:
        metric_names = list(metrics_history.keys())

    n = len(metric_names)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), squeeze=False)

    # 中文标签映射
    label_map = {
        "coverage": "覆盖率",
        "weighted_coverage": "加权覆盖率",
        "hotspot_distance": "与热点距离",
        "overlap": "重叠率",
    }

    for i, name in enumerate(metric_names):
        ax = axes[i][0]
        values = metrics_history.get(name, [])
        ax.plot(values, linewidth=1.5, color=f"C{i}")
        ax.set_ylabel(label_map.get(name, name))
        ax.set_xlabel("步数")
        ax.grid(True, alpha=0.3)
        ax.set_title(label_map.get(name, name))

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def animate_simulation(
    frames_data: List[Dict],
    grid_extent: List[float],
    coverage_radius: float = 1.0,
    save_path: Optional[str] = None,
    interval: int = 50,
) -> Optional[animation.FuncAnimation]:
    """
    制作仿真动画（可选保存为 gif）。

    参数：
        frames_data: 帧数据列表，每帧为字典，包含：
            - "field": 二维场值数组
            - "agent_positions": 智能体位置列表
            - "hotspot_centers": 热点中心列表
            - "step": 当前步数
        grid_extent: [xmin, xmax, ymin, ymax]
        coverage_radius: 覆盖半径
        save_path: 若指定则保存 gif 到该路径
        interval: 帧间隔毫秒数
    返回：
        FuncAnimation 对象（若保存失败则返回 None）
    """
    if not frames_data:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))

    # 初始化场图
    frame0 = frames_data[0]
    im = ax.imshow(
        frame0["field"],
        extent=grid_extent,
        origin="lower",
        cmap="hot",
        interpolation="bilinear",
        vmin=0.0,
        vmax=1.0,
        animated=True,
    )

    # 智能体散点
    agent_scat = ax.scatter([], [], c="blue", s=80, zorder=5, label="智能体")
    # 热点中心散点
    hotspot_scat = ax.scatter([], [], c="red", s=200, marker="*", zorder=6, label="热点中心")

    ax.set_xlim(grid_extent[0], grid_extent[1])
    ax.set_ylim(grid_extent[2], grid_extent[3])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper left")
    title_text = ax.set_title("Step 0")

    def update(frame_idx):
        data = frames_data[frame_idx]
        im.set_array(data["field"])

        # 更新智能体位置
        positions = np.array(data["agent_positions"]) if data["agent_positions"] else np.empty((0, 2))
        agent_scat.set_offsets(positions)

        # 更新热点位置
        centers = np.array(data["hotspot_centers"]) if data["hotspot_centers"] else np.empty((0, 2))
        hotspot_scat.set_offsets(centers)

        title_text.set_text(f"Step {data.get('step', frame_idx)}")
        return im, agent_scat, hotspot_scat, title_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames_data),
        interval=interval,
        blit=True,
    )

    if save_path:
        try:
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        except Exception as e:
            print(f"[viz] 动画保存失败：{e}")

    plt.tight_layout()
    return anim
