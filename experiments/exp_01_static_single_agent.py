"""
experiments/exp_01_static_single_agent.py
实验 01：单智能体 + 静态热点

验证问题：
    在一个静态热点场里，单个反应式智能体是否真的按预期朝热点运动？
    实验结果是否可信（覆盖率提升、距离下降）？

运行方式：
    python -m experiments.exp_01_static_single_agent
"""

import os
import sys
import numpy as np

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.base import BaseConfig
from envs.field import SensitivityField
from agents.dynamics import AgentDynamics
from controllers.reactive import ReactiveController
from metrics.coverage_metrics import CoverageTracker
from utils.viz import plot_field, plot_trajectory, plot_metrics


def run_experiment():
    """运行实验 01：单智能体 + 静态热点"""

    # ------------------------------------------------------------------ #
    # 1. 加载配置（覆盖为静态、单智能体）
    # ------------------------------------------------------------------ #
    config = BaseConfig(
        area_size=10.0,
        dt=0.1,
        steps=200,
        n_agents=1,
        max_speed=1.0,
        sigma=1.5,
        seed=42,
        coverage_radius=1.0,
        hotspot_centers=[[7.0, 7.0]],   # 单个静态热点
        hotspot_speed=0.0,              # 静态
        hotspot_trajectory="static",
    )

    print("=" * 50)
    print("实验 01：单智能体 + 静态热点")
    print("=" * 50)
    print(f"区域大小: {config.area_size} x {config.area_size}")
    print(f"时间步长: {config.dt}，总步数: {config.steps}")
    print(f"热点位置: {config.hotspot_centers}")
    print(f"最大速度: {config.max_speed}，覆盖半径: {config.coverage_radius}")
    print()

    # ------------------------------------------------------------------ #
    # 2. 创建环境：静态热点场
    # ------------------------------------------------------------------ #
    field = SensitivityField(config)
    grid = (field.xx, field.yy)  # 网格坐标，供指标计算使用

    # ------------------------------------------------------------------ #
    # 3. 创建智能体：从远离热点的位置出发
    #    热点在 (7, 7)，初始位置设为 (2, 2)，较远
    # ------------------------------------------------------------------ #
    init_pos = [[2.0, 2.0]]
    agent = AgentDynamics(config, init_positions=init_pos)

    # ------------------------------------------------------------------ #
    # 4. 创建反应式控制器
    # ------------------------------------------------------------------ #
    controller = ReactiveController(config)

    # ------------------------------------------------------------------ #
    # 5. 创建指标追踪器
    # ------------------------------------------------------------------ #
    tracker = CoverageTracker()

    # 历史数据记录
    trajectory = []           # 智能体轨迹 [[x, y], ...]
    hotspot_traj = []         # 热点轨迹（静态情况下不变）
    frames_data = []          # 动画帧数据

    # ------------------------------------------------------------------ #
    # 6. 主仿真循环
    # ------------------------------------------------------------------ #
    print("开始仿真...")
    for step in range(config.steps):
        t = step * config.dt

        # 获取当前场值和热点位置
        field_values = field.get_field(t)
        hotspot_centers = field.get_hotspot_centers(t)
        agent_positions = agent.get_positions()

        # 记录轨迹
        trajectory.append(agent_positions[0].copy())
        hotspot_traj.append([c.copy() for c in hotspot_centers])

        # 控制器计算控制输入
        controls = controller.compute(agent_positions, field, t)

        # 智能体更新位置
        agent.step(controls)

        # 计算并记录指标
        metrics = tracker.record(
            agent_positions,
            field_values,
            grid,
            hotspot_centers,
            config.coverage_radius,
        )

        # 记录动画帧（每 10 步记录一帧，减少内存占用）
        if step % 10 == 0:
            frames_data.append({
                "field": field_values,
                "agent_positions": [p.copy() for p in agent_positions],
                "hotspot_centers": [c.copy() for c in hotspot_centers],
                "step": step,
            })

        # 每 50 步打印一次进度
        if step % 50 == 0:
            print(
                f"  Step {step:4d} | "
                f"覆盖率: {metrics['coverage']:.3f} | "
                f"加权覆盖率: {metrics['weighted_coverage']:.3f} | "
                f"热点距离: {metrics['hotspot_distance']:.3f}"
            )

    # ------------------------------------------------------------------ #
    # 7. 打印最终汇总指标
    # ------------------------------------------------------------------ #
    summary = tracker.summary()
    print()
    print("=" * 50)
    print("实验结果汇总")
    print("=" * 50)
    for key, val in summary.items():
        print(f"  {key}: {val:.4f}")
    print()

    # ------------------------------------------------------------------ #
    # 8. 可视化并保存图片
    # ------------------------------------------------------------------ #
    # 确保 results/ 目录存在
    results_dir = os.path.join(_PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("生成可视化图片...")

    # 最终场 + 智能体位置
    final_t = config.steps * config.dt
    final_field = field.get_field(final_t)
    final_positions = agent.get_positions()
    final_hotspots = field.get_hotspot_centers(final_t)

    fig1 = plot_field(
        final_field,
        field.grid_extent,
        agent_positions=final_positions,
        hotspot_centers=final_hotspots,
        coverage_radius=config.coverage_radius,
        title="实验01：最终状态（场 + 智能体位置）",
        save_path=os.path.join(results_dir, "exp01_final_field.png"),
    )
    plt_close(fig1)
    print(f"  已保存：results/exp01_final_field.png")

    # 智能体轨迹图
    fig2 = plot_trajectory(
        trajectory,
        hotspot_trajectory=hotspot_traj,
        area_size=config.area_size,
        title="实验01：智能体轨迹",
        save_path=os.path.join(results_dir, "exp01_trajectory.png"),
    )
    plt_close(fig2)
    print(f"  已保存：results/exp01_trajectory.png")

    # 指标曲线
    fig3 = plot_metrics(
        tracker.history,
        metric_names=["coverage", "weighted_coverage", "hotspot_distance"],
        title="实验01：覆盖指标随时间变化",
        save_path=os.path.join(results_dir, "exp01_metrics.png"),
    )
    plt_close(fig3)
    print(f"  已保存：results/exp01_metrics.png")

    print()
    print("实验完成！所有图片已保存到 results/ 目录。")
    return summary


def plt_close(fig):
    """关闭 matplotlib 图形，释放内存"""
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    run_experiment()
