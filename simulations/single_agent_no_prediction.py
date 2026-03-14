"""
仿真主程序：动态单热点、单智能体、无预测覆盖控制
直接运行: python simulations/single_agent_no_prediction.py
"""
import sys
import os

# 无GUI环境支持
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.dynamic_field import DynamicSensitivityField
from src.agents.single_uav import SingleUAV
from src.control.reactive_controller import ReactiveController
from src.metrics.coverage_metrics import CoverageMetrics

# ── 仿真参数 ──────────────────────────────────────────────────────────────────
DT = 0.1          # 时间步长 (s)
TOTAL_TIME = 100.0  # 总时长 (s)
DOMAIN = (0.0, 100.0, 0.0, 100.0)
INITIAL_POSITION = [30.0, 30.0]  # 智能体初始位置

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_simulation():
    """运行仿真并返回记录的指标"""

    # 初始化各模块
    field = DynamicSensitivityField(domain=DOMAIN, resolution=50)
    agent = SingleUAV(
        initial_position=INITIAL_POSITION,
        max_velocity=5.0,
        sensing_radius=15.0,
    )
    controller = ReactiveController(
        sensing_radius=15.0,
        max_velocity=5.0,
        gradient_gain=1.0,
        local_max_gain=0.5,
        boundary_margin=10.0,
        boundary_gain=3.0,
        num_directions=12,
        num_radii=3,
        domain=DOMAIN,
    )
    metrics = CoverageMetrics(
        domain=DOMAIN,
        resolution=50,
        sensing_radius=15.0,
    )

    # 记录指标
    times = []
    coverages = []
    costs = []
    hotspot_distances = []
    hotspot_trajectory = []  # 记录热点运动轨迹

    n_steps = int(TOTAL_TIME / DT)

    print(f"开始仿真：总时长 {TOTAL_TIME}s，步长 {DT}s，共 {n_steps} 步")

    for step in range(n_steps):
        t = step * DT

        # 1. 更新敏感度场（移动热点）
        field.update(DT)

        # 2. 记录热点位置
        hotspot_trajectory.append(field.get_hotspot_position())

        # 3. 计算控制速度（无预测，纯反应式）
        velocity = controller.compute_control(agent.position, field)

        # 4. 更新智能体位置
        agent.set_velocity(velocity, DT)

        # 5. 记录指标
        coverage = metrics.compute_weighted_coverage(agent.position, field)
        cost = metrics.compute_coverage_cost(agent.position, field)
        dist = metrics.compute_hotspot_distance(agent.position, field)

        times.append(t)
        coverages.append(coverage)
        costs.append(cost)
        hotspot_distances.append(dist)

        # 每500步打印进度
        if (step + 1) % 500 == 0:
            print(f"  t={t+DT:.1f}s  覆盖率={coverage:.3f}  热点距离={dist:.2f}m")

    print("仿真完成！")

    return {
        'times': np.array(times),
        'coverages': np.array(coverages),
        'costs': np.array(costs),
        'hotspot_distances': np.array(hotspot_distances),
        'agent_trajectory': agent.get_trajectory(),
        'hotspot_trajectory': np.array(hotspot_trajectory),
        'field': field,
    }


def plot_results(results):
    """绘制4张子图并保存"""
    times = results['times']
    coverages = results['coverages']
    costs = results['costs']
    hotspot_distances = results['hotspot_distances']
    agent_traj = results['agent_trajectory']
    hotspot_traj = results['hotspot_trajectory']
    field = results['field']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('动态单热点·单智能体·无预测 覆盖控制仿真', fontsize=14)

    # ── 子图1：最终敏感度场 + 轨迹 ─────────────────────────────────────────
    ax1 = axes[0, 0]
    field_grid, xx, yy = field.get_field_grid()
    im = ax1.contourf(xx, yy, field_grid, levels=20, cmap='YlOrRd')
    plt.colorbar(im, ax=ax1, label='敏感度')

    # 热点运动轨迹（圆形虚线参考）
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = field.origin[0] + field.amplitude * np.cos(theta)
    circle_y = field.origin[1] + field.amplitude * np.sin(theta)
    ax1.plot(circle_x, circle_y, 'b--', linewidth=1, alpha=0.5, label='热点轨迹（参考圆）')

    # 智能体轨迹
    ax1.plot(agent_traj[:, 0], agent_traj[:, 1], 'g-', linewidth=1.5, alpha=0.7, label='智能体轨迹')
    ax1.plot(agent_traj[0, 0], agent_traj[0, 1], 'go', markersize=8, label='起点')
    ax1.plot(agent_traj[-1, 0], agent_traj[-1, 1], 'g^', markersize=8, label='终点')

    # 最终时刻热点位置
    final_hotspot = hotspot_traj[-1]
    ax1.plot(final_hotspot[0], final_hotspot[1], 'r*', markersize=12, label='热点（最终）')

    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('最终敏感度场与智能体轨迹')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.set_aspect('equal')

    # ── 子图2：覆盖率随时间变化 ───────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.plot(times, coverages, 'b-', linewidth=1.5)
    ax2.axhline(y=np.mean(coverages), color='r', linestyle='--',
                label=f'平均={np.mean(coverages):.3f}')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('加权覆盖率')
    ax2.set_title('覆盖率随时间变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # ── 子图3：覆盖代价随时间变化 ─────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.plot(times, costs, 'orange', linewidth=1.5)
    ax3.axhline(y=np.mean(costs), color='r', linestyle='--',
                label=f'平均={np.mean(costs):.1f}')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('覆盖代价 H')
    ax3.set_title('覆盖代价随时间变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ── 子图4：热点追踪距离随时间变化 ─────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.plot(times, hotspot_distances, 'purple', linewidth=1.5)
    ax4.axhline(y=np.mean(hotspot_distances), color='r', linestyle='--',
                label=f'平均={np.mean(hotspot_distances):.2f}m')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('到热点距离 (m)')
    ax4.set_title('热点追踪距离随时间变化')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'single_agent_no_prediction.png')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"结果图已保存至: {output_path}")


def print_statistics(results):
    """打印统计信息"""
    coverages = results['coverages']
    costs = results['costs']
    hotspot_distances = results['hotspot_distances']

    print("\n── 统计信息 ──────────────────────────────────────")
    print(f"  平均覆盖率:       {np.mean(coverages):.4f}")
    print(f"  最大覆盖率:       {np.max(coverages):.4f}")
    print(f"  平均热点追踪距离: {np.mean(hotspot_distances):.2f} m")
    print(f"  最小热点距离:     {np.min(hotspot_distances):.2f} m")
    print(f"  平均覆盖代价:     {np.mean(costs):.2f}")
    print(f"  最小覆盖代价:     {np.min(costs):.2f}")
    print("──────────────────────────────────────────────────")


if __name__ == '__main__':
    results = run_simulation()
    print_statistics(results)
    plot_results(results)
