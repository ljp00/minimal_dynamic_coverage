"""
仿真主程序：动态单热点、单智能体、无预测覆盖控制
用法: python simulations/single_agent_no_prediction.py [--time 100] [--dt 0.1] [--no-animation] [--fps 10] [--show]
"""
import sys
import os
import argparse

# 在导入其他 matplotlib 模块之前设置后端（无GUI环境支持）
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.dynamic_field import DynamicSensitivityField
from src.agents.single_uav import SingleUAV
from src.control.reactive_controller import ReactiveController
from src.metrics.coverage_metrics import CoverageMetrics


class SingleAgentSimulation:
    """单智能体无预测覆盖控制仿真"""

    def __init__(self, total_time=100.0, dt=0.1):
        self.total_time = total_time
        self.dt = dt
        self.domain = (0.0, 100.0, 0.0, 100.0)

        # 输出目录
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化各模块
        self.field = DynamicSensitivityField(domain=self.domain, resolution=50)
        self.agent = SingleUAV(
            initial_position=[30.0, 30.0],
            max_velocity=5.0,
            sensing_radius=15.0,
        )
        self.controller = ReactiveController(
            sensing_radius=15.0,
            max_velocity=5.0,
            gradient_gain=1.0,
            local_max_gain=0.5,
            boundary_margin=10.0,
            boundary_gain=3.0,
            num_directions=12,
            num_radii=3,
            domain=self.domain,
        )
        self.metrics = CoverageMetrics(
            domain=self.domain,
            resolution=50,
            sensing_radius=15.0,
        )

        # 历史数据记录（每2步记录一次，减少数据量）
        self.history = {
            'time': [],
            'agent_position': [],       # 每步智能体位置
            'hotspot_position': [],     # 每步热点位置
            'coverage': [],             # 加权覆盖率
            'coverage_cost': [],        # 覆盖代价
            'hotspot_distance': [],     # 热点追踪距离
            'field_snapshots': [],      # 敏感度场快照（用于动画）
            'agent_speed': [],          # 智能体瞬时速度
            'sensed_value': [],         # 智能体当前位置感知值
        }

        self._step_count = 0

    def step(self):
        """执行一步仿真，返回当前时间"""
        t = self._step_count * self.dt

        # 1. 更新敏感度场（移动热点）
        self.field.update(self.dt)

        # 2. 计算控制速度（无预测，纯反应式）
        velocity = self.controller.compute_control(self.agent.position, self.field)
        speed = float(np.linalg.norm(velocity))

        # 3. 更新智能体位置
        self.agent.set_velocity(velocity, self.dt)

        # 4. 感知当前位置场值
        _, sensed_value = self.agent.sense(self.field)

        # 5. 每2步记录一次（减少数据量）
        if self._step_count % 2 == 0:
            coverage = self.metrics.compute_weighted_coverage(self.agent.position, self.field)
            cost = self.metrics.compute_coverage_cost(self.agent.position, self.field)
            dist = self.metrics.compute_hotspot_distance(self.agent.position, self.field)

            self.history['time'].append(t)
            self.history['agent_position'].append(self.agent.position.copy())
            self.history['hotspot_position'].append(self.field.get_hotspot_position())
            self.history['coverage'].append(coverage)
            self.history['coverage_cost'].append(cost)
            self.history['hotspot_distance'].append(dist)
            self.history['agent_speed'].append(speed)
            self.history['sensed_value'].append(float(sensed_value))

            # 保存场快照（供动画使用）
            field_grid, _, _ = self.field.get_field_grid()
            self.history['field_snapshots'].append({
                'time': t,
                'field': field_grid.copy(),
                'history_idx': len(self.history['time']) - 1,
            })

        self._step_count += 1
        return t

    def run(self):
        """运行完整仿真"""
        n_steps = int(self.total_time / self.dt)
        print(f"开始仿真：总时长 {self.total_time}s，步长 {self.dt}s，共 {n_steps} 步")

        for step in range(n_steps):
            t = self.step()

            # 每500步打印进度
            if (step + 1) % 500 == 0:
                h = self.history
                if len(h['coverage']) > 0:
                    cov = h['coverage'][-1]
                    dist = h['hotspot_distance'][-1]
                    print(f"  t={t + self.dt:.1f}s  覆盖率={cov:.3f}  热点距离={dist:.2f}m")

        print("仿真完成！")

        # 转换为 numpy 数组
        for key in ['time', 'agent_position', 'hotspot_position', 'coverage',
                    'coverage_cost', 'hotspot_distance', 'agent_speed', 'sensed_value']:
            self.history[key] = np.array(self.history[key])

    def print_statistics(self):
        """打印详细统计信息"""
        h = self.history
        times = h['time']
        coverages = h['coverage']
        costs = h['coverage_cost']
        distances = h['hotspot_distance']
        speeds = h['agent_speed']
        sensed = h['sensed_value']

        half = len(times) // 2
        first_dist = np.mean(distances[:half])
        second_dist = np.mean(distances[half:])
        first_cov = np.mean(coverages[:half])
        second_cov = np.mean(coverages[half:])
        improvement = (first_dist - second_dist) / max(first_dist, 1e-6) * 100

        sep = "=" * 68
        line = "-" * 68
        print()
        print(sep)
        print("SIMULATION STATISTICS")
        print(sep)
        print(f"Total time:          {self.total_time}s")
        print(f"Time step:           {self.dt}s")
        print(f"Motion type:         circular")
        print(line)
        print(f"{'Metric':<24}{'Mean':<12}{'Std':<12}{'Min':<12}{'Max':<12}")
        print(line)

        def _row(name, data, fmt):
            """打印一行统计数据"""
            vals = [fmt.format(v) for v in
                    [np.mean(data), np.std(data), np.min(data), np.max(data)]]
            print(f"{name:<24}{vals[0]:<12}{vals[1]:<12}{vals[2]:<12}{vals[3]:<12}")

        _row("Coverage Rate", coverages, "{:.3f}")
        _row("Coverage Cost", costs, "{:.0f}")
        _row("Hotspot Distance", distances, "{:.2f}m")
        _row("Agent Speed", speeds, "{:.2f}m/s")
        _row("Sensed Value", sensed, "{:.3f}")
        print(line)
        print()
        print("Performance by Phase:")
        print(f"  First half  (0-{self.total_time / 2:.0f}s):  "
              f"avg_dist={first_dist:.2f}m  avg_coverage={first_cov:.3f}")
        print(f"  Second half ({self.total_time / 2:.0f}-{self.total_time:.0f}s): "
              f"avg_dist={second_dist:.2f}m  avg_coverage={second_cov:.3f}")
        print(f"  Improvement: {improvement:.1f}%")
        print(sep)

    @staticmethod
    def _moving_average(data, window=20):
        """使用 np.convolve 计算移动平均"""
        w = min(window, len(data))
        kernel = np.ones(w) / w
        return np.convolve(data, kernel, mode='same')

    @staticmethod
    def _make_colored_segments(traj):
        """将轨迹转换为 LineCollection 线段格式，颜色按时间进度编码"""
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    def plot_results(self, show=False):
        """绘制6子图主结果图（3x2 布局）并保存"""
        h = self.history
        times = h['time']
        coverages = h['coverage']
        costs = h['coverage_cost']
        distances = h['hotspot_distance']
        speeds = h['agent_speed']
        sensed = h['sensed_value']
        agent_traj = self.agent.get_trajectory()
        hotspot_traj = h['hotspot_position']

        # 移动平均窗口：约取数据总长的 1/20，使曲线平滑同时保留趋势
        window = max(5, len(times) // 20)
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('动态单热点·单智能体·无预测 覆盖控制仿真结果',
                     fontsize=16, fontweight='bold')

        # ── 子图1：敏感度场热力图 + 轨迹 ─────────────────────────────────────
        ax1 = axes[0, 0]
        field_grid, _, _ = self.field.get_field_grid()
        im = ax1.imshow(
            field_grid,
            extent=[0, 100, 0, 100],
            origin='lower',
            cmap='YlOrRd',
            aspect='auto',
        )
        plt.colorbar(im, ax=ax1, label='敏感度密度')

        # 热点运动圆的理论轨迹（灰色虚线）
        theta = np.linspace(0, 2 * np.pi, 300)
        ax1.plot(
            self.field.origin[0] + self.field.amplitude * np.cos(theta),
            self.field.origin[1] + self.field.amplitude * np.sin(theta),
            '--', color='gray', linewidth=1.0, alpha=0.6, label='热点理论轨迹',
        )

        # 热点运动轨迹（红色虚线）
        ax1.plot(hotspot_traj[:, 0], hotspot_traj[:, 1],
                 'r--', linewidth=1.2, alpha=0.6, label='热点轨迹')

        # 智能体完整轨迹（彩色线段，颜色编码时间进度）
        if len(agent_traj) > 1:
            segments = self._make_colored_segments(agent_traj)
            n_seg = len(segments)
            colors = plt.cm.Blues(np.linspace(0.3, 1.0, n_seg))
            lc = LineCollection(segments, colors=colors, linewidth=1.5, alpha=0.8)
            ax1.add_collection(lc)

        # 起点（方形标记）
        ax1.plot(agent_traj[0, 0], agent_traj[0, 1], 's', color='blue',
                 markersize=8, label='起点', zorder=5)
        # 最终智能体位置（大三角标记）
        ax1.plot(agent_traj[-1, 0], agent_traj[-1, 1], '^', color='blue',
                 markersize=12, label='终点', zorder=6)
        # 最终热点位置（红色星号）
        ax1.plot(hotspot_traj[-1, 0], hotspot_traj[-1, 1], 'r*',
                 markersize=14, label='热点（最终）', zorder=6)

        # 智能体感知范围圆（蓝色虚线）
        sensing_circle = plt.Circle(
            agent_traj[-1], self.agent.sensing_radius,
            color='blue', fill=False, linestyle='--', linewidth=1.2, alpha=0.7,
        )
        ax1.add_patch(sensing_circle)

        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_xlabel('x (m)', fontsize=11)
        ax1.set_ylabel('y (m)', fontsize=11)
        ax1.set_title('最终敏感度场与智能体轨迹', fontweight='bold', fontsize=12)
        ax1.legend(loc='upper right', fontsize=7)
        ax1.grid(True, alpha=0.2)

        # ── 子图2：覆盖率随时间变化 ──────────────────────────────────────────
        ax2 = axes[0, 1]
        ax2.plot(times, coverages, color='steelblue', linewidth=1.0,
                 alpha=0.3, label='原始数据')
        ax2.plot(times, self._moving_average(coverages, window),
                 color='steelblue', linewidth=2.0, label='移动平均')
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('时间 (s)', fontsize=11)
        ax2.set_ylabel('加权覆盖率', fontsize=11)
        ax2.set_title('覆盖率随时间变化', fontweight='bold', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ── 子图3：覆盖代价随时间变化 ────────────────────────────────────────
        ax3 = axes[1, 0]
        ax3.plot(times, costs, color='darkorange', linewidth=1.0,
                 alpha=0.3, label='原始数据')
        ax3.plot(times, self._moving_average(costs, window),
                 color='darkorange', linewidth=2.0, label='移动平均')
        ax3.set_xlabel('时间 (s)', fontsize=11)
        ax3.set_ylabel('覆盖代价 H', fontsize=11)
        ax3.set_title('覆盖代价随时间变化', fontweight='bold', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # ── 子图4：热点追踪距离随时间变化 ────────────────────────────────────
        ax4 = axes[1, 1]
        ax4.plot(times, distances, color='purple', linewidth=1.0,
                 alpha=0.3, label='原始数据')
        ax4.plot(times, self._moving_average(distances, window),
                 color='purple', linewidth=2.0, label='移动平均')
        ax4.axhline(y=10.0, color='red', linestyle='--', linewidth=1.2,
                    alpha=0.7, label='10m 阈值')
        ax4.set_xlabel('时间 (s)', fontsize=11)
        ax4.set_ylabel('到热点距离 (m)', fontsize=11)
        ax4.set_title('热点追踪距离随时间变化', fontweight='bold', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        # ── 子图5：智能体速度随时间变化 ──────────────────────────────────────
        ax5 = axes[2, 0]
        ax5.plot(times, speeds, color='teal', linewidth=1.0,
                 alpha=0.3, label='原始数据')
        ax5.plot(times, self._moving_average(speeds, window),
                 color='teal', linewidth=2.0, label='移动平均')
        ax5.set_xlabel('时间 (s)', fontsize=11)
        ax5.set_ylabel('速度 (m/s)', fontsize=11)
        ax5.set_title('智能体速度随时间变化', fontweight='bold', fontsize=12)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        # ── 子图6：感知值随时间变化 ───────────────────────────────────────────
        ax6 = axes[2, 1]
        ax6.plot(times, sensed, color='green', linewidth=1.0,
                 alpha=0.3, label='原始数据')
        ax6.plot(times, self._moving_average(sensed, window),
                 color='green', linewidth=2.0, label='移动平均')
        ax6.set_xlabel('时间 (s)', fontsize=11)
        ax6.set_ylabel('感知场值', fontsize=11)
        ax6.set_title('感知值随时间变化', fontweight='bold', fontsize=12)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'single_agent_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        print(f"结果图已保存至: {output_path}")

    def create_animation(self, fps=10, show=False):
        """创建仿真动画 GIF（两子图并排：地图视图 + 实时指标）"""
        snapshots = self.history['field_snapshots']
        if not snapshots:
            print("无场快照数据，跳过动画生成")
            return

        h = self.history
        times = h['time']
        agent_positions = h['agent_position']
        hotspot_positions = h['hotspot_position']
        coverages = h['coverage']
        costs = h['coverage_cost']
        distances = h['hotspot_distance']

        # 归一化覆盖代价和热点距离供右图显示
        max_cost = max(float(np.max(costs)), 1e-6)
        max_dist = max(float(np.max(distances)), 1e-6)
        norm_costs = costs / max_cost
        norm_dists = distances / max_dist

        # 抽帧：skip = max(1, n // 300)，最终帧数约为 min(n, 300)
        num_frames = len(snapshots)
        skip = max(1, num_frames // 300)
        selected = snapshots[::skip]
        print(f"动画帧数：{len(selected)}（共 {num_frames} 帧，skip={skip}）")

        # 预计算场的全局值域（保持色标一致）
        all_fields = np.array([s['field'] for s in snapshots])
        vmin, vmax = float(all_fields.min()), float(all_fields.max())

        tail_length = 50  # 左图（地图视图）显示最近 N 个历史记录点的智能体轨迹

        fig, (ax_map, ax_metrics) = plt.subplots(1, 2, figsize=(14, 6))

        def draw_frame(snap):
            """绘制单帧动画"""
            hist_idx = snap['history_idx']
            t = snap['time']
            field_grid = snap['field']
            end_idx = hist_idx + 1  # 切片上界（不含）

            # ── 左图：地图视图 ────────────────────────────────────────────────
            ax_map.cla()
            ax_map.imshow(
                field_grid,
                extent=[0, 100, 0, 100],
                origin='lower',
                cmap='YlOrRd',
                alpha=0.6,
                vmin=vmin,
                vmax=vmax,
                aspect='auto',
            )

            # 热点运动轨迹（红色虚线）
            if end_idx > 1:
                ax_map.plot(hotspot_positions[:end_idx, 0],
                            hotspot_positions[:end_idx, 1],
                            'r--', linewidth=1.0, alpha=0.7, label='热点轨迹')

            # 最近 tail_length 步的智能体轨迹（蓝色实线）
            tail_start = max(0, end_idx - tail_length)
            if end_idx > tail_start + 1:
                ax_map.plot(agent_positions[tail_start:end_idx, 0],
                            agent_positions[tail_start:end_idx, 1],
                            'b-', linewidth=2.0, alpha=0.7, label='智能体轨迹')

            # 当前智能体位置（蓝色三角，黑色边框）
            cur_pos = agent_positions[hist_idx]
            ax_map.plot(cur_pos[0], cur_pos[1], '^', color='blue',
                        markersize=12, markeredgecolor='black',
                        markeredgewidth=1.5, label='智能体', zorder=6)

            # 当前热点位置（红色星号）
            cur_hot = hotspot_positions[hist_idx]
            ax_map.plot(cur_hot[0], cur_hot[1], 'r*', markersize=14,
                        label='热点', zorder=6)

            # 智能体感知范围圆（蓝色虚线）
            circle = plt.Circle(
                cur_pos, self.agent.sensing_radius,
                color='blue', fill=False, linestyle='--', linewidth=1.2, alpha=0.7,
            )
            ax_map.add_patch(circle)

            # 智能体到热点的连线（绿色虚线，标注距离）
            dist_val = distances[hist_idx]
            ax_map.plot([cur_pos[0], cur_hot[0]], [cur_pos[1], cur_hot[1]],
                        'g--', linewidth=1.2, alpha=0.8)
            mid_x = (cur_pos[0] + cur_hot[0]) / 2
            mid_y = (cur_pos[1] + cur_hot[1]) / 2
            ax_map.text(mid_x, mid_y, f'{dist_val:.1f}m',
                        fontsize=8, color='green', ha='center')

            ax_map.set_xlim(0, 100)
            ax_map.set_ylim(0, 100)
            ax_map.set_xlabel('x (m)', fontsize=10)
            ax_map.set_ylabel('y (m)', fontsize=10)
            ax_map.set_title(
                f't = {t:.1f}s  热点距离 = {dist_val:.2f}m',
                fontweight='bold', fontsize=11,
            )
            ax_map.legend(loc='upper right', fontsize=7)

            # ── 右图：实时指标 ─────────────────────────────────────────────────
            ax_metrics.cla()
            plot_times = times[:end_idx]
            ax_metrics.plot(plot_times, coverages[:end_idx],
                            'b-', linewidth=1.5, label='覆盖率', alpha=0.9)
            ax_metrics.plot(plot_times, norm_costs[:end_idx],
                            color='darkorange', linewidth=1.5,
                            label='归一化覆盖代价', alpha=0.9)
            ax_metrics.plot(plot_times, norm_dists[:end_idx],
                            color='purple', linewidth=1.5,
                            label='归一化热点距离', alpha=0.9)
            ax_metrics.set_xlim(0, self.total_time)
            ax_metrics.set_ylim(0, 1.1)
            ax_metrics.set_xlabel('时间 (s)', fontsize=10)
            ax_metrics.set_ylabel('指标值（归一化）', fontsize=10)
            ax_metrics.set_title('实时指标', fontweight='bold', fontsize=11)
            ax_metrics.legend(fontsize=8, loc='upper left')
            ax_metrics.grid(True, alpha=0.3)

        ani = FuncAnimation(
            fig,
            func=lambda i: draw_frame(selected[i]),
            frames=len(selected),
            interval=1000 // fps,
            repeat=False,
        )

        output_path = os.path.join(self.output_dir, 'single_agent_animation.gif')
        writer = PillowWriter(fps=fps)
        print(f"正在生成动画，共 {len(selected)} 帧，fps={fps}...")
        ani.save(output_path, writer=writer)
        if show:
            plt.show()
        plt.close()
        print(f"动画已保存至: {output_path}")


def _parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='动态单热点单智能体无预测覆盖控制仿真'
    )
    parser.add_argument('--time', type=float, default=100.0,
                        help='总仿真时长，单位秒（默认100.0）')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='时间步长，单位秒（默认0.1）')
    parser.add_argument('--no-animation', action='store_true',
                        help='禁用动画生成')
    parser.add_argument('--fps', type=int, default=10,
                        help='动画帧率（默认10）')
    parser.add_argument('--show', action='store_true',
                        help='显示图表（默认只保存，不显示）')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    # 如果需要显示图表，尝试切换到交互式后端
    if args.show:
        try:
            matplotlib.use('TkAgg')
        except Exception:
            pass

    sim = SingleAgentSimulation(total_time=args.time, dt=args.dt)
    sim.run()
    sim.print_statistics()
    sim.plot_results(show=args.show)

    if not args.no_animation:
        sim.create_animation(fps=args.fps, show=args.show)
