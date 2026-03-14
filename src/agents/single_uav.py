"""
单智能体（无人机）模块
一阶运动模型，直接设置速度
"""
import numpy as np


class SingleUAV:
    """单架无人机，一阶运动模型"""

    def __init__(self, initial_position, max_velocity=5.0, sensing_radius=15.0):
        """
        参数:
            initial_position: (2,) 初始位置
            max_velocity: 最大速度 (m/s)
            sensing_radius: 感知半径 (m)
        """
        self.position = np.array(initial_position, dtype=float)
        self.v_max = max_velocity
        self.sensing_radius = sensing_radius

        # 记录轨迹历史
        self.trajectory = [self.position.copy()]

    def set_velocity(self, velocity, dt):
        """
        设置速度并更新位置

        参数:
            velocity: (2,) 速度向量
            dt: 时间步长 (s)
        """
        velocity = np.array(velocity, dtype=float)
        speed = np.linalg.norm(velocity)
        # 速度限幅
        if speed > self.v_max:
            velocity = velocity / speed * self.v_max
        self.position = self.position + velocity * dt
        self.trajectory.append(self.position.copy())

    def sense(self, field, time=None):
        """
        感知当前位置处的场值

        参数:
            field: 具有 get_density 方法的敏感度场对象
            time: 当前时间（保留接口，本模块未使用）

        返回:
            position: 当前位置副本
            value: 当前位置的敏感度值
        """
        value = field.get_density(self.position)[0]
        return self.position.copy(), value

    def get_trajectory(self):
        """返回历史轨迹数组，shape (T, 2)"""
        return np.array(self.trajectory)
