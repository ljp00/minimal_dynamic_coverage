"""
反应式控制器模块
无预测，纯基于局部感知的梯度上升控制
"""
import numpy as np


class ReactiveController:
    """
    反应式局部感知控制器

    在感知半径内多方向采样，计算局部梯度方向，
    引导智能体向敏感度最高的方向移动。
    加入边界力防止智能体驶出工作域。
    """

    def __init__(
        self,
        sensing_radius=15.0,
        max_velocity=5.0,
        gradient_gain=1.0,
        local_max_gain=0.5,
        boundary_margin=10.0,
        boundary_gain=3.0,
        num_directions=12,
        num_radii=3,
        domain=(0, 100, 0, 100),
    ):
        """
        参数:
            sensing_radius: 感知半径 (m)
            max_velocity: 最大输出速度 (m/s)
            gradient_gain: 梯度方向增益
            local_max_gain: 局部最大值方向增益
            boundary_margin: 边界缓冲距离 (m)
            boundary_gain: 边界斥力强度
            num_directions: 采样方向数量
            num_radii: 采样半径层数
            domain: 工作域 (x_min, x_max, y_min, y_max)
        """
        self.sensing_radius = sensing_radius
        self.max_velocity = max_velocity
        self.gradient_gain = gradient_gain
        self.local_max_gain = local_max_gain
        self.boundary_margin = boundary_margin
        self.boundary_gain = boundary_gain
        self.num_directions = num_directions
        self.num_radii = num_radii
        self.x_min, self.x_max, self.y_min, self.y_max = domain

        # 预计算方向角
        self._angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
        # 采样半径（均匀分布在感知范围内）
        self._radii = np.linspace(
            sensing_radius / num_radii, sensing_radius, num_radii
        )

    def _sample_sensing_points(self, position):
        """
        在感知范围内均匀采样点

        返回:
            sample_points: (N, 2) 采样位置
            sample_dirs: (N, 2) 对应单位方向向量
        """
        sample_points = []
        sample_dirs = []
        for r in self._radii:
            for angle in self._angles:
                direction = np.array([np.cos(angle), np.sin(angle)])
                point = position + r * direction
                sample_points.append(point)
                sample_dirs.append(direction)
        return np.array(sample_points), np.array(sample_dirs)

    def _compute_local_gradient(self, position, sample_points, sample_values, center_value):
        """
        根据采样点计算局部梯度

        使用加权方向平均：权重为采样点相对于当前位置的密度差值（取正部分）

        参数:
            center_value: 当前位置处的实际密度值，作为梯度基准
        """
        # 各采样点到智能体的方向向量
        deltas = sample_points - position  # (N, 2)
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1e-8, norms)
        unit_dirs = deltas / norms  # (N, 2)

        # 权重 = max(0, 采样值 - 当前位置密度值)
        weights = np.maximum(0, sample_values - center_value)
        weight_sum = weights.sum()

        if weight_sum < 1e-10:
            return np.zeros(2)

        gradient = (weights[:, None] * unit_dirs).sum(axis=0) / weight_sum
        return gradient

    def _find_local_maximum_direction(self, position, sample_points, sample_values):
        """找到感知范围内密度最大的采样点方向"""
        best_idx = np.argmax(sample_values)
        best_point = sample_points[best_idx]
        direction = best_point - position
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return np.zeros(2)
        return direction / norm

    def _compute_boundary_force(self, position):
        """
        计算边界斥力，防止智能体驶出工作域

        在距离边界 boundary_margin 以内时产生斥力
        """
        force = np.zeros(2)
        x, y = position

        # 左边界
        dist_left = x - self.x_min
        if dist_left < self.boundary_margin:
            force[0] += self.boundary_gain * (1 - dist_left / self.boundary_margin)

        # 右边界
        dist_right = self.x_max - x
        if dist_right < self.boundary_margin:
            force[0] -= self.boundary_gain * (1 - dist_right / self.boundary_margin)

        # 下边界
        dist_bottom = y - self.y_min
        if dist_bottom < self.boundary_margin:
            force[1] += self.boundary_gain * (1 - dist_bottom / self.boundary_margin)

        # 上边界
        dist_top = self.y_max - y
        if dist_top < self.boundary_margin:
            force[1] -= self.boundary_gain * (1 - dist_top / self.boundary_margin)

        return force

    def _limit_velocity(self, velocity):
        """速度限幅"""
        speed = np.linalg.norm(velocity)
        if speed > self.max_velocity:
            velocity = velocity / speed * self.max_velocity
        return velocity

    def compute_control(self, position, field):
        """
        计算控制速度

        参数:
            position: (2,) 智能体当前位置
            field: 具有 get_density 方法的敏感度场对象

        返回:
            velocity: (2,) 控制速度向量
        """
        # 在感知范围内采样
        sample_points, _ = self._sample_sensing_points(position)
        sample_values = field.get_density(sample_points)

        # 获取当前位置的实际密度值作为梯度基准
        center_value = field.get_density(position)[0]

        # 计算梯度方向
        gradient = self._compute_local_gradient(position, sample_points, sample_values, center_value)

        # 找局部最大值方向
        local_max_dir = self._find_local_maximum_direction(
            position, sample_points, sample_values
        )

        # 合成速度：梯度上升 + 局部最大方向
        velocity = self.gradient_gain * gradient + self.local_max_gain * local_max_dir

        # 加入边界斥力
        velocity += self._compute_boundary_force(position)

        # 速度限幅
        velocity = self._limit_velocity(velocity)

        return velocity
