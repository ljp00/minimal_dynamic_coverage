"""
predictors/oracle.py
提供理想未来信息的 Oracle 预测器（上界基准）。
直接从场模型获取未来真实热点位置，用于验证预测信息是否有价值。
"""

import numpy as np
from typing import List


class OraclePredictor:
    """
    理想预测器（作弊预测器）：直接查询场模型获取未来真实热点位置。

    用途：
        - 验证"如果预测完美，控制框架能否利用预测带来提升"
        - 作为预测性能的理论上界
    """

    def __init__(self, field):
        """
        参数：
            field: SensitivityField 实例
        """
        self.field = field

    def predict(self, t: float, horizon: float) -> List[np.ndarray]:
        """
        返回未来 t + horizon 时刻的真实热点中心位置列表。

        参数：
            t: 当前时刻
            horizon: 预测时域（秒）
        返回：
            未来热点中心位置列表
        """
        return self.field.get_future_hotspot_centers(t, horizon)
