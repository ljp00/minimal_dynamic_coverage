# minimal_dynamic_coverage

动态单热点、单智能体、无预测覆盖控制仿真。

## 项目描述

本项目实现了一个最小化的动态覆盖控制仿真框架：
- **单智能体**：1架无人机
- **单热点**：1个做圆周运动的动态高斯热点
- **无预测**：纯反应式/局部感知控制，不使用GP预测

## 使用方法

安装依赖：

```bash
pip install -r requirements.txt
```

运行仿真：

```bash
python simulations/single_agent_no_prediction.py
```

结果图将保存至 `output/single_agent_no_prediction.png`。

## 项目结构

```
minimal_dynamic_coverage/
├── config/
│   └── params.yaml                    # 仿真参数配置
├── output/                            # 结果输出目录
├── simulations/
│   └── single_agent_no_prediction.py  # 主仿真脚本
├── src/
│   ├── agents/
│   │   └── single_uav.py              # 单无人机模型
│   ├── control/
│   │   └── reactive_controller.py     # 反应式控制器
│   ├── environment/
│   │   └── dynamic_field.py           # 动态敏感度场
│   └── metrics/
│       └── coverage_metrics.py        # 覆盖率评估指标
└── requirements.txt
```

## 输出说明

仿真结束后输出4张子图：

1. **最终敏感度场**：显示热点位置、智能体轨迹及热点圆形参考轨迹
2. **覆盖率曲线**：加权覆盖率随时间的变化
3. **覆盖代价曲线**：覆盖代价 H = Σ s(q)·‖q-p‖² 随时间的变化
4. **热点追踪距离**：智能体到热点中心的距离随时间的变化