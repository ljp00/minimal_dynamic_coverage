# minimal_dynamic_coverage

最小可验证动态覆盖实验框架。

## 项目目标

搭建一条最小可验证实验链：

**环境生成 → 智能体运动 → 控制决策 → 指标计算 → 结果可视化 → 单个实验入口运行**

回答最基础的问题：在一个最简单的动态/静态热点覆盖任务里，智能体是否真的按预期运动，实验结果是否可信。

## 项目结构

```
minimal_dynamic_coverage/
├── configs/
│   └── base.py                    # 基础实验参数（区域大小、步长、速度、热点宽度等）
├── envs/
│   └── field.py                   # 敏感度场/热点场（高斯热点，支持静态和动态）
├── agents/
│   └── dynamics.py                # 智能体运动学模型（一阶积分器 + 速度限幅）
├── controllers/
│   └── reactive.py                # 反应式控制器（朝当前热点移动，无预测）
├── predictors/
│   └── oracle.py                  # 理想预测器（直接获取未来真实热点位置）
├── metrics/
│   └── coverage_metrics.py        # 覆盖率、加权覆盖率、热点距离等指标计算
├── utils/
│   └── viz.py                     # 热力图、轨迹图、指标曲线可视化
├── experiments/
│   └── exp_01_static_single_agent.py  # 实验01：单智能体 + 静态热点
├── results/                       # 实验结果图片（自动创建）
└── requirements.txt
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行第一个实验

```bash
python -m experiments.exp_01_static_single_agent
```

实验结果图片将保存到 `results/` 目录：

- `results/exp01_final_field.png`：最终时刻的敏感度场 + 智能体位置
- `results/exp01_trajectory.png`：智能体运动轨迹
- `results/exp01_metrics.png`：覆盖率、加权覆盖率、热点距离随时间变化曲线

## 实验链说明

```
configs/base.py
    ↓ 配置参数
envs/field.py          ← 热点场（静态/动态高斯热点）
    ↓ 场值 + 热点位置
agents/dynamics.py     ← 一阶积分器运动学
controllers/reactive.py ← 朝最近热点移动
    ↓ 位置序列
metrics/coverage_metrics.py ← 覆盖率、距离指标
utils/viz.py           ← 热力图、轨迹图、指标曲线
    ↓
experiments/exp_01_static_single_agent.py ← 实验入口
```

## 各模块说明

| 模块 | 说明 |
|------|------|
| `configs/base.py` | `BaseConfig` dataclass，统一管理实验参数 |
| `envs/field.py` | `SensitivityField`，高斯热点场，支持静态/圆周/线性运动 |
| `agents/dynamics.py` | `AgentDynamics`，一阶积分器，含速度限幅和边界裁剪 |
| `controllers/reactive.py` | `ReactiveController`，朝最近热点移动的反应式控制器 |
| `predictors/oracle.py` | `OraclePredictor`，理想预测器（验证预测价值的上界基准） |
| `metrics/coverage_metrics.py` | 纯函数指标计算 + `CoverageTracker` 追踪器 |
| `utils/viz.py` | `plot_field`/`plot_trajectory`/`plot_metrics`/`animate_simulation` |