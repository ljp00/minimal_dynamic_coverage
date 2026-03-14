# minimal_dynamic_coverage

A minimal, self-contained framework for dynamic single-hotspot coverage-control
experiments. No GP prediction, no Voronoi partitioning, no CBF/CVXPY — just
`numpy`, `matplotlib`, and `scipy`.

---

## Project Structure

```
minimal_dynamic_coverage/
├── src/
│   ├── environment/
│   │   └── dynamic_field.py      # DynamicSensitivityField (single Gaussian hotspot)
│   ├── agents/
│   │   └── single_uav.py         # SingleUAV (first-order kinematics)
│   ├── control/
│   │   └── reactive_controller.py # ReactiveController & OracleController
│   └── metrics/
│       └── coverage_metrics.py   # CoverageMetrics (weighted coverage, cost, distance)
├── simulations/
│   └── single_agent_reactive_vs_oracle.py  # Main experiment script
├── output/                        # Generated figures and animations (git-ignored)
└── README.md
```

---

## Experiments

### Single-agent Reactive vs Oracle

**Reactive** — no prediction; tracks the *current* hotspot position.  
**Oracle** — perfect predictor; tracks the hotspot position `horizon` seconds
in the future. Acts as a theoretical upper bound.

#### Quick start

```bash
# Install dependencies (if not already installed)
pip install numpy matplotlib scipy

# Run with default settings (T=100 s, dt=0.1 s, horizon=0.5 s)
python simulations/single_agent_reactive_vs_oracle.py
```

#### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--time T` | 100.0 | Total simulation time (s) |
| `--dt DT` | 0.1 | Time step (s) |
| `--horizon H` | 0.5 | Oracle look-ahead horizon (s) |
| `--seed S` | 42 | Random seed |
| `--no-animation` | — | Skip GIF generation (faster) |
| `--fps F` | 10 | Frames per second for the animation |
| `--show` | — | Display figures interactively |

#### Example: longer horizon, no animation, show figure

```bash
python simulations/single_agent_reactive_vs_oracle.py \
    --time 200 --horizon 1.0 --no-animation --show
```

#### Output Files

| File | Description |
|------|-------------|
| `output/reactive_vs_oracle_results.png` | 3×2 panel static figure: field + trajectories, coverage rate, coverage cost, hotspot distance, speed, sensed value |
| `output/reactive_vs_oracle_animation.gif` | Animated simulation (omitted with `--no-animation`) |

#### Terminal Output

After each simulation run the script prints a comparison table like:

```
----------------------------------------------------------------------
  REACTIVE vs ORACLE — SUMMARY STATISTICS
----------------------------------------------------------------------
Metric                    Reactive              Oracle     Improvement
----------------------------------------------------------------------
Weighted Coverage    μ=0.312 σ=0.051    μ=0.378 σ=0.040       +21.2%
Coverage Cost        μ=0.688 σ=0.051    μ=0.622 σ=0.040       -21.2%
Hotspot Distance     μ=8.432 σ=3.201    μ=5.214 σ=2.876       -38.2%
Speed                μ=7.821 σ=2.103    μ=7.615 σ=2.021            —
Sensed Value         μ=0.698 σ=0.201    μ=0.812 σ=0.182       +16.3%
----------------------------------------------------------------------
```

---

## Design Principles

- **Minimal dependencies**: `numpy`, `matplotlib` (and optionally `scipy`)
- **No GP, no Voronoi, no CBF, no CVXPY**
- **Headless-friendly**: defaults to `matplotlib.use('Agg')`; use `--show` for interactive display
- **Reproducible**: fixed random seed via `--seed`
