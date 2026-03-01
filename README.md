# Planar Pushing Tools

A Python package for planar pushing manipulation with online contact model learning and trajectory planning.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ with numpy and matplotlib. For video recording, ffmpeg must be installed.

## Demo Scripts

### demo_planars.py

Compares two trajectory planning methods for planar pushing: DDP (Differential Dynamic Programming) and Dubins curves.

**What it does:**
1. **Probing phase**: Executes alternating push actions to excite the system
2. **Model learning**: Identifies the contact model from observed motions using SVD
3. **Planning**: Generates trajectories using both DDP and Dubins planners
4. **Simulation**: Executes the DDP trajectory with feedback control
5. **Comparison**: Plots both trajectories side-by-side and reports planning time

**Output:**
- `pushing_demo_simple.mp4`: Animation of the pushing simulation
- Comparison plots showing XY trajectories, state evolution, and controls
- Planning time comparison (Dubins is typically 500-1000x faster than DDP)

```bash
python demo_planars.py
```

### demo_online_learning_replanning.py

Full closed-loop pushing controller with online model learning and replanning under a time-varying contact model.

**What it does:**
1. **Online identification**: Continuously learns the contact model from recent observations
2. **Adaptive planning**: Replans trajectory when model estimates change significantly
3. **Feedback control**: Executes planned trajectory with DDP-derived feedback gains
4. **State machine**: `PushDecision` coordinates between probing, learning, and planning phases

**Output:**
- `pushing_demo.mp4`: Animation of the pushing simulation
- Plots showing state trajectories, controls, feedback gains, and contact model evolution

```bash
python demo_online_learning_replanning.py
```

## File Structure

```
planar_pushing_tools/
├── demo_planars.py                 # Compare DDP vs Dubins planners
├── demo_online_learning_replanning.py  # Online learning + replanning demo
├── requirements.txt
├── README.md
└── planar_pushing_tools/           # Main package
    ├── __init__.py
    ├── config.py                   # OptsModel dataclass, contact model functions
    ├── model.py                    # Dynamics f_, Jacobians fx_, fu_, cost functions
    ├── ddp_solver.py               # DDP/iLQR solver with boxQP
    ├── dubins.py                   # Dubins curves solver (LSL, RSR, LSR, RSL, RLR, LRL)
    ├── push_planner.py             # PushPlanner: DDP-based trajectory optimization
    ├── push_planner_dubin.py       # PushPlannerDubin: Dubins path in flat output space
    ├── push_learner.py             # PushLearner: SVD-based contact model identification
    ├── push_decision.py            # PushDecision: state machine for probing/planning
    └── push_animation.py           # PushAnimation: matplotlib visualization + video
```

### Module Descriptions

| Module | Description |
|--------|-------------|
| `config.py` | `OptsModel` dataclass holding all model/cost parameters; `set_contact_model_b()` and `get_contact_model_b()` for contact model management |
| `model.py` | Pushing dynamics `f_(x, u, T, D_inv)`, Jacobians `fx_`, `fu_`, and cost functions `L_`, `Final_`, Hamiltonian derivatives for DDP |
| `ddp_solver.py` | Full DDP/iLQR implementation with backward pass, forward rollout, line search, regularization, and `box_qp()` for control bounds |
| `dubins.py` | Pure Python Dubins curves solver supporting all 6 path types |
| `push_planner.py` | `PushPlanner` class wrapping DDP with two-pass planning (exploration + tracking) |
| `push_planner_dubin.py` | `PushPlannerDubin` using differential flatness to plan in flat output space |
| `push_learner.py` | `PushLearner` with weighted SVD to identify contact normal from twist observations |
| `push_decision.py` | `PushDecision` state machine: state 0 = probing, state 1 = planning/control |
| `push_animation.py` | Real-time matplotlib animation with optional mp4 video export |

## License

This repository is released under the MIT license.

## Citation

The methods in this package implement:

```bibtex
@article{doi:10.1177/0278364919872532,
  author = {Jiaji Zhou and Yifan Hou and Matthew T Mason},
  title = {Pushing revisited: Differential flatness, trajectory planning, and stabilization},
  journal = {The International Journal of Robotics Research},
  volume = {38},
  number = {12-13},
  pages = {1477-1489},
  year = {2019},
  doi = {10.1177/0278364919872532},
}
```

This package was created by merging and translating the original MATLAB packages:
- [CloseLoopPushing](https://github.com/yifan-hou/CloseLoopPushing) - model learning, DDP planning & control
- [DDP](https://github.com/yifan-hou/DDP) - differential dynamic programming implementation
- [PlanarManipulationToolBox](https://github.com/robinzhoucmu/PlanarManipulationToolBox) - Dubins curves and differential flatness
