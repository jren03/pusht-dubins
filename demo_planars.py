"""Simplified demo: fixed model, learn once, plan once, simulate with feedback.

Compares DDP planner and Dubins planner trajectories side by side.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from planar_pushing_tools.config import OptsModel, set_contact_model_b, get_contact_model_b
from planar_pushing_tools.model import f_
from planar_pushing_tools.push_planner import PushPlanner
from planar_pushing_tools.push_planner_dubin import PushPlannerDubin
from planar_pushing_tools.push_learner import PushLearner
from planar_pushing_tools.push_animation import PushAnimation


def estimate_dubins_limit_surface_from_b(b, c, rho, eps=1e-9):
    """Estimate Dubins limit-surface coefficients from identified contact model.

    Dubins planner assumes A ~= diag(a_ls, a_ls, a_theta) and uses:
      A @ b ~= c,  b_ls = a_theta / rho^2.

    The identified b is normalized (scale-free), so only the ratio a_ls / b_ls
    is observable. We return a normalized equivalent pair with b_ls = 1.
    """
    b_xy = b[:2]
    c_xy = c[:2]
    denom_xy = float(np.dot(b_xy, b_xy))
    if denom_xy < eps:
        raise ValueError("Cannot estimate a_ls: planar components of b are too small.")
    planar_scale = float(np.dot(b_xy, c_xy) / denom_xy)

    if abs(float(b[2])) < eps:
        raise ValueError("Cannot estimate b_ls: rotational component b[2] is too small.")
    rotational_scale = float(c[2] / b[2])

    # ratio_ab = a_ls / b_ls = a * rho^2 / a_theta
    ratio_ab = abs((planar_scale / rotational_scale) * (rho ** 2))
    a_ls = ratio_ab
    b_ls = 1.0
    return a_ls, b_ls


def main():
    # ---------------------------------------------------------
    #       Simulation settings
    # ---------------------------------------------------------
    NH = 200   # length of horizon
    T = 0.1    # duration of a time step
    N_initial = 6  # number of probing steps for model learning

    # ---------------------------------------------------------
    #       Task setup
    # ---------------------------------------------------------
    x0 = np.array([-21.4 / 1000, -279.7 / 1000, 0.0]) # initial pose (x, y, theta)
    xstar = np.array([166.4 / 1000, -332.0 / 1000, 1.57]) # goal pose (x, y, theta)
    
    # ---------------------------------------------------------
    #       Model parameter
    # ---------------------------------------------------------
    # pushing action limits for DDP planner
    u_limit_ang = np.array([-65, 65]) * np.pi / 180
    u_limit_mag = np.array([0, 0.08])

    # Friction coefficient for Dubins planner
    mu = 0.3

    # point of contact in object local frame
    pt = np.array([-35.0 / 2 / 1000, 0.0])
    
    # Fixed contact model
    A = np.array([
        [0.8381,  -0.2401, -0.0076],
        [-0.2401,  0.7016,  0.1892],
        [-0.0076,  0.1892,  0.5717],
    ])
    rho = 0.02
    c = np.array([-pt[1] / rho, pt[0] / rho, -1.0])
    b_true = np.linalg.solve(A, c)
    b_true = b_true / np.linalg.norm(b_true)

    # ---------------------------------------------------------
    #       Planner parameter
    # ---------------------------------------------------------
    opts_model = OptsModel()
    opts_model.Q = 20 * np.array([50.0, 50.0, 20.0])
    opts_model.Qf = 20 * 20 * np.array([50.0, 50.0, 20.0])
    opts_model.ep = np.array([0.0005, 0.0005, 0.001])
    opts_model.epf = np.array([0.0005, 0.0005, 0.001])
    opts_model.R = np.array([0.0, 1.0])
    opts_model.T = T
    opts_model.NX = 3
    opts_model.NU = 2
    opts_model.rho = rho
    opts_model.pt = pt
    opts_model.c = c
    opts_model.xinit = x0.copy()
    opts_model.xref = np.tile(xstar, (NH, 1)).T  # (3, NH)

    # Start with c as initial guess for contact model
    set_contact_model_b(opts_model, c / np.linalg.norm(c))

    # True D_inv for simulation (fixed throughout)
    _, D_inv_true = get_contact_model_b(opts_model, b_true)

    # ---------------------------------------------------------
    #       Phase 1: Probing — collect N_initial data points
    # ---------------------------------------------------------
    print("=" * 60)
    print(f"Phase 1: Probing with {N_initial} random actions")
    print("=" * 60)

    learner = PushLearner(N_initial, 0.9, opts_model)

    x_probe = np.zeros((3, N_initial + 1))
    u_probe = np.zeros((2, N_initial))
    x_probe[:, 0] = x0.copy()

    # Alternate pushing direction to excite the system
    u_rand = np.array([u_limit_ang[0], u_limit_mag[1]])
    for i in range(N_initial):
        u_rand = np.array([-u_rand[0], u_rand[1]])
        u_probe[:, i] = u_rand
        x_probe[:, i + 1] = f_(x_probe[:, i], u_probe[:, i], T, D_inv_true)

        # Feed data to learner
        flag = learner.receive_data(x_probe[:, i + 1], x_probe[:, i], opts_model)
        b_est, flag = learner.train_svd(opts_model, flag)

    if flag < 0:
        print("Learning failed — not enough distinct motion. Falling back to c.")
        b_est = c / np.linalg.norm(c)
    else:
        print(f"Learned b: {b_est}")
        print(f"True    b: {b_true}")
        print(f"Error    : {np.linalg.norm(b_est - b_true):.6f}")

    # Set learned contact model
    set_contact_model_b(opts_model, b_est)

    x_start = x_probe[:, N_initial]  # state after probing

    # ---------------------------------------------------------
    #       Phase 2a: Plan with DDP
    # ---------------------------------------------------------
    print("=" * 60)
    print("Phase 2a: Planning trajectory with DDP")
    print("=" * 60)

    planner = PushPlanner(NH, u_limit_ang, u_limit_mag)
    planner.set_initial_pose(x_start, opts_model)
    planner.set_goal_pose(xstar)
    opts_model.xref = np.tile(xstar, (NH, 1)).T

    # Random initial control guess
    planner.unom = np.random.rand(2, NH)
    planner.unom[0, :] = (np.random.rand(NH) - 0.5) * 2 * u_limit_ang[1]
    planner.unom[1, :] = np.random.rand(NH) * u_limit_mag[1]

    t0_ddp = time.perf_counter()
    info = planner.train_controller(None, opts_model)
    t_ddp = time.perf_counter() - t0_ddp
    print(f"DDP converged: exitflag={info.exitflag}, iterations={info.iter}")
    print(f"DDP planning time: {t_ddp:.4f} s")

    # ---------------------------------------------------------
    #       Phase 2b: Plan with Dubins
    # ---------------------------------------------------------
    print("=" * 60)
    print("Phase 2b: Planning trajectory with Dubins")
    print("=" * 60)

    # Extract limit surface parameters for Dubins from identified model
    a_ls, b_ls = estimate_dubins_limit_surface_from_b(b_est, c, rho)
    r_dist = np.linalg.norm(pt)
    print(f"Dubins LS from sys-id: a={a_ls:.4f}, b={b_ls:.4f}")

    dubins_planner = PushPlannerDubin(a_ls, b_ls, r_dist, mu)
    print(f"Dubins turning radius: {dubins_planner.radius_turn:.4f}")

    t0_dubins = time.perf_counter()
    dubins_xnom, dubins_u = dubins_planner.plan(x_start, xstar)
    t_dubins = time.perf_counter() - t0_dubins
    print(f"Dubins path: {dubins_xnom.shape[1]} waypoints")
    print(f"Dubins planning time: {t_dubins:.6f} s")

    print()
    print(f"Planning time comparison: DDP={t_ddp:.4f}s, Dubins={t_dubins:.6f}s, "
          f"ratio={t_ddp / t_dubins:.0f}x")

    # ---------------------------------------------------------
    #       Phase 3: Simulate DDP with feedback control
    # ---------------------------------------------------------
    print("=" * 60)
    print("Phase 3: Simulating DDP with feedback control")
    print("=" * 60)

    N_exec = NH
    x = np.zeros((3, N_exec))
    u = np.zeros((2, N_exec))
    x[:, 0] = x_start.copy()

    # Object size for animation
    W = 35.0 / 1000
    L = 50.0 / 1000
    plot_range = np.array([-101, 230, -380, -218]) / 1000
    black = (0.0, 0.0, 0.0)

    animation = PushAnimation(5, black, pt, W, L, plot_range, x0, xstar,
                              save_video=True, video_path="pushing_demo_simple.mp4", fps=10)

    Nend = N_exec - 1
    for i in range(N_exec - 1):
        u[:, i], _ = planner.control_output_i(x[:, i], i)
        x[:, i + 1] = f_(x[:, i], u[:, i], T, D_inv_true)
        animation.draw_frame(x[:, i + 1], u[:, i], i)

        if np.linalg.norm(x[:, i + 1] - x[:, i]) < 1e-6 and i > 10:
            Nend = i + 1
            print(f"Converged at step {i}")
            break

    animation.finish_video()
    print(f"Video saved to pushing_demo_simple.mp4")

    # Trim trailing stationary part
    for i in range(Nend, 0, -1):
        if np.linalg.norm(x[:, i] - x[:, i - 1]) > 1e-4:
            Nend = i
            break

    # ---------------------------------------------------------
    #       Comparison Plots
    # ---------------------------------------------------------
    xplot = x[:, :Nend]
    uplot = u[:, :Nend]
    dubins_x = dubins_xnom

    # Figure 1: XY trajectory comparison
    fig0, ax0 = plt.subplots(1, 1, num=0, figsize=(8, 6))
    ax0.plot(xplot[0, :], xplot[1, :], "-b", linewidth=2, label="DDP")
    ax0.plot(dubins_x[0, :], dubins_x[1, :], "--m", linewidth=2, label="Dubins")
    ax0.plot(x_start[0], x_start[1], "go", markersize=10, label="start")
    ax0.plot(xstar[0], xstar[1], "r*", markersize=12, label="goal")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_title("Trajectory comparison: DDP vs Dubins")
    ax0.legend()
    ax0.set_aspect("equal")
    ax0.grid(True, alpha=0.3)

    # Figure 2: State trajectories over time
    fig1, axs1 = plt.subplots(3, 1, num=1, figsize=(8, 6))
    # DDP
    axs1[0].plot(xplot[0, :], "-b", label="DDP")
    axs1[0].plot(xstar[0] * np.ones(Nend), "-r", label="goal")
    axs1[0].set_title("x")
    axs1[0].legend()
    axs1[1].plot(xplot[1, :], "-b")
    axs1[1].plot(xstar[1] * np.ones(Nend), "-r")
    axs1[1].set_title("y")
    axs1[2].plot(xplot[2, :], "-b")
    axs1[2].plot(xstar[2] * np.ones(Nend), "-r")
    axs1[2].set_title("theta")
    # Dubins
    n_dub = dubins_x.shape[1]
    axs1[0].plot(dubins_x[0, :], "--m", label="Dubins")
    axs1[0].legend()
    axs1[1].plot(dubins_x[1, :], "--m")
    axs1[2].plot(dubins_x[2, :], "--m")
    fig1.tight_layout()

    # Figure 3: DDP controls
    fig2, axs2 = plt.subplots(2, 1, num=2, figsize=(8, 4))
    axs2[0].plot(uplot[0, :], "-*")
    axs2[0].set_title("DDP Control: angle")
    axs2[1].plot(uplot[1, :], "-*")
    axs2[1].set_title("DDP Control: magnitude")
    fig2.tight_layout()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
