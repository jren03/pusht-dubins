"""Demo controller for closed-loop pushing. Port of wrapper/demo_controller.m."""

import numpy as np
import matplotlib.pyplot as plt

from planar_pushing_tools.config import OptsModel, set_contact_model_b, get_contact_model_b
from planar_pushing_tools.model import f_
from planar_pushing_tools.push_planner import PushPlanner
from planar_pushing_tools.push_learner import PushLearner
from planar_pushing_tools.push_decision import PushDecision
from planar_pushing_tools.push_animation import PushAnimation


def main():
    # ---------------------------------------------------------
    #       Simulation settings
    # ---------------------------------------------------------
    N = 200    # length of simulation
    NH = 200   # length of horizon
    T = 0.1    # timestep
    
    # ---------------------------------------------------------
    #       Task setup
    # ---------------------------------------------------------
    x0 = np.array([-21.4 / 1000, -300. / 1000, 0]) # initial pose (x, y, theta)
    xstar = np.array([166.4 / 1000, -290.0 / 1000, 0.5]) # goal pose (x, y, theta)

    # ---------------------------------------------------------
    #       Model parameter
    # ---------------------------------------------------------
    # pushing action limits for DDP planner
    u_limit_ang = np.array([-40, 40]) * np.pi / 180 # pushing angle limit
    u_limit_mag = np.array([0, 0.08])

    # point of contact in the local frame
    pt = np.array([-35.0 / 2 / 1000, 0.0])

    # Varying contact model between two extremes
    A1 = np.array([
        [0.8381,  -0.0401, -0.0076],
        [-0.0401,  0.7016,  0.0892],
        [-0.0076,  0.0892,  0.5717],
    ])

    A2 = np.array([
        [1.0123,  0.1024, 0.0921],
        [0.1024,  1.0030, 0.076],
        [0.0921,  0.076, 1.4629],
    ])

    KA = np.arange(N) / (N - 1)  # blending factor from A1 to A2

    rho = 0.04
    c = np.array([-pt[1] / rho, pt[0] / rho, -1.0])

    A = A1 * KA[0] + A2 * (1 - KA[0])
    b = np.linalg.solve(A, c)
    b = b / np.linalg.norm(b)



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

    # ---------------------------------------------------------
    #       Learner parameter
    # ---------------------------------------------------------
    online_identification = True
    data_NH = 4
    data_discount = 0.9

    if online_identification:
        solver_N = NH
        besti = c.copy()
    else:
        solver_N = N
        besti = b.copy()

    opts_model.xinit = x0.copy()
    opts_model.xref = np.tile(xstar, (NH, 1)).T  # (3, NH)

    # Set initial contact model
    set_contact_model_b(opts_model, besti)

    # Create components
    pushplanner = PushPlanner(solver_N, u_limit_ang, u_limit_mag)
    pushlearner = PushLearner(data_NH, data_discount, opts_model)

    pushplanner.set_initial_pose(x0, opts_model)
    pushplanner.set_goal_pose(xstar)

    # Initialize random control sequence
    pushplanner.unom = np.random.rand(2, NH)
    pushplanner.unom[0, :] = (np.random.rand(NH) - 0.5) * 2 * u_limit_ang[1]
    pushplanner.unom[1, :] = np.random.rand(NH) * u_limit_mag[1]

    # ---------------------------------------------------------
    #       Simulation setup
    # ---------------------------------------------------------
    xinit = x0 + 0 * (np.random.rand(3) - 0.5)

    # Disturbance
    dist_n = 15
    dist_mag = 0 * np.array([-0.1, 0, 0.06])

    # Noise level
    noise = 0 * np.array([0.2, 0.2, 0.2])

    # Object size
    W = 35.0 / 1000
    L = 50.0 / 1000
    plot_range = np.array([-101, 230, -380, -218]) / 1000

    black = (0.0, 0.0, 0.0)
    animation = PushAnimation(5, black, pt, W, L, plot_range, x0, xstar,
                              save_video=True, video_path="pushing_demo.mp4", fps=10)

    # ---------------------------------------------------------
    #       Begin simulation
    # ---------------------------------------------------------
    decision = PushDecision(pushplanner, pushlearner, xinit, data_NH, NH)

    x = np.zeros((3, N))
    u = np.zeros((2, N))
    x[:, 0] = xinit

    record_b = np.tile(c, (N, 1)).T        # (3, N)
    record_besti = np.tile(c, (N, 1)).T     # (3, N)
    id_learning = []
    BetaNorm = np.zeros(N - 1)

    Nend = N - 1
    for i in range(N - 1):
        # Decision: controller/planner/learner
        u[:, i], info = decision.decide(x[:, i], opts_model)

        # Record
        if isinstance(info.besti, np.ndarray):
            record_besti[:, i] = info.besti
        if info.learned:
            id_learning.append(i)
        BetaNorm[i] = info.bnorm

        # Simulate with varying contact model
        A = A1 * KA[i] + A2 * (1 - KA[i])
        b_true = np.linalg.solve(A, c)
        b_true = b_true / np.linalg.norm(b_true)
        record_b[:, i] = b_true

        print(f"b_model - b_true: {opts_model.b - b_true}")
        _, D_inv = get_contact_model_b(opts_model, b_true)
        x[:, i + 1] = f_(x[:, i], u[:, i], opts_model.T, D_inv)

        # Add noise
        dx = x[:, i + 1] - x[:, i]
        x[:, i + 1] = x[:, i + 1] + 2 * (np.random.rand() - 0.5) * dx * noise

        # Print
        print(f"Ts {i} State: {decision.state}")

        if info.exitflag < 0:
            Nend = i
            print("Solver failed at:")
            print(i)
            break

        # Drawing
        if i == dist_n:
            x[:, i + 1] = x[:, i + 1] + dist_mag
            animation.draw_frame(x[:, i + 1], u[:, i], i, dist_mag)
        else:
            animation.draw_frame(x[:, i + 1], u[:, i], i)

    # Save video
    animation.finish_video()

    # Trim trailing stationary part
    for i in range(Nend, 0, -1):
        dist = np.linalg.norm(x[:, i] - x[:, i - 1])
        if dist > 1e-4:
            Nend = i
            break

    # ---------------------------------------------------------
    #       Plot contact model evolution
    # ---------------------------------------------------------
    fig3, axs3 = plt.subplots(3, 1, num=3)
    labels = ["b1", "b2", "b3"]
    for j in range(3):
        axs3[j].plot(record_b[j, :Nend], "r.-", label="true b")
        axs3[j].plot(record_besti[j, :Nend], "b-", label="estimated b")
        if id_learning:
            axs3[j].plot(id_learning, record_besti[j, id_learning], "b.", markersize=15)
        axs3[j].set_ylabel(labels[j])
        axs3[j].legend()
    axs3[0].set_title("Contact model evolution")

    # ---------------------------------------------------------
    #       Plot state trajectories
    # ---------------------------------------------------------
    xnom = x[:, :Nend]
    unom = u[:, :Nend]

    fig1, axs1 = plt.subplots(3, 1, num=1)
    axs1[0].plot(xnom[0, :], "-b")
    axs1[0].plot(xstar[0] * np.ones(Nend), "-r")
    axs1[0].set_title("x")
    axs1[1].plot(xnom[1, :], "-b")
    axs1[1].plot(xstar[1] * np.ones(Nend), "-r")
    axs1[1].set_title("y")
    axs1[2].plot(xnom[2, :], "-b")
    axs1[2].plot(xstar[2] * np.ones(Nend), "-r")
    axs1[2].set_title("theta")

    # ---------------------------------------------------------
    #       Plot controls
    # ---------------------------------------------------------
    fig2, axs2 = plt.subplots(4, 1, num=2)
    axs2[0].plot(unom[0, :], "-*")
    axs2[0].set_title("k1 (angle)")
    axs2[1].plot(unom[1, :], "-*")
    axs2[1].set_title("k2 (magnitude)")
    axs2[2].set_title("(unused)")
    axs2[3].plot(BetaNorm[:Nend], "-*")
    axs2[3].set_title("Feedback Gain")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
