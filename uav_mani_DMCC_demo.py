#!/usr/bin/env python
# coding=UTF-8
"""
Author: Wei Luo
Date: 2021-10-06 18:20:46
LastEditors: Wei Luo
LastEditTime: 2024-12-29 10:32:04
Note: this is a demo file that shows how to implement a DMCC trajectory planner for a UAV without a manipulator
"""
import numpy as np
import time
import casadi as ca
import argparse

from dmcc_model.uav_manipulator_DMCC_3D_discrete_pos_force import (
    DMCC_UAV_Manipulator,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="DMCC Demo", description="UAV DMCC trajectory planner"
    )
    parser.add_argument(
        "--task", type=str, choices=["drone_racing", "fixed_point_contact"], default="drone_racing"
    )
    args = parser.parse_args()
    task = args.task  
    print("mission: {}".format(task))

    if task == "drone_racing":
        start_point = [0.0, 0.0, 0.55]  # x,y,z [m]
        end_point = [2.5, 0, 0.55]  # x,y,z [m]
        waypoints = np.array(
            [
                [
                    start_point[0],
                    start_point[1],
                    start_point[2],
                    0.0,
                    0.0,
                    0.0,
                    np.pi / 2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.55,
                    0.2,
                    0.4,
                    0.0,
                    0.0,
                    0.0,
                    np.pi / 2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.95,
                    -0.1,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    np.pi / 2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    end_point[0],
                    end_point[1],
                    end_point[2],
                    0.0,
                    0.0,
                    0.0,
                    np.pi / 2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )

        g = 9.7
        mass_quad = 1.659
        obj = DMCC_UAV_Manipulator(
            mass_quadrotor=mass_quad,
            has_contact_target=False,
            has_manipulator=True,
            manipulator_length=0.21,
            montage_offset_b=[0, 0, -0.1],
            inertia_moment=[0.01558, 0.01523, 0.01926],
            manipulator_inertia_moment=[2.054e-5, 0.0014302, 0.0014264],
            mass_manipulator=0.136,
            g=g,
        )

        obj.get_path_waypoints(
            waypoints[0],
            waypoints[1:],
            avg_vel=0.5,
            waypoint_distance=0.02,
            dt_waypoint=0.1,
        )
        obj.initialization_formulation(kappa0=2)

        # setup optimization conditions
        lbg = []
        ubg = []
        # equality constraints
        for _ in range(obj.equal_constraint_length):
            lbg.append(0.0)
            ubg.append(0.0)

        # velocity constraints vx, vy, vz, d_roll, d_pitch, d_yaw
        for _ in range(obj.num_waypoints - 1):
            lbg.append(-0.25)
            lbg.append(-0.25)
            # lbg.append(0.0)
            lbg.append(-0.15)
            lbg.append(-8)
            lbg.append(-8)
            lbg.append(-2)
            lbg.append(-np.pi / 8)
            ubg.append(0.25)
            ubg.append(0.25)
            # ubg.append(0.3**2)
            ubg.append(0.15)
            ubg.append(8)
            ubg.append(8)
            ubg.append(2)
            ubg.append(np.pi / 8)

        # setup state variable
        lbx = []
        ubx = []

        # Tn
        lbx = [0.01]
        ubx = [ca.inf]

        # U
        U1_min = 0.5 * mass_quad * g / 4
        U1_max = 1.5 * mass_quad * g / 4
        U2_min = 0.5 * mass_quad * g / 4
        U2_max = 1.5 * mass_quad * g / 4
        U3_min = 0.5 * mass_quad * g / 4
        U3_max = 1.5 * mass_quad * g / 4
        U4_min = 0.5 * mass_quad * g / 4
        U4_max = 1.5 * mass_quad * g / 4
        U5_min = -1.0
        U5_max = 1.0
        for _ in range(obj.num_waypoints):
            lbx.append(U1_min)
            lbx.append(U2_min)
            lbx.append(U3_min)
            lbx.append(U4_min)
            lbx.append(U5_min)
            ubx.append(U1_max)
            ubx.append(U2_max)
            ubx.append(U3_max)
            ubx.append(U4_max)
            ubx.append(U5_max)

        # X
        for _ in range(obj.num_waypoints):
            lbx = lbx + [-np.inf, -np.inf, 0.0]  # x/y/z
            lbx = lbx + [
                -np.pi / 4.0,
                -np.pi / 4.0,
                -np.pi,
                np.pi / 3.0,
            ]  # phi/theta/psi/alpha

            ubx = ubx + [np.inf, np.inf, 1.5]  # x/y/z
            ubx = ubx + [
                np.pi / 4.0,
                np.pi / 4.0,
                np.pi,
                np.pi / 3.0 * 2.0,
            ]  # phi/theta/psi/alpha

        # lambda
        for _ in range(obj.num_waypoints):
            for _ in range(obj.num_def_point):
                lbx.append(0)
                ubx.append(1)

        for _ in range(obj.num_waypoints - 1):
            for _ in range(obj.num_def_point):
                lbx.append(0)
                ubx.append(0.05)

        c_p = ca.vcat([obj.init_pose.flatten().tolist(), obj.Wp_ref.flatten().tolist()])
        t_ = time.time()
        obj.get_results(
            init_state=None, parameters=c_p, lb_x=lbx, ub_x=ubx, lb_g=lbg, ub_g=ubg
        )
        print("Solving requries {0}".format(time.time() - t_))
        print("dt is {0}".format(obj.result_tn / (obj.num_waypoints - 1)))

        obj.showPath_XYplane()
        obj.showPath_XZplane()
        obj.showPath_3D()

    elif task == "fixed_point_contact":

        def fixed_ref_fct(x_o, w, R, direction=1):
            dt = ca.SX.sym("dt")
            i = ca.SX.sym("i")
            result_pos = [x_o[0], x_o[1], x_o[2]]

            result_vel = [0.0, 0.0, 0.0]
            return ca.Function(
                "circle_traj", [dt, i], [ca.vcat(result_pos)]
            ), ca.Function("circle_vel_traj", [dt, i], [ca.vcat(result_vel)])

        start_point = [0.0, 0.0, 0.65]  # x,y,z [m]
        end_point = [2.5, 0, 0.65]  # x,y,z [m]
        waypoints = np.array(
            [
                [
                    start_point[0],
                    start_point[1],
                    start_point[2],
                    0.0,
                    0.0,
                    0.0,
                    np.pi / 2.0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1,
                    0.0,
                    0.54,
                    0.0,
                    0.0,
                    0.0,
                    np.pi / 2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    end_point[0],
                    end_point[1],
                    end_point[2],
                    0.0,
                    0.0,
                    0.0,
                    np.pi / 2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )
        circle_middle_point = [1, 0.0, 0.4]  # x,y,z [m]
        mobile_robot_trajectory, mobile_robot_vel_trajectory = fixed_ref_fct(
            x_o=circle_middle_point, w=0.2, R=0.25, direction=1
        )

        obj = DMCC_UAV_Manipulator(
            has_contact_target=True,
            has_manipulator=True,
            trajectory_function=mobile_robot_trajectory,
            vel_trajectory_function=mobile_robot_vel_trajectory,
            manipulator_length=0.1,
            montage_offset_b=[0, 0, -0.05],
        )

        g = 9.8066
        mass_quad = 1.659

        obj.get_path_waypoints(
            waypoints[0],
            waypoints[1:],
            avg_vel=1.5,
            waypoint_distance=0.02,
            dt_waypoint=0.06,
        )
        obj.initialization_formulation(kappa0=2)

        # setup optimization conditions
        lbg = []
        ubg = []
        # equality constraints
        for _ in range(obj.equal_constraint_length):
            lbg.append(0.0)
            ubg.append(0.0)

        # velocity constraints vx, vy, vz, d_roll, d_pitch, d_yaw
        for _ in range(obj.num_waypoints - 1):
            lbg.append(-1.3)
            lbg.append(-1.3)
            lbg.append(-1.15)
            lbg.append(-8)
            lbg.append(-8)
            lbg.append(-2)
            lbg.append(-np.pi / 2)
            ubg.append(1.3)
            ubg.append(1.3)
            ubg.append(1.15)
            ubg.append(8)
            ubg.append(8)
            ubg.append(2)
            ubg.append(np.pi / 2)
        for i in range(obj.num_waypoints - 1):
            lbg.append(0.0)
            lbg.append(0.0)
            ubg.append(0.01**2)
            ubg.append(0.01**2)

        # Z_Endposition_manipulator >= -elastic formulation
        for _ in range(obj.num_waypoints - 1):
            lbg.append(-0.01)
            ubg.append(ca.inf)

        # setup state variable
        lbx = []
        ubx = []

        # Tn
        lbx = [0.01]
        ubx = [ca.inf]

        # U
        U1_min = 0.5 * mass_quad * g / 4
        U1_max = 1.5 * mass_quad * g / 4
        U2_min = 0.5 * mass_quad * g / 4
        U2_max = 1.5 * mass_quad * g / 4
        U3_min = 0.5 * mass_quad * g / 4
        U3_max = 1.5 * mass_quad * g / 4
        U4_min = 0.5 * mass_quad * g / 4
        U4_max = 1.5 * mass_quad * g / 4
        U5_min = -2.0
        U5_max = 2.0
        for _ in range(obj.num_waypoints):
            lbx.append(U1_min)
            lbx.append(U2_min)
            lbx.append(U3_min)
            lbx.append(U4_min)
            lbx.append(U5_min)
            ubx.append(U1_max)
            ubx.append(U2_max)
            ubx.append(U3_max)
            ubx.append(U4_max)
            ubx.append(U5_max)

        # X
        for _ in range(obj.num_waypoints):
            lbx = lbx + [-np.inf, -np.inf, 0.0]  # x/y/z
            lbx = lbx + [
                -np.pi / 4.0,
                -np.pi / 4.0,
                -np.pi,
                np.pi / 3.0,
            ]  # phi/theta/psi/alpha

            ubx = ubx + [np.inf, np.inf, 1.5]  # x/y/z
            ubx = ubx + [
                np.pi / 4.0,
                np.pi / 4.0,
                np.pi,
                np.pi / 3.0 * 2.0,
            ]  # phi/theta/psi/alpha

        # epsilon_param
        for _ in range(obj.num_waypoints - 1):
            lbx.append(0)
            ubx.append(1)

        # kappa_param
        for _ in range(obj.num_waypoints):
            lbx.append(0)
            ubx.append(obj.kappa_0)

        # l_param
        for _ in range(obj.num_waypoints - 1):
            lbx.append(0.0)
            ubx.append(0.02)

        c_p = ca.vcat([obj.init_pose.flatten().tolist(), obj.Wp_ref.flatten().tolist()])
        t_ = time.time()
        obj.get_results(
            init_state=None, parameters=c_p, lb_x=lbx, ub_x=ubx, lb_g=lbg, ub_g=ubg
        )
        print("Solving requries {0}".format(time.time() - t_))
        print("dt is {0}".format(obj.result_tn / (obj.num_waypoints - 1)))
        # obj.showPath_XYplane()
        # obj.showPath_XZplane()
        obj.showPath_3D()

        obj.save_final_result_as_npy(
            file_name="./exp_results_npy/uav_manipulator_fixedpoint.npy"
        )
