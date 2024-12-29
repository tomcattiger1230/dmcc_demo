#!/usr/bin/env python
# coding=UTF-8
"""
Author: Wei Luo
Date: 2021-10-18 13:42:11
LastEditors: Wei Luo
LastEditTime: 2024-12-29 10:47:22
Note: This file uses four forces to control the quadrotors, considering montage
offset of the manipulator
"""

import numpy as np
import casadi as ca
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib


class DMCC_base(object):
    def __init__(
        self,
        manipulator_length=0.34,
        mass_quadrotor=1.659,
        mass_manipulator=0.0,
        inertia_moment=[0.0348, 0.0459, 0.0977],
        manipulator_inertia_moment=[0, 0.0019, 0],
        has_contact_target=False,
        has_manipulator=False,
        montage_offset_b=[0.0, 0.0, 0.0],
        g=9.8066,
    ):
        self.num_waypoints = 0
        self.g_ = g
        if manipulator_length > 0.05:
            self.dis_arm = manipulator_length / 2.0
        else:
            self.dis_arm = 0.0
        self.arm_length = manipulator_length
        self.mass_quadrotor = mass_quadrotor
        self.mass_manipulator = mass_manipulator
        self.total_mass = mass_quadrotor + mass_manipulator
        self.Ixx, self.Iyy, self.Izz = inertia_moment
        self.I_manipulator = manipulator_inertia_moment
        self.num_def_point = None
        self.has_contact_target = has_contact_target
        self.has_manipulator = has_manipulator
        self.montage_offset_b = montage_offset_b
        # results
        self.X_state = None
        self.U_state = None

    # math functions
    @staticmethod
    def rotation_matrix(angle):
        phi = angle[0]
        theta = angle[1]
        psi = angle[2]
        Rz = np.array(
            [[ca.cos(psi), -ca.sin(psi), 0], [ca.sin(psi), ca.cos(psi), 0], [0, 0, 1]]
        )
        Ry = np.array(
            [
                [ca.cos(theta), 0, ca.sin(theta)],
                [0, 1, 0],
                [-ca.sin(theta), 0, ca.cos(theta)],
            ]
        )
        Rx = np.array(
            [[1, 0, 0], [0, ca.cos(phi), -ca.sin(phi)], [0, ca.sin(phi), ca.cos(phi)]]
        )
        eRb = ca.mtimes([Rz, Ry, Rx])
        return eRb

    def discrete_lagrange(self, dt, q_n, q_np1, fct_L, num_state=6):
        """
        second-order accurate diescrete Lagrangian
        Args:
            dt: diescrete time step
            q_n: state at n step
            q_np1: state at n+1 step
            fct_L: lagrange function
        Returns:
            symbolic
        """
        q = (q_n + q_np1) / 2.0
        # average of euler angles in Rad, [x, y, z, r, p, y]
        q[3:6] = self.average_rpy(q_n[3:6], q_np1[3:6])

        q_dot = (q_np1 - q_n) / dt
        q_dot[3:6] = self.difference_rpy(q_n[3:6], q_np1[3:6], dt)
        if num_state > 6:
            for i in range(6, num_state):
                s_ = ca.sin(q_n[i]) + ca.sin(q_np1[i])
                c_ = ca.cos(q_n[i]) + ca.cos(q_np1[i])
                q[i] = ca.atan2(s_, c_)
                diff_ = q_np1[i] - q_n[i]
                q_dot[i] = ca.atan2(ca.sin(diff_), ca.cos(diff_)) / dt
        L_d = dt * fct_L(q, q_dot)
        return L_d

    def discrete_lagrange_verlet(self, dt, q_n, q_np1, fct_L, num_state=6):
        """
        second-order accurate diescrete Lagrangian
        Args:
            dt: diescrete time step
            q_n: state at n step
            q_np1: state at n+1 step
            fct_L: lagrange function
        Returns:
            symbolic
        """
        q_dot = (q_np1 - q_n) / dt
        q_dot[3:6] = self.difference_rpy(q_n[3:6], q_np1[3:6], dt)
        if num_state > 6:
            for i in range(6, num_state):
                diff_ = q_np1[i] - q_n[i]
                q_dot[i] = ca.atan2(ca.sin(diff_), ca.cos(diff_)) / dt
        L_d = 0.5 * dt * fct_L(q_n, q_dot) + 0.5 * dt * fct_L(q_np1, q_dot)
        return L_d

    def discrete_forces(self, dt, f, q_n, q_np1, u_n, num_state=6):
        """
        diescrete forces estimator

        """
        assert q_n.shape == (num_state, 1)
        q = (q_n + q_np1) / 2.0
        # average of euler angles
        q[3:6] = self.average_rpy(q_n[3:6], q_np1[3:6])
        if num_state > 6:
            for i in range(6, num_state):
                s_ = ca.sin(q_n[i]) + ca.sin(q_np1[i])
                c_ = ca.cos(q_n[i]) + ca.cos(q_np1[i])
                q[i] = ca.atan2(s_, c_)

        f_d = 1 / 2.0 * dt * f(q, u_n)
        return f_d

    @staticmethod
    def discrete_forces_v2(dt, f, q, u_n, u_np1):
        f_d = 1 / 4.0 * dt * (f(q, u_n) + f(q, u_np1))
        return f_d

    @staticmethod
    def average_rpy(state_1, state_2):
        # only average the roll, pitch and yaw
        assert state_1.shape == (3, 1) and state_2.shape == (
            3,
            1,
        ), "input size should be (3, 1)"
        state = state_1 + state_2
        for i in range(3):
            s_ = ca.sin(state_1[i]) + ca.sin(state_2[i])
            c_ = ca.cos(state_1[i]) + ca.cos(state_2[i])
            state[i] = ca.atan2(s_, c_)
        return state

    # @staticmethod
    # def difference_rpy(rpy_1, rpy_2, dt):
    #     # only calculate difference between two SO(3) rotation
    #     assert rpy_1.shape == (3, 1) and rpy_2.shape == (
    #         3, 1), 'input size should be (3, 1)'
    #     rpy_dot = rpy_2 - rpy_1
    #     for i in range(3):
    #         diff_ = rpy_2[i] - rpy_1[i]
    #         rpy_dot[i] = ca.atan2(ca.sin(diff_), ca.cos(diff_)) / dt
    #     return rpy_dot

    @staticmethod
    def difference_rpy(rpy_1, rpy_2, dt):
        # only calculate difference between two SO(3) rotation
        assert rpy_1.shape == (3, 1) and rpy_2.shape == (
            3,
            1,
        ), "input size should be (3, 1)"

        def rotation_matrix(angle):
            phi = angle[0]
            theta = angle[1]
            psi = angle[2]
            Rz = np.array(
                [
                    [ca.cos(psi), -ca.sin(psi), 0],
                    [ca.sin(psi), ca.cos(psi), 0],
                    [0, 0, 1],
                ]
            )
            Ry = np.array(
                [
                    [ca.cos(theta), 0, ca.sin(theta)],
                    [0, 1, 0],
                    [-ca.sin(theta), 0, ca.cos(theta)],
                ]
            )
            Rx = np.array(
                [
                    [1, 0, 0],
                    [0, ca.cos(phi), -ca.sin(phi)],
                    [0, ca.sin(phi), ca.cos(phi)],
                ]
            )
            return ca.mtimes([Rz, Ry, Rx])

        r_1 = rotation_matrix(rpy_1)
        r_2 = rotation_matrix(rpy_2)
        # return ca.trace(np.diag([1, 1, 1]) - ca.mtimes([r_1.T, r_2])) / dt
        # return ca.inv_skew(ca.mtimes([r_1.T, (r_2 - r_1)])) / dt
        return ca.inv_skew(ca.mtimes([r_1.T, r_2]) - np.diag([1, 1, 1])) / dt
        # skew_ = ca.mtimes([r_1.T, r_2]) - np.diag([1, 1, 1])
        # result = np.zeros((3, 1))
        # result[0], result[1], result[2] = skew_[2, 1], skew_[0, 2], skew_[1, 0]
        # return result / dt

    # @staticmethod
    # def quaternion_to_rpy(quaternion):
    #     q0, q1, q2, q3 = quaternion[0], quaternion[1], quaternion[
    #         2], quaternion[3]
    #     roll_ = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    #     pitch_ = np.arcsin(2 * (q0 * q2 - q3 * q1))
    #     yaw_ = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    #     return np.array([roll_, pitch_, yaw_])

    # @staticmethod
    # def quaternion_exponential_t(q, t):
    #     theta_ = ca.arccos(q[0])
    #     u_ = q[1:] / ca.sin(theta_)
    #     new_u_ = u_ * ca.sin(t * theta_)
    #     return np.array([ca.cos(t * theta_), new_u_[0], new_u_[1], new_u_[2]])

    # @staticmethod
    # def quaternion_product(q_0, q_1):
    #     return np.array([
    #         q_0[0] * q_1[0] - q_0[1] * q_1[1] - q_0[2] * q_1[2] -
    #         q_0[3] * q_1[3], q_0[0] * q_1[1] + q_0[1] * q_1[0] +
    #         q_0[2] * q_1[3] - q_0[3] * q_1[2], q_0[0] * q_1[2] -
    #         q_0[1] * q_1[3] + q_0[2] * q_1[0] + q_0[3] * q_1[1],
    #         q_0[0] * q_1[3] + q_0[1] * q_1[2] - q_0[2] * q_1[1] +
    #         q_0[3] * q_1[0]
    #     ])

    # @staticmethod
    # def quaternion_inverse(q):
    #     return np.array([q[0], -q[1], -q[2], -q[3]])

    # def quaternion_average(self, q_0, q_1):
    #     # based on slerp (spherical linear interpolation)
    #     return self.quaternion_product(
    #         q_0,
    #         self.quaternion_exponential_t(
    #             self.quaternion_product(self.quaternion_inverse(q_0), q_1),
    #             0.5))

    # @staticmethod
    # def rpy_to_quaternion(angle):
    #     r = angle[0]
    #     p = angle[1]
    #     y = angle[2]
    #     cy = np.cos(y * 0.5)
    #     sy = np.sin(y * 0.5)
    #     cp = np.cos(p * 0.5)
    #     sp = np.sin(p * 0.5)
    #     cr = np.cos(r * 0.5)
    #     sr = np.sin(r * 0.5)

    #     qw = cr * cp * cy + sr * sp * sy
    #     qx = sr * cp * cy - cr * sp * sy
    #     qy = cr * sp * cy + sr * cp * sy
    #     qz = cr * cp * sy - sr * sp * cy
    #     return np.array([qw, qx, qy, qz])

    # def average_rpy(self, rpy_1, rpy_2):
    #     # only average the roll, pitch and yaw
    #     assert rpy_1.shape == (3, 1) and rpy_2.shape == (
    #         3, 1), 'input size should be (3, 1)'

    #     quat_n = self.rpy_to_quaternion(rpy_1)
    #     quat_np1 = self.rpy_to_quaternion(rpy_2)

    #     return self.quaternion_to_rpy(self.quaternion_average(
    #         quat_n, quat_np1))

    def average_full_state(self, state_1, state_2, dt, num_state=6):
        # average whole state
        assert state_1.shape == (num_state * 2, 1) and state_2.shape == (
            num_state * 2,
            1,
        ), state_1.shape
        q = (state_1 + state_2) / 2
        # average of euler angles
        q[3:6] = self.average_rpy(state_1[3:6], state_2[3:6])
        if num_state > 6:
            for i in range(6, num_state):
                s_ = ca.sin(state_1[i]) + ca.sin(state_2[i])
                c_ = ca.cos(state_1[i]) + ca.cos(state_2[i])
                q[i] = ca.atan2(s_, c_)

        # velocity (q_2 - q_1)/dt
        d_q = state_2 - state_1

        for i in range(0, 3):
            q[i + num_state] = d_q[i - 3] / dt

        q[num_state + 3 : num_state + 6] = self.difference_rpy(
            state_1[3:6], state_2[3:6], dt
        )

        if num_state == 7:
            q[-1] = (
                ca.atan2(ca.sin(d_q[num_state - 1]), ca.cos(d_q[num_state - 1])) / dt
            )
        return q

    def average_state(self, state_1, state_2, num_state=6):
        # average whole state
        assert state_1.shape == (num_state, 1) and state_2.shape == (
            num_state,
            1,
        ), state_1.shape
        q = (state_1 + state_2) / 2
        if num_state > 3:
            # mean of euler angles
            q[3:6] = self.average_rpy(state_1[3:6], state_2[3:6])
            if num_state > 6:
                for i in range(6, num_state):
                    s_ = ca.sin(state_1[i]) + ca.sin(state_2[i])
                    c_ = ca.cos(state_1[i]) + ca.cos(state_2[i])
                    q[i] = ca.atan2(s_, c_)
        return q

    def average_velocity(self, state_1, state_2, dt, num_state=6):
        # velocity (q_2 - q_1)/dt
        assert state_1.shape == (num_state, 1) and state_2.shape == (
            num_state,
            1,
        ), state_1.shape
        d_q = state_2 - state_1

        if num_state > 3:
            q_dot = deepcopy(state_1)
            for i in range(3):
                q_dot[i] = d_q[i] / dt
            q_dot[3:6] = self.difference_rpy(state_1[3:6], state_2[3:6], dt)
            if num_state == 7:
                q_dot[-1] = ca.atan2(ca.sin(d_q[-1]), ca.cos(d_q[-1])) / dt

        return q_dot

    def get_path_waypoints(
        self,
        current_pose,
        waypoints,
        avg_vel=None,
        waypoint_distance=0.5,
        dt_waypoint=None,
        i_switch=None,
        num_waypoints=None,
    ):
        self.init_pose = current_pose
        print("the current position of the UAV:\n", self.init_pose)
        self.Wp_ref = waypoints
        print("the points that the UAV should be passed:\n", self.Wp_ref)
        self.num_def_point = waypoints.shape[0]
        print("num of way points: %d" % self.num_def_point)
        dist = [
            np.linalg.norm(self.Wp_ref[0, :3].flatten() - self.init_pose[:3].flatten()),
        ]
        if dist[0] < 0.01:
            dist[0] += 0.01  # incase derivative with zero
        for i in range(self.num_def_point - 1):
            dist += [
                dist[i]
                + np.linalg.norm(
                    self.Wp_ref[i + 1, :2].flatten() - self.Wp_ref[i, :2].flatten()
                )
            ]
        if avg_vel is not None:
            self.initial_time_speed_guess(dist[-1], avg_vel=avg_vel)

        if num_waypoints is None:
            if dt_waypoint is not None:
                self.num_waypoints = int(self.time_guess / dt_waypoint)
            else:
                self.num_waypoints = int(np.ceil(dist[-1] / waypoint_distance))
        else:
            self.num_waypoints = num_waypoints

        if i_switch is None:
            self.i_switch = np.array(
                self.num_waypoints * np.array(dist) / dist[-1], dtype=int
            )
        else:
            self.i_switch = i_switch
        if self.num_waypoints <= self.i_switch[-1]:
            self.num_waypoints += 1
        print("dist is {}".format(dist))
        print("i_switch is {}".format(self.i_switch))
        print("we have {0} waypoints".format(self.num_waypoints))

    def initial_time_speed_guess(self, total_dist_guess, t_guess=None, avg_vel=None):
        if t_guess is not None:
            self.time_guess = t_guess
            self.vel_guess = total_dist_guess / self.time_guess
        elif avg_vel is not None:
            self.vel_guess = avg_vel
            self.time_guess = total_dist_guess / self.vel_guess
        else:
            self.vel_guess = 0.5
            self.time_guess = total_dist_guess / self.vel_guess

    def init_state_guess(self, with_additional_relaxation=True, num_state=6):
        """
        Guess states for whole prediction
        """
        x0 = None
        i_wp = 0
        self.with_additional_relaxation = with_additional_relaxation

        if num_state == 6:
            u0 = np.tile(
                np.array([self.total_mass / 4 * self.g_] * 4), self.num_waypoints
            )
        elif num_state == 7:
            u0 = np.tile(
                np.array([self.total_mass / 4 * self.g_] * 4 + [0.0]),
                self.num_waypoints,
            )

        if self.has_contact_target:
            kappa0 = self.kappa_0 * np.ones(self.num_waypoints)
            epsilon_param0 = np.zeros(self.num_waypoints - 1)
            for i in range(self.num_waypoints):
                if i > self.i_switch[i_wp]:
                    i_wp += 1
                if i_wp == 0:
                    wp_last = self.init_pose[:3].flatten()
                else:
                    wp_last = self.Wp_ref[i_wp - 1, :3].flatten()
                wp_next = self.Wp_ref[i_wp, :3].flatten()
                if i_wp > 0:
                    interp = (i - self.i_switch[i_wp - 1]) / (
                        self.i_switch[i_wp] - self.i_switch[i_wp - 1]
                    )
                else:
                    interp = i / (self.i_switch[0] + 1e-6)

                pos_guess = (1 - interp) * wp_last + interp * wp_next
                vel_guess = (
                    (
                        self.vel_guess
                        * (wp_next - wp_last)
                        / ca.norm_2(wp_next - wp_last)
                    )
                    .full()
                    .flatten()
                )
                if x0 is None:
                    x0 = self.init_pose
                else:
                    if num_state == 7:
                        print(
                            np.array(
                                [
                                    pos_guess[0],
                                    pos_guess[1],
                                    pos_guess[2],
                                    0,
                                    0,
                                    0,
                                    np.pi / 2.0,
                                    vel_guess[0],
                                    vel_guess[1],
                                    vel_guess[2],
                                    0,
                                    0,
                                    0,
                                    0,
                                ]
                            )
                        )
                        x0 = np.append(
                            x0.flatten(),
                            np.array(
                                [
                                    pos_guess[0],
                                    pos_guess[1],
                                    pos_guess[2],
                                    0,
                                    0,
                                    0,
                                    np.pi / 2.0,
                                    vel_guess[0],
                                    vel_guess[1],
                                    vel_guess[2],
                                    0,
                                    0,
                                    0,
                                    0,
                                ]
                            ).flatten(),
                        )
                    elif num_state == 6:
                        x0 = np.append(
                            x0,
                            np.array(
                                [
                                    pos_guess[0],
                                    pos_guess[1],
                                    pos_guess[2],
                                    0,
                                    0,
                                    0,
                                    vel_guess[0],
                                    vel_guess[1],
                                    vel_guess[2],
                                    0,
                                    0,
                                    0,
                                ]
                            ),
                        )
                if i == self.i_switch[0]:
                    kappa0[i] = self.kappa_0 - 1
                    epsilon_param0[i] = 1
                    j = i
                    while kappa0[j] != 0:
                        kappa0[j + 1] = kappa0[j] - 1
                        j = j + 1
                        epsilon_param0[j] = 1
                    kappa0[i + self.kappa_0 - 1 :] = 0
            x0 = x0.reshape(-1, self.Wp_ref.shape[1])
            self.x_guess = x0
            print("kappa_0: {}".format(kappa0))
            print("epsilon_param0: {}".format(epsilon_param0))
            l0 = np.zeros(self.num_waypoints - 1)
            gamma0 = np.zeros(self.num_waypoints - 1)
            vel_0 = np.zeros(self.num_waypoints - 1)
            if with_additional_relaxation:
                self.guess_state = ca.vcat(
                    [
                        self.time_guess,
                        u0.reshape(-1, 1),
                        x0[:, :num_state].reshape(-1, 1),
                        epsilon_param0.reshape(-1, 1),
                        kappa0.reshape(-1, 1),
                        l0.flatten(),
                        vel_0.reshape(-1, 1),
                        gamma0.reshape(-1, 1),
                    ]
                )
            else:
                self.guess_state = ca.vcat(
                    [
                        self.time_guess,
                        u0.reshape(-1, 1),
                        x0[:, :num_state].reshape(-1, 1),
                        epsilon_param0.reshape(-1, 1),
                        kappa0.reshape(-1, 1),
                        l0.flatten(),
                    ]
                )

        else:
            lambda0 = []
            for i in range(self.num_waypoints):
                if i > self.i_switch[i_wp]:
                    i_wp += 1
                if i_wp == 0:
                    wp_last = self.init_pose[:3].flatten()
                else:
                    wp_last = self.Wp_ref[i_wp - 1, :3].flatten()
                wp_next = self.Wp_ref[i_wp, :3].flatten()
                if i_wp > 0:
                    interp = (i - self.i_switch[i_wp - 1]) / (
                        self.i_switch[i_wp] - self.i_switch[i_wp - 1]
                    )
                else:
                    interp = i / (self.i_switch[0] + 1e-6)

                pos_guess = (1 - interp) * wp_last + interp * wp_next
                vel_guess = (
                    self.vel_guess * (wp_next - wp_last) / ca.norm_2(wp_next - wp_last)
                )
                # due to new numpy version, squeeze is needed (DM to array)
                vel_guess = np.array(vel_guess).squeeze()
                if x0 is None:
                    x0 = self.init_pose
                else:
                    if num_state == 6:
                        x0 = np.append(
                            x0,
                            np.array(
                                [
                                    pos_guess[0],
                                    pos_guess[1],
                                    pos_guess[2],
                                    0,
                                    0,
                                    0,
                                    vel_guess[0],
                                    vel_guess[1],
                                    vel_guess[2],
                                    0,
                                    0,
                                    0,
                                ]
                            ),
                        )
                    elif num_state == 7:
                        a = np.array(
                            [
                                pos_guess[0],
                                pos_guess[1],
                                pos_guess[2],
                                0,
                                0,
                                0,
                                np.pi / 2.0,
                                vel_guess[0],
                                vel_guess[1],
                                vel_guess[2],
                                0,
                                0,
                                0,
                                0,
                            ]
                        )
                        print(a.shape)
                        x0 = np.append(
                            x0,
                            np.array(
                                [
                                    pos_guess[0],
                                    pos_guess[1],
                                    pos_guess[2],
                                    0,
                                    0,
                                    0,
                                    np.pi / 2.0,
                                    vel_guess[0],
                                    vel_guess[1],
                                    vel_guess[2],
                                    0,
                                    0,
                                    0,
                                    0,
                                ]
                            ),
                        )
                if i == self.num_waypoints - 1:
                    lamg = [0] * (self.num_def_point)
                else:
                    lamg = np.array([1] * (self.num_def_point))
                    # if i < self.i_switch[i_wp]:
                    lamg[range(i_wp)] = 0
                    lamg = lamg.tolist()
                lambda0 += lamg

            x0 = x0.reshape(-1, self.Wp_ref.shape[1])
            self.x_guess = x0
            print("x guess: ", self.x_guess)
            print("time guess: ", self.time_guess)
            # refit the reference points
            for i in range(self.i_switch.shape[0]):
                if num_state == 6:
                    self.Wp_ref[i, [6, 7, 8]] = self.x_guess[
                        self.i_switch[i], [6, 7, 8]
                    ]
                if num_state == 7:
                    self.Wp_ref[i, [7, 8, 9]] = self.x_guess[
                        self.i_switch[i], [7, 8, 9]
                    ]
            lambda0 = np.array(lambda0).reshape(
                self.num_waypoints, (self.num_def_point)
            )
            tolerance_param0 = np.tile(
                np.zeros(self.num_def_point), self.num_waypoints - 1
            )
            self.guess_state = ca.vcat(
                [
                    self.time_guess,
                    u0.reshape(-1, 1),
                    x0[:, :num_state].reshape(-1, 1),
                    lambda0.reshape(-1, 1),
                    tolerance_param0,
                ]
            )

    def setup_target_features(self, traj_function, traj_vel_function, features=None):
        if features is not None:
            self.target_linear_speed = features["linear_speed"]
            self.target_rotation_speed = features["rotation_speed"]
        self.target_trajectory_func = traj_function
        self.target_velocity_func = traj_vel_function

    def get_velocity(self, num_state=6):
        velocity_list = [self.init_pose[num_state:].flatten()]
        x_ = self.X_state.copy()
        for i in range(1, self.num_waypoints - 1):
            velocity_list.append(
                self.average_velocity(
                    x_[i].reshape(-1, 1),
                    x_[i + 1].reshape(-1, 1),
                    self.result_tn / (self.num_waypoints - 1),
                    num_state=num_state,
                ).flatten()
            )

        velocity_list.append(self.Wp_ref[-1][num_state:].flatten())

        self.velocity_list = np.array(velocity_list).reshape(-1, num_state)

    def get_manipulator_state(self, pos_mode=True):
        # get needed datas from results
        traj_manip = []  # the x,y,z position of end of the manipulator
        vel_manip = []  # the dx,dy,dz of end of the manipulator
        # vdiff_uav_man = []
        # vdiff_uav_ground = []
        # vdiff_man_ground = []
        x_state = deepcopy(self.X_state)

        if pos_mode:
            for i in range(self.num_waypoints):
                x_manip = np.array(self.end_effector_pos_func(x_state[i, :])).reshape(
                    -1, 3
                )
                traj_manip.append(x_manip)

                v_manip = np.array(
                    self.end_effector_vel_func(x_state[i, :], self.velocity_list[i, :])
                ).reshape(-1, 3)
                vel_manip.append(v_manip)
        else:
            for i in range(self.num_waypoints):
                x_manip = np.array(
                    self.end_effector_pos_func(self.X_state[i, :])
                ).reshape(-1, 3)
                traj_manip.append(x_manip)

                v_manip = np.array(
                    self.end_effector_vel_func(self.X_state[i, :])
                ).reshape(-1, 3)
                vel_manip.append(v_manip)

                # vdiff_uav_man.append(self.X_state[i, 6:9].reshape(-1, 3) -
                #                      v_manip)

        # vdiff_uav_man = np.array(vdiff_uav_man).reshape(-1, 3)
        # print('the velocity difference on the frame of
        # world between uav and manipulator: \n',vdiff_uav_man)
        # vdiff_uav_ground = np.array(vdiff_uav_ground).reshape(-1, 2)
        # print('the velocity difference on the frame of
        # world between uav and mobile ground robot: \n',vdiff_uav_ground)
        # vdiff_man_ground = np.array(vdiff_man_ground).reshape(-1, 2)
        # print('the velocity difference on the frame of

        self.traj_manip = np.array(traj_manip).reshape(-1, 3)
        self.vel_manip = np.array(vel_manip).reshape(-1, 3)
        print("vel")
        print(self.vel_manip)

    def get_contact_index(self, criterion=0.2):
        contact_time = []
        contact_index = 0
        contact_index_list = []
        for i in range(self.result_epsilon_param.shape[0]):
            if self.result_epsilon_param[i] > criterion:
                contact_time.append(
                    contact_index * self.result_tn / (self.num_waypoints - 1)
                )
                contact_index_list.append(contact_index)
            contact_index += 1
        self.contact_time = np.array(contact_time).reshape(-1, 1)
        self.contact_index_list = np.array(contact_index_list).reshape(-1, 1)

    def get_target_state(
        self,
    ):
        target_state = []
        target_velocity = []
        for i in range(self.num_waypoints):
            target_state.append(
                self.target_trajectory_func(
                    self.result_tn / (self.num_waypoints - 1), i
                )
                .full()
                .flatten()
            )
            target_velocity.append(
                self.target_velocity_func(self.result_tn / (self.num_waypoints - 1), i)
                .full()
                .flatten()
            )
        self.target_state = np.array(target_state).reshape(-1, 3)
        self.target_velocity = np.array(target_velocity).reshape(-1, 3)

    ################
    # PLOT functions
    ################
    def showPath_XYplane(
        self,
    ):
        fig = plt.figure()
        ax = fig.gca()

        ax.scatter(
            self.init_pose[0],
            self.init_pose[1],
            color="r",
            marker="x",
        )
        for i in self.Wp_ref:
            ax.scatter(i[0], i[1], color="b", marker="x")
        ax.plot(self.X_state[:, 0], self.X_state[:, 1], "k-")

        if self.has_contact_target:
            # draw the direction of mobile ground robot
            for i in range(self.X_state.shape[0]):
                if i in self.contact_index_list:
                    ax.scatter(
                        self.target_state[i, 0],
                        self.target_state[i, 1],
                        color="r",
                        marker="o",
                    )
                    ax.plot(
                        [
                            self.target_state[i, 0],
                            self.target_state[i, 0]
                            + 0.1
                            * np.cos(
                                ca.atan2(
                                    self.target_velocity[i, 1],
                                    self.target_velocity[i, 0],
                                )
                            ),
                        ],
                        [
                            self.target_state[i, 1],
                            self.target_state[i, 1]
                            + 0.1
                            * np.sin(
                                ca.atan2(
                                    self.target_velocity[i, 1],
                                    self.target_velocity[i, 0],
                                )
                            ),
                        ],
                        "r-",
                    )
                else:
                    ax.scatter(
                        self.target_state[i, 0],
                        self.target_state[i, 1],
                        color="y",
                        marker="o",
                    )

        for i in range(self.X_state.shape[0]):
            eRb = self.rotation_matrix(self.X_state[i, 3:6])
            ax.plot(
                [self.X_state[i, 0], self.traj_manip[i, 0]],
                [self.X_state[i, 1], self.traj_manip[i, 1]],
                "g-",
            )
            _x_value = np.array(ca.mtimes([eRb, np.array([0.1, 0.0, 0.0])])).squeeze()
            ax.plot(
                [
                    self.X_state[i, 0],
                    self.X_state[i, 0] + _x_value[0],
                ],
                [
                    self.X_state[i, 1],
                    self.X_state[i, 1] + _x_value[1],
                ],
                "r-",
            )
            _y_value = np.array(ca.mtimes([eRb, np.array([0.0, 0.1, 0.0])])).squeeze()
            ax.plot(
                [
                    self.X_state[i, 0],
                    self.X_state[i, 0] + _y_value[0],
                ],
                [
                    self.X_state[i, 1],
                    self.X_state[i, 1] + _y_value[1],
                ],
                "y-",
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Trajectory")
        plt.axis("equal")
        plt.show()

    def showPath_XZplane(self, num_state=6):
        fig = plt.figure()
        ax = fig.gca()

        v = np.ndarray((self.X_state.shape[0], 1))
        for i in range(self.X_state.shape[0]):
            v[i, 0] = np.sqrt(
                self.velocity_list[i, 0] ** 2
                + self.velocity_list[i, 1] ** 2
                + self.velocity_list[i, 2] ** 2
            )

        heatmap = plt.scatter(
            self.X_state[:, 0],
            self.X_state[:, 2],
            c=np.squeeze(v),
            cmap=cm.rainbow,
            edgecolor="none",
            marker="o",
        )
        cbar = plt.colorbar(heatmap, fraction=0.035)
        cbar.set_label("velocity of UAV [m/s]")

        ax.scatter(
            self.init_pose[0],
            self.init_pose[2],
            color="r",
            marker="x",
        )
        for i in self.Wp_ref:
            ax.scatter(i[0], i[2], color="b", marker="x")
        ax.plot(self.X_state[:, 0], self.X_state[:, 2], "k-")

        # draw the trajectory of gripper
        ax.scatter(self.traj_manip[:, 0], self.traj_manip[:, 2], color="g", marker="o")

        # draw the collision area
        if self.has_contact_target:
            for i in range(self.X_state.shape[0]):
                if i in self.contact_index_list:
                    ax.scatter(
                        self.target_state[i, 0],
                        self.target_state[i, 2],
                        color="r",
                        marker="o",
                    )
                    theta = np.arange(0, 1 * np.pi, 0.01)
                    x = self.target_state[i, 0] + self.result_l_param[i] * np.cos(theta)
                    z = self.target_state[i, 2] + self.result_l_param[i] * np.sin(theta)
                    ax.plot(x, z, "r--")
                    # print("the angle position of UAV inside collision area: \
                    #     roll[{0}], pitch[{1}], yaw[{2}]".format(self.X_state[i,3],self.X_state[i,4],self.X_state[i,5]))
                    print(
                        "vx_uav[{0}], vy_uav[{1}], vz_uav[{2}]\n".format(
                            self.X_state[i, num_state + 1],
                            self.X_state[i, num_state + 2],
                            self.X_state[i, num_state + 3],
                        )
                    )
                    print(
                        "vx_man[{0}], vy_man[{1}], vz_man[{2}]\n".format(
                            self.vel_manip[i, 0],
                            self.vel_manip[i, 1],
                            self.vel_manip[i, 2],
                        )
                    )
                    print(
                        "vx_ground[{0}], vy_ground[{1}]\n".format(
                            self.target_velocity[i, 0], self.target_velocity[i, 1]
                        )
                    )
                else:
                    ax.scatter(
                        self.target_state[i, 0],
                        self.target_state[i, 2],
                        color="y",
                        marker="o",
                    )

        for i in range(self.X_state[:, 3].shape[0]):
            eRb = self.rotation_matrix(self.X_state[i, 3:6])
            ax.plot(
                self.X_state[i, 0] + ca.mtimes([eRb, self.montage_offset_b]).full()[0],
                self.X_state[i, 2] + ca.mtimes([eRb, self.montage_offset_b]).full()[2],
                color="g",
                marker="o",
            )
            ax.plot(
                [
                    self.X_state[i, 0],
                    self.X_state[i, 0]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[0],
                ],
                [
                    self.X_state[i, 2],
                    self.X_state[i, 2]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[2],
                ],
                "g-",
            )
            ax.plot(
                [
                    self.X_state[i, 0]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[0],
                    self.traj_manip[i, 0],
                ],
                [
                    self.X_state[i, 2]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[2],
                    self.traj_manip[i, 2],
                ],
                "g-",
            )
            ax.plot(
                [
                    self.X_state[i, 0],
                    self.X_state[i, 0]
                    + ca.mtimes([eRb, np.array([0.1, 0.0, 0.0])]).full().flatten()[0],
                ],
                [
                    self.X_state[i, 2],
                    self.X_state[i, 2]
                    + ca.mtimes([eRb, np.array([0.1, 0.0, 0.0])]).full().flatten()[2],
                ],
                "b-",
            )
            ax.plot(
                [
                    self.X_state[i, 0],
                    self.X_state[i, 0]
                    + ca.mtimes([eRb, np.array([-0.1, 0.0, 0.0])]).full().flatten()[0],
                ],
                [
                    self.X_state[i, 2],
                    self.X_state[i, 2]
                    + ca.mtimes([eRb, np.array([-0.1, 0.0, 0.0])]).full().flatten()[2],
                ],
                "b-",
            )
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title("Trajectory")
        plt.legend(
            [
                "UAV",
                "Point area",
            ]
        )
        plt.axis("equal")
        plt.show()

    def showPath_3D(self, axis_equal=False):
        fig = plt.figure()
        if int(matplotlib.__version__.split(".")[0]) < 3:
            ax = fig.add_subplot(111, projection="3d")  #
        else:
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)
        # ax = fig.gca(111, projection='3d')
        ax.scatter(
            self.init_pose[0],
            self.init_pose[1],
            self.init_pose[2],
            color="r",
            marker="x",
        )
        for i in self.Wp_ref:
            ax.scatter(i[0], i[1], i[2], color="b", marker="x")

        ax.plot(self.X_state[:, 0], self.X_state[:, 1], self.X_state[:, 2], "k-")
        ax.scatter(
            self.traj_manip[:, 0], self.traj_manip[:, 1], self.traj_manip[:, 2], "go"
        )

        if self.has_contact_target:
            for i in range(self.X_state.shape[0]):
                if i in self.contact_index_list:
                    ax.scatter(
                        self.target_state[i, 0],
                        self.target_state[i, 1],
                        self.target_state[i, 2],
                        color="r",
                        marker="o",
                    )
                else:
                    ax.scatter(
                        self.target_state[i, 0],
                        self.target_state[i, 1],
                        self.target_state[i, 2],
                        color="y",
                        marker="o",
                    )

        # draw the head pointing of uav
        for i in range(self.X_state.shape[0]):
            eRb = self.rotation_matrix(self.X_state[i, 3:6])
            ax.plot(
                [
                    self.X_state[i, 0]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[0]
                ],
                [
                    self.X_state[i, 1]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[1]
                ],
                [
                    self.X_state[i, 2]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[2]
                ],
                color="g",
                marker="o",
            )
            ax.plot(
                [
                    self.X_state[i, 0],
                    self.X_state[i, 0]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[0],
                ],
                [
                    self.X_state[i, 1],
                    self.X_state[i, 1]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[1],
                ],
                [
                    self.X_state[i, 2],
                    self.X_state[i, 2]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[2],
                ],
                "g-",
            )
            ax.plot(
                [
                    self.X_state[i, 0]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[0],
                    self.traj_manip[i, 0],
                ],
                [
                    self.X_state[i, 1]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[1],
                    self.traj_manip[i, 1],
                ],
                [
                    self.X_state[i, 2]
                    + ca.mtimes([eRb, self.montage_offset_b]).full().flatten()[2],
                    self.traj_manip[i, 2],
                ],
                "g-",
            )

            # ax.plot([self.X_state[i, 0], self.traj_manip[i, 0]],
            #         [self.X_state[i, 1], self.traj_manip[i, 1]],
            #         [self.X_state[i, 2], self.traj_manip[i, 2]], 'g-')
            ax.plot(
                [
                    self.X_state[i, 0],
                    self.X_state[i, 0]
                    + ca.mtimes([eRb, np.array([0.1, 0.0, 0.0])]).full().flatten()[0],
                ],
                [
                    self.X_state[i, 1],
                    self.X_state[i, 1]
                    + ca.mtimes([eRb, np.array([0.1, 0.0, 0.0])]).full().flatten()[1],
                ],
                [
                    self.X_state[i, 2],
                    self.X_state[i, 2]
                    + ca.mtimes([eRb, np.array([0.1, 0.0, 0.0])]).full().flatten()[2],
                ],
                color="r",
            )
            ax.plot(
                [
                    self.X_state[i, 0],
                    self.X_state[i, 0]
                    + ca.mtimes([eRb, np.array([0.0, 0.1, 0.0])]).full().flatten()[0],
                ],
                [
                    self.X_state[i, 1],
                    self.X_state[i, 1]
                    + ca.mtimes([eRb, np.array([0.0, 0.1, 0.0])]).full().flatten()[1],
                ],
                [
                    self.X_state[i, 2],
                    self.X_state[i, 2]
                    + ca.mtimes([eRb, np.array([0.0, 0.1, 0.0])]).full().flatten()[2],
                ],
                color="y",
            )
            ax.plot(
                [
                    self.X_state[i, 0],
                    self.X_state[i, 0]
                    + ca.mtimes([eRb, np.array([0.0, 0.0, 0.1])]).full().flatten()[0],
                ],
                [
                    self.X_state[i, 1],
                    self.X_state[i, 1]
                    + ca.mtimes([eRb, np.array([0.0, 0.0, 0.1])]).full().flatten()[1],
                ],
                [
                    self.X_state[i, 2],
                    self.X_state[i, 2]
                    + ca.mtimes([eRb, np.array([0.0, 0.0, 0.1])]).full().flatten()[2],
                ],
                color="b",
            )
        if axis_equal:
            # ax.axis('equal')
            ax.set_aspect("auto")
        else:
            ax.set_xlim([np.min(self.X_state[:, 0]), np.max(self.X_state[:, 0])])
            ax.set_ylim(
                [np.min(self.X_state[:, 1]) - 0.8, np.max(self.X_state[:, 1]) + 0.8]
            )
            ax.set_zlim([0, np.max(self.X_state[:, 2])])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("test")

        plt.show()
