#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Wei Luo
Date: 2022-03-08 09:54
LastEditTime: 2022-12-29 16:53:48
Note: this file gives two choice to have Lagrangian, using casadi
taking offset
'''
import numpy as np
import casadi as ca


class DerivationUAV(object):
    def __init__(self,
                 g=9.8066,
                 arm_length=0.34,
                 mass_arm=0.36,
                 mass_quad=1,
                 I_quad=[0.03, 0.028, 0.03],
                 I_arm=[0, 0.0019, 0],
                 is_2d_model=False,
                 frame_size=0.33,
                 motor_torque_const=0.013,
                 montage_offset_b=[0, 0, 0]):
        # self.theta = theta
        self.g = g
        self.arm_length = arm_length  # [m]
        self.mass_arm = mass_arm  # [kg] mass of the manipulator
        self.mass_quad = mass_quad  # [kg] mass of the drone
        self.Ixx = I_quad[0]  # [kgm^2]
        self.Iyy = I_quad[1]  # [kgm^2]
        self.Izz = I_quad[2]  # [kgm^2]
        self.I_quad = np.array(I_quad)
        self.I_arm = np.array(I_arm)  # [kgm^2] manipulator rotational inertia
        self.is_2d_model = is_2d_model  # if generate only a 2d model
        self.frame_size = frame_size  # size of frame [m]
        self.motor_torque_const = motor_torque_const  # torque moment const [m]
        self.montage_offset_b = np.array(
            montage_offset_b)  # montage offset ref to body frame

    def get_Lagrangian_casadi(self):
        # set symbolic parameters using casadi
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        z = ca.SX.sym('z')
        phi = ca.SX.sym('phi')
        theta = ca.SX.sym('theta')
        psi = ca.SX.sym('psi')
        alpha = ca.SX.sym('alpha')
        states_2 = ca.vertcat(x, y, z, phi, theta, psi, alpha)
        p = ca.vertcat(x, y, z)

        # set states_dot
        d_x = ca.SX.sym('dx')
        d_y = ca.SX.sym('dy')
        d_z = ca.SX.sym('dz')
        d_phi = ca.SX.sym('dphi')
        d_theta = ca.SX.sym('dtheta')
        d_psi = ca.SX.sym('dpsi')
        d_alpha = ca.SX.sym('dalpha')
        d_states = ca.vertcat(d_x, d_y, d_z, d_phi, d_theta, d_psi, d_alpha)
        states = ca.vertcat(states_2, d_states)
        p_dot = ca.vertcat(d_x, d_y, d_z)

        Rz = np.array([[ca.cos(psi), -ca.sin(psi), 0],
                       [ca.sin(psi), ca.cos(psi), 0], [0, 0, 1]])
        Ry = np.array([[ca.cos(theta), 0, ca.sin(theta)], [0, 1, 0],
                       [-ca.sin(theta), 0, ca.cos(theta)]])
        Rx = np.array([[1, 0, 0], [0, ca.cos(phi), -ca.sin(phi)],
                       [0, ca.sin(phi), ca.cos(phi)]])
        eRb = ca.mtimes([Rz, Ry, Rx])  # body to inertial
        e3 = np.array([0, 0, 1]).reshape(-1, 1)
        # T_matrix = np.array(
        #     [[1, ca.sin(phi) * ca.tan(theta),
        #       ca.cos(phi) * ca.tan(theta)], [0.0,
        #                                      ca.cos(phi), -ca.sin(phi)],
        #      [0.0,
        #       ca.sin(phi) / ca.cos(theta),
        #       ca.cos(phi) / ca.cos(theta)]])
        # d_xi = np.array([d_phi, d_theta, d_psi])

        # # skew matrix of body angular velocity
        # omega_b = ca.mtimes(ca.inv(T_matrix), d_xi)
        # skew_b = ca.skew(omega_b)
        # # angular velocity in world coordinates
        # omega = ca.mtimes([eRb, omega_b])

        T_matrix = np.array([[1, 0.0, -ca.sin(theta)],
                             [0.0,
                              ca.cos(phi),
                              ca.sin(phi) * ca.cos(theta)],
                             [0.0, -ca.sin(phi),
                              ca.cos(phi) * ca.cos(theta)]])
        bW = ca.mtimes([T_matrix, np.array([d_phi, d_theta, d_psi])])
        # the angle velocity of uav in the frame of world
        # the angle velocity of uav in the frame of world
        eW = ca.mtimes([eRb, bW])
        skew_eW_ca = ca.skew(eW)

        # manipulator mass middle point
        # on the quadrotor frame
        bO = np.array([ca.cos(alpha), 0, -ca.sin(alpha)
                       ]) * self.arm_length / 2. + self.montage_offset_b
        # on the initial frame
        eO = p + ca.mtimes([eRb, bO])

        Jl1 = np.array([-ca.sin(alpha), 0, -ca.cos(alpha)
                        ]) * self.arm_length / 2.
        bO_dot = Jl1 * d_alpha
        # eO_dot = p_dot + ca.mtimes([eRb, skew_b, bO])\
        #     + ca.mtimes([eRb, bO_dot]
        #                 )  # manipulator CoM velocity in the frame of world

        eO_dot = p_dot + ca.mtimes([skew_eW_ca, eRb, bO])\
            + ca.mtimes([eRb, bO_dot]
                        )  # manipulator CoM velocity in the frame of world

        # manipulator angle velocity in the frame of world
        bRm = np.array([[ca.cos(alpha), 0, ca.sin(alpha)], [0, 1, 0],
                        [-ca.sin(alpha), 0, ca.cos(alpha)]])
        Jol1 = np.array([0, 1, 0])
        bWl1 = Jol1 * d_alpha
        eWl1 = eW + ca.mtimes([eRb, bWl1])
        mani_angular_velocity_b = ca.mtimes([bRm.T, eRb.T, eWl1])
        manipulator_inertia_moment = np.diag(self.I_arm)
        # Kinetic energy
        K_quad = 1 / 2 * ca.mtimes([self.mass_quad, p_dot.T, p_dot])\
            + 1 / 2 * ca.mtimes([bW.T, np.diag(self.I_quad), bW])  # kinetic energy of uav
        K_arm = 1 / 2 * ca.mtimes([self.mass_arm, eO_dot.T, eO_dot])\
            + 1 / 2 * ca.mtimes([mani_angular_velocity_b.T, manipulator_inertia_moment, mani_angular_velocity_b]
                                )  # kinetic energy of manipulator
        K = K_quad + K_arm

        # potential energy
        U_quad = ca.mtimes([self.mass_quad, self.g, e3.T, p])
        U_arm = ca.mtimes([self.mass_arm, self.g, e3.T, eO])
        U = U_quad + U_arm

        L = K - U
        fct_L = ca.Function('fct_L', [states_2, d_states], [L])

        L_ddstates = ca.gradient(L, d_states)
        fct_L_ddstates = ca.Function('fct_L_ddstates', [states_2, d_states],
                                     [L_ddstates])
        # generalized forces
        # according to modeling, f1,f2,f3,f4 and tau1,tau2,tau3,tau4 are the thrust and moment \
        # which are produced from four motors and dependent on Rotation speed of motors.
        # a1 and a2 are the distance from motor to X-axis and Y-axis in the frame of uav.
        # Here we choose 4 inputs + 1(input for manipulator) for the whole system to express the relationship between them.
        # Thrust_total: U1 = f1 + f2 + f3 +f4
        # Moment_total_X: U2 = a1*(f1 + f4 - f2 - f3)
        # Moment_total_Y: U3 = a2*(f3 + f4 - f1 - f2)
        # Moment_total_Z: U4 = tau1 + tau2 + tau3 + tau4
        # Moment_manipulator: taum
        U1 = ca.SX.sym("U1")  # motor 1
        U2 = ca.SX.sym("U2")  # motor 2
        U3 = ca.SX.sym("U3")  # motor 3
        U4 = ca.SX.sym("U4")  # motor 4
        taum = ca.SX.sym("taum")  # torque from manipulator
        u = ca.vertcat(U1, U2, U3, U4, taum)
        self.n_controls = u.size()[0]
        f_local = U1 + U2 + U3 + U4
        f_xyz = ca.mtimes([eRb, e3, f_local])
        moment_x = (U2 + U3 - U1 - U4) * self.frame_size / 2 / np.sqrt(2)
        moment_y = (U2 + U4 - U1 - U3) * self.frame_size / 2 / np.sqrt(2)
        moment_z = (U3 + U4 - U1 - U2) * self.motor_torque_const
        rhs_f = ca.vertcat(f_xyz, moment_x, moment_y - taum, moment_z, taum)
        fct_f = ca.Function('fct_f', [states_2, u], [rhs_f])

        self.n_states = states.size()[0]

        # get the end-position of the manipulator
        b_Mani_End = np.array([ca.cos(alpha), 0, -ca.sin(alpha)
                               ]) * self.arm_length + self.montage_offset_b
        e_Mani_End = p + ca.mtimes([eRb, b_Mani_End])
        self.end_position_function = ca.Function('fct_e_Mani_End', [states_2],
                                                 [e_Mani_End])

        # get the end-velocity of the manipulator
        Jl1_Mani_End = np.array([-ca.sin(alpha), 0, -ca.cos(alpha)
                                 ]) * self.arm_length
        b_Mani_End_dot = Jl1_Mani_End * d_alpha
        e_Mani_End_dot = p_dot + ca.mtimes([skew_eW_ca, eRb, b_Mani_End])\
            + ca.mtimes([eRb, b_Mani_End_dot])
        self.end_velocity_function = ca.Function('fct_e_Mani_End_dot',
                                                 [states_2, d_states],
                                                 [e_Mani_End_dot])

        return fct_L, fct_f, fct_L_ddstates

    def get_endeffector_position_function(self):
        return self.end_position_function

    def get_endeffector_velocity_function(self):
        return self.end_velocity_function

    def get_model_number_states(self):
        return self.n_states, self.n_controls
