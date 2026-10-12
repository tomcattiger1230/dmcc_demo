#!/usr/bin/env python
# coding=UTF-8
"""
Author: Wei Luo
Date: 2021-09-27 17:16:11
LastEditors: Wei Luo
LastEditTime: 2024-12-29 10:43:49
Note: to implement a dmcc
"""

import casadi as ca
import numpy as np

# get system dynamic equations
from .derivation_uav_manipulator_pos_state_force import DerivationUAV
from .dmcc_base_force_offset_lie import DMCC_base


class DMCC_UAV_Manipulator(DMCC_base):
    def __init__(
        self,
        manipulator_length=0.34,
        mass_quadrotor=1.659,
        mass_manipulator=0.36,
        inertia_moment=[0.0348, 0.0459, 0.0977],
        manipulator_inertia_moment=[0, 0.0019, 0],
        has_contact_target=False,
        has_manipulator=True,
        target_features=None,
        trajectory_function=None,
        vel_trajectory_function=None,
        montage_offset_b=[0, 0, -0.08],
        g=9.8066,
    ):
        super().__init__(
            manipulator_length,
            mass_quadrotor,
            inertia_moment=inertia_moment,
            has_contact_target=has_contact_target,
            has_manipulator=has_manipulator,
            manipulator_inertia_moment=manipulator_inertia_moment,
            mass_manipulator=mass_manipulator,
            montage_offset_b=montage_offset_b,
            g=g,
        )

        self.num_def_point = None
        self.has_contact_target = has_contact_target
        if not has_manipulator:
            raise ValueError("This 7-DOF model includes a manipulator; the flag cannot remove its dynamics")

        if self.has_contact_target:
            self.setup_target_features(
                traj_function=trajectory_function,
                traj_vel_function=vel_trajectory_function,
                features=target_features,
            )

    def initialization_formulation(self, kappa0=2, *, build_solver=True,
                                   dynamics_convention="legacy", objective_mode="regularized",
                                   waypoint_radius=.05, minimum_pass_speed=None, contact_smoothing=1e-6):
        """Build the problem; opt into consistent Euler-coordinate mechanics."""
        if dynamics_convention not in {"legacy", "euler_consistent"}:
            raise ValueError("Unknown dynamics convention")
        if objective_mode not in {"regularized", "minimum_time"}:
            raise ValueError("Unknown objective mode")
        if waypoint_radius <= 0 or contact_smoothing <= 0:
            raise ValueError("Radius and contact smoothing must be positive")
        if minimum_pass_speed is not None and minimum_pass_speed <= 0:
            raise ValueError("Pass speed must be positive")
        inertia = np.array([self.Ixx, self.Iyy, self.Izz])
        if dynamics_convention == "euler_consistent" and (
            not np.isfinite(inertia).all() or np.any(inertia <= 0) or 2*inertia.max() > inertia.sum()
        ):
            raise ValueError("Inertia violates physical principal-moment inequalities")
        self.dynamics_convention = dynamics_convention
        self.objective_mode = objective_mode
        self.waypoint_radius = waypoint_radius
        self.minimum_pass_speed = minimum_pass_speed
        if not hasattr(self, '_baseline_difference_rpy'):
            self._baseline_difference_rpy = self.difference_rpy
        self.difference_rpy = (self.euler_coordinate_rates if dynamics_convention == 'euler_consistent'
                              else self._baseline_difference_rpy)
        self.kappa_0 = kappa0

        self.init_state_guess(with_additional_relaxation=False, num_state=7)

        # get Lagrangian
        quad_model = DerivationUAV(
            g=self.g_,
            arm_length=self.arm_length,
            mass_quad=self.mass_quadrotor,
            mass_arm=self.mass_manipulator,
            I_quad=[self.Ixx, self.Iyy, self.Izz],
            I_arm=self.I_manipulator,
            montage_offset_b=self.montage_offset_b,
            is_2d_model=False,
        )

        # get Lagrangian
        # l_function: Lagrangian function L = K - U
        # f_function: force/moment function f = [f, m_roll, m_pitch, m_yaw]
        # l_dot_function: \partial{L}/\partial{\dot{x}}
        l_function, f_function, l_dot_function = quad_model.get_Lagrangian_casadi()
        if dynamics_convention == "euler_consistent":
            q_force, u_force = ca.SX.sym('q_force', 7), ca.SX.sym('u_force', 5)
            phi, theta = q_force[3], q_force[4]
            transform = ca.vertcat(ca.horzcat(1, 0, -ca.sin(theta)),
                                  ca.horzcat(0, ca.cos(phi), ca.sin(phi)*ca.cos(theta)),
                                  ca.horzcat(0, -ca.sin(phi), ca.cos(phi)*ca.cos(theta)))
            raw = f_function(q_force, u_force)
            # alpha is relative to the body: internal motor virtual work is
            # tau_m*d(alpha), with no extra common-rotation torque.
            torque = raw[3:6]+ca.vertcat(0, u_force[4], 0)
            mapped = ca.vertcat(raw[:3], transform.T@torque, raw[6])
            f_function = ca.Function('generalized_force', [q_force, u_force], [mapped])
        self.generalized_force = f_function
        self.end_effector_pos_func = quad_model.get_endeffector_position_function()
        self.end_effector_vel_func = quad_model.get_endeffector_velocity_function()
        self.n_full_states, self.n_controls = quad_model.get_model_number_states()
        print(self.n_full_states, self.n_controls)
        self.n_states = int(self.n_full_states / 2)
        print(
            "the quadrotor model has {0} states, and {1} controls".format(
                self.n_states, self.n_controls
            )
        )

        # DMCC
        Tn = ca.SX.sym("Tn")
        self.dt = Tn / (self.num_waypoints - 1)
        U = ca.SX.sym("U", self.n_controls, self.num_waypoints)
        X = ca.SX.sym("X", self.n_states, self.num_waypoints)
        init_pose = ca.SX.sym("init_pose", int(self.n_states * 2))
        W_ref = ca.SX.sym("W_ref", int(self.n_states * 2), self.num_def_point)

        if self.has_contact_target:
            # index for contact points
            epsilon_param = ca.SX.sym("epsilon_param", self.num_waypoints - 1)
            kappa_param = ca.SX.sym("kappa_param", self.num_waypoints)
            # contact / angle relax parameters
            c_r_param = ca.SX.sym("l_param", self.num_waypoints - 1)
        else:
            lambda_param_ = ca.SX.sym(
                "lambda_param", self.num_def_point, self.num_waypoints
            )
            tolerance_param_ = ca.SX.sym(
                "tolerance", self.num_def_point, self.num_waypoints - 1
            )

        # cost function
        obj = Tn
        Q = np.array([10, 10, 10, 10, 10])
        u0 = np.array(
            [(self.mass_quadrotor + self.mass_manipulator) * self.g_ / 4.0] * 4 + [0.0]
        )
        if objective_mode == 'regularized':
            for i in range(0, self.num_waypoints):
                obj = obj + 0.0003 * self.dt * ca.dot((U[:, i] - u0) * Q, (U[:, i] - u0))
            for i in range(self.num_waypoints - 1):
                obj += 0.0003 * ca.dot((X[:3, i + 1] - X[:3, i]), (X[:3, i + 1] - X[:3, i]))

        # equal constraints
        g = []
        P = []
        for i in range(int(self.n_states * 2)):
            P.append(init_pose[i])
        for i in range(int(self.n_states * 2)):
            P.append(W_ref[i, -1])
        P = np.array(P)

        # initial position and end position
        g.append(X[:, 0] - P[: self.n_states])
        g.append(X[:, -1] - P[2 * self.n_states : 3 * self.n_states])
        if objective_mode == 'regularized':
            g.append(U[:, 0] - u0)
            g.append(U[:, -1] - u0)

        # discrete lagrange equations
        q_nm1 = ca.SX.sym("qnm1", self.n_states)
        q_n = ca.SX.sym("qn", self.n_states)
        q_np1 = ca.SX.sym("qnp1", self.n_states)
        D2L_d = ca.gradient(
            self.discrete_lagrange_verlet(self.dt, q_nm1, q_n, l_function, num_state=7), q_n
        )
        D1L_d = ca.gradient(
            self.discrete_lagrange_verlet(self.dt, q_n, q_np1, l_function, num_state=7), q_n
        )
        d_EulerLagrange = ca.Function("dEL", [Tn, q_nm1, q_n, q_np1], [D2L_d + D1L_d])
        q_b = ca.SX.sym("q_b", self.n_states)
        q_b_dot = ca.SX.sym("q_b_dot", self.n_states)
        D2L = l_dot_function(q_b, q_b_dot)
        d_EulerLagrange_init = ca.Function(
            "dEl_init", [Tn, q_b, q_b_dot, q_n, q_np1], [D2L + D1L_d]
        )
        d_EulerLagrange_end = ca.Function(
            "dEl_end", [Tn, q_b, q_b_dot, q_nm1, q_n], [-D2L + D2L_d]
        )

        for i in range(1, self.num_waypoints - 1):
            f_d_nm1 = self.discrete_forces_v2(
                self.dt, f_function, X[:, i - 1], U[:, i - 1], U[:, i]
            )
            f_d_n = self.discrete_forces_v2(
                self.dt, f_function, X[:, i], U[:, i], U[:, i + 1]
            )
            if dynamics_convention == 'euler_consistent':
                f_d_nm1 = .5*self.dt*f_function(X[:, i], U[:, i])
                f_d_n = .5*self.dt*f_function(X[:, i], U[:, i])
            sum = (
                # dEL takes total time, not the already divided step self.dt.
                d_EulerLagrange(Tn, X[:, i - 1], X[:, i], X[:, i + 1])
                + f_d_nm1
                + f_d_n
            )
            g.append(sum)

        # boundary condition (x_0, x_end)
        f_0 = self.discrete_forces_v2(self.dt, f_function, X[:, 0], U[:, 0], U[:, 1])
        if dynamics_convention == 'euler_consistent':
            f_0 = .5*self.dt*f_function(X[:, 0], U[:, 0])
        g.append(
            d_EulerLagrange_init(
                Tn,
                P[: self.n_states],
                P[self.n_states : 2 * self.n_states],
                X[:, 0],
                X[:, 1],
            )
            + f_0
        )
        f_N_1 = self.discrete_forces_v2(
            self.dt,
            f_function,
            X[:, self.num_waypoints - 2],
            U[:, self.num_waypoints - 2],
            U[:, self.num_waypoints - 1],
        )
        if dynamics_convention == 'euler_consistent':
            f_N_1 = .5*self.dt*f_function(X[:, -1], U[:, -1])
        g.append(
            d_EulerLagrange_end(
                Tn,
                P[2 * self.n_states : 3 * self.n_states],
                P[3 * self.n_states : 4 * self.n_states],
                X[:, self.num_waypoints - 2],
                X[:, self.num_waypoints - 1],
            )
            + f_N_1
        )

        if self.has_contact_target:
            # initial value of kappa_param = init kappa
            # endpoint value of kappa_param = 0
            g.append(kappa_param[0] - self.kappa_0)
            g.append(kappa_param[-1])

            # kappa_param[i+1] = kappa_param[i]-epsilon[i]
            for i in range(self.num_waypoints - 1):
                g.append(kappa_param[i + 1] - kappa_param[i] + epsilon_param[i])

            # epsilon_param[i]*(|End_Manip-pickup_point|-relax)==0
            for i in range(self.num_waypoints - 1):
                # middle state
                mid_result_ = X[:, i]
                contact_error = (self.end_effector_pos_func(mid_result_)
                                 - self.target_trajectory_func(self.dt, i))
                dis_temp = ca.sqrt(ca.sumsqr(contact_error)+contact_smoothing**2)-contact_smoothing
                g.append(epsilon_param[i] * (dis_temp - c_r_param[i]))

        else:
            for j in range(self.num_def_point):
                g.append(lambda_param_[j, 0] - 1.0)
                g.append(lambda_param_[j, -1])
            for i in range(self.num_waypoints - 1):
                for j in range(self.num_def_point):
                    mu_ = -lambda_param_[j, i + 1] + lambda_param_[j, i]
                    mid_result_ = X[:, i+1]
                    cost_ = (
                        ca.sumsqr(mid_result_[:3] - W_ref[:3, j])
                        - tolerance_param_[j, i]
                    )
                    temp_ = mu_ * cost_
                    g.append(temp_)

        self.equal_constraint_length = int(ca.vertcat(*g).numel())
        print(
            "Total number of equality constraints {}:".format(
                self.equal_constraint_length
            )
        )

        # unequal constraints

        # velocity constraints
        for i in range(self.num_waypoints - 1):
            g.append(self.average_velocity(X[:, i], X[:, i + 1], self.dt, num_state=7))

        self.waypoint_inequality_bounds = []
        if not self.has_contact_target:
            for i in range(self.num_waypoints-1):
                for j in range(self.num_def_point):
                    g.append(lambda_param_[j, i]-lambda_param_[j, i+1])
                    self.waypoint_inequality_bounds.append((0., ca.inf))
            for i in range(self.num_waypoints):
                for j in range(self.num_def_point-1):
                    g.append(lambda_param_[j, i]-lambda_param_[j+1, i])
                    self.waypoint_inequality_bounds.append((-ca.inf, 0.))
            if minimum_pass_speed is not None:
                for i in range(self.num_waypoints-1):
                    velocity = self.average_velocity(X[:, i], X[:, i+1], self.dt, num_state=7)
                    for j in range(self.num_def_point-1):
                        mu = lambda_param_[j, i]-lambda_param_[j, i+1]
                        g.append(mu*(minimum_pass_speed**2-ca.sumsqr(velocity[:3])))
                        self.waypoint_inequality_bounds.append((-ca.inf, 0.))

        if self.has_contact_target:
            # # 1. kappa_param[i+1] <= kappa_param[i]
            for i in range(self.num_waypoints - 1):
                mid_result_ = X[:, i]

                mid_velocity_ = self.average_velocity(
                    X[:, i], X[:, i + 1], self.dt, num_state=7
                )
                v1 = self.end_effector_vel_func(mid_result_, mid_velocity_)
                v2 = self.target_velocity_func(self.dt, i)
                # v_ = ca.cross(v1, v2)
                # g.append(epsilon_param[i] * ca.norm_2(v_))
                speed_error = ca.sqrt(ca.sumsqr(v1-v2)+contact_smoothing**2)-contact_smoothing
                g.append(epsilon_param[i] * speed_error)

                # heading of uav
                unit_e = np.array([1, 0, 0]).reshape(-1, 1)
                angle_ = self.average_rpy(X[3:6, i], X[3:6, i + 1])
                # angle_ = X[3:6, i]
                unit_heading_uav = ca.mtimes([self.rotation_matrix(angle_), unit_e])
                g.append(epsilon_param[i]*self.heading_error(unit_heading_uav, v2))
                # g.append(epsilon_param[i] *
                #          ((angle_[2] - ca.atan2(v2[1], v2[0]))**2))

            # 2. Endposition_manipulator z >= -self.d
            for i in range(self.num_waypoints - 1):
                mid_result_ = X[:, i]
                g.append(
                    self.end_effector_pos_func(mid_result_)[2]
                    - self.target_trajectory_func(self.dt, i)[2]
                )

            optimization_target = ca.vcat(
                [
                    Tn,
                    ca.reshape(U, -1, 1),
                    ca.reshape(X, -1, 1),
                    ca.reshape(epsilon_param, -1, 1),
                    ca.reshape(kappa_param, -1, 1),
                    ca.reshape(c_r_param, -1, 1),
                ]
            )
            print("state x should be in {} size".format(optimization_target.shape))
        else:
            optimization_target = ca.vcat(
                [
                    Tn,
                    ca.reshape(U, -1, 1),
                    ca.reshape(X, -1, 1),
                    ca.reshape(lambda_param_, -1, 1),
                    ca.reshape(tolerance_param_, -1, 1),
                ]
            )
            print("state x should be in {} size".format(optimization_target.shape))

        self.unequal_constraint_length = (
            int(ca.vertcat(*g).numel()) - self.equal_constraint_length
        )
        print(
            "Total number of unequal constraints {}:".format(
                self.unequal_constraint_length
            )
        )

        optimization_params = ca.vcat(
            [ca.reshape(init_pose, -1, 1), ca.reshape(W_ref, -1, 1)]
        )

        nlp_def = {
            "x": optimization_target,
            "f": obj,
            "p": optimization_params,
            "g": ca.vertcat(*g),
        }

        opts_setting = {
            "ipopt.max_iter": 2000,
            "ipopt.print_level": 3,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-5,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }

        self.nlp_def = nlp_def
        self.solver_options = opts_setting
        if build_solver:
            self.solver = ca.nlpsol("solver", "ipopt", nlp_def, opts_setting)

    def get_results(self, parameters, lb_x, ub_x, lb_g, ub_g, init_state=None):
        if len(lb_x) != self.nlp_def['x'].numel() or len(ub_x) != self.nlp_def['x'].numel():
            raise ValueError('Variable bounds do not match the formulation')
        if len(lb_g) != self.nlp_def['g'].numel() or len(ub_g) != self.nlp_def['g'].numel():
            raise ValueError('Constraint bounds do not match the formulation')
        if init_state is None:
            sol = self.solver(
                x0=self.guess_state,
                p=parameters,
                lbx=lb_x,
                ubx=ub_x,
                lbg=lb_g,
                ubg=ub_g,
            )
        else:
            sol = self.solver(
                x0=init_state, p=parameters, lbx=lb_x, ubx=ub_x, lbg=lb_g, ubg=ub_g
            )
        result = sol["x"].full()
        self.solver_stats = self.solver.stats()
        values = sol['g'].full().ravel()
        state = result.ravel()
        self.constraint_violation = float(max(np.max(np.asarray(lb_g)-values),
                                             np.max(values-np.asarray(ub_g)),
                                             np.max(np.asarray(lb_x)-state),
                                             np.max(state-np.asarray(ub_x)), 0.))
        if (not self.solver_stats['success'] or not np.isfinite(state).all()
                or not np.isfinite(values).all() or not np.isfinite(self.constraint_violation)
                or self.constraint_violation > 1e-5):
            raise RuntimeError(f"Rejected solution: {self.solver_stats['return_status']}; violation={self.constraint_violation}")
        self.solution = state.copy()
        self.result_tn = result[0]
        print("Tn : {0}".format(self.result_tn))
        self.u_ = result[1 : self.num_waypoints * self.n_controls + 1]
        self.U_state = self.u_.reshape(-1, self.n_controls)
        print("Inputs U :\n", self.U_state)
        self.result_x = result[
            self.num_waypoints * self.n_controls
            + 1 : self.num_waypoints * self.n_controls
            + 1
            + (self.num_waypoints) * self.n_states
        ]
        self.X_state = self.result_x.reshape(-1, self.n_states)
        self.get_velocity(num_state=7)
        self.get_manipulator_state()
        print("estimated velocity: {0}".format(self.velocity_list[:, :2]))
        # print(self.velocity_list)
        # print(self.velocity_list.shape)
        self.X_state = np.hstack((self.X_state, self.velocity_list))
        print("States X :\n", self.X_state)

        if self.has_contact_target:
            self.result_epsilon_param = result[
                self.num_waypoints * self.n_controls
                + 1
                + (self.num_waypoints)
                * self.n_states : self.num_waypoints
                * self.n_controls
                + 1
                + (self.num_waypoints) * self.n_states
                + (self.num_waypoints - 1)
            ]
            print("epsilon parameter: {}".format(self.result_epsilon_param))

            self.result_kappa_param = result[
                self.num_waypoints * self.n_controls
                + 1
                + (self.num_waypoints) * self.n_states
                + (self.num_waypoints - 1) : self.num_waypoints * self.n_controls
                + 1
                + (self.num_waypoints) * self.n_states
                + (self.num_waypoints - 1)
                + (self.num_waypoints)
            ]
            print("kappa parameter: {}".format(self.result_kappa_param))

            self.result_l_param = result[
                self.num_waypoints * self.n_controls
                + 1
                + (self.num_waypoints) * self.n_states
                + (self.num_waypoints - 1)
                + (self.num_waypoints) : self.num_waypoints * self.n_controls
                + 1
                + (self.num_waypoints) * self.n_states
                + (self.num_waypoints - 1)
                + (self.num_waypoints)
                + (self.num_waypoints - 1)
            ]
            self.get_target_state()
            self.get_contact_index()
        else:
            self.result_lambda_param = result[
                self.num_waypoints * self.n_controls
                + 1
                + (self.num_waypoints)
                * self.n_states : self.num_waypoints
                * self.n_controls
                + 1
                + (self.num_waypoints) * self.n_states
                + self.num_waypoints * self.num_def_point
            ]
            print(self.result_lambda_param.reshape(-1, self.num_def_point))

    def save_final_result_as_npy(self, file_name):
        from pathlib import Path
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, "wb") as f:
            np.save(f, self.result_tn)
            np.save(f, self.X_state)
            np.save(f, self.U_state)
            np.save(f, self.velocity_list)
            np.save(f, self.traj_manip)
            np.save(f, self.vel_manip)
            if self.has_contact_target:
                np.save(f, self.target_state)
                np.save(f, self.target_velocity)
                np.save(f, self.result_epsilon_param)
                np.save(f, self.result_kappa_param)
                np.save(f, self.result_l_param)
            else:
                np.save(f, self.result_lambda_param)

    @staticmethod
    def euler_coordinate_rates(first, second, dt):
        difference = second-first
        return ca.atan2(ca.sin(difference), ca.cos(difference))/dt

    @staticmethod
    def heading_error(heading, target_velocity):
        """Static targets have no direction; moving targets reject antiparallel headings."""
        speed2 = ca.sumsqr(target_velocity[:2])
        source = heading[:2]/ca.sqrt(ca.sumsqr(heading[:2])+1e-12)
        target = target_velocity[:2]/ca.sqrt(speed2+1e-12)
        return speed2/(speed2+1e-12)*ca.sumsqr(source-target)

    def pack_parameters(self, initial_pose=None, waypoints=None):
        initial = np.asarray(self.init_pose if initial_pose is None else initial_pose).ravel()
        refs = np.asarray(self.Wp_ref if waypoints is None else waypoints)
        if (initial.shape != (14,) or refs.shape != (self.num_def_point, 14)
                or not np.isfinite(initial).all() or not np.isfinite(refs).all()):
            raise ValueError('Expected a 14-D start and an (M,14) reference array')
        return np.r_[initial, refs.ravel()]
