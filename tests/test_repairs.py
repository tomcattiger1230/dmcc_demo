"""Physical and task regression tests for the corrected public demo."""

import contextlib
import io
import unittest

import casadi as ca
import numpy as np
from dmcc_model.bounds import build_solver_bounds
from dmcc_model.dmcc_base_force_offset_lie import DMCC_base
from dmcc_model.uav_manipulator_DMCC_3D_discrete_pos_force import DMCC_UAV_Manipulator
from uav_mani_DMCC_demo import build_demo, fixed_target, reference_state


def setup(mode="legacy", **options):
    refs = np.stack([reference_state([1, 0, 1]), reference_state([2, 0, 1])])
    start = reference_state([0, 0, 1])
    model = DMCC_UAV_Manipulator(inertia_moment=[0.03, 0.04, 0.05])
    with contextlib.redirect_stdout(io.StringIO()):
        model.get_path_waypoints(
            start, refs, avg_vel=1.0, num_waypoints=11, i_switch=np.array([5, 10])
        )
        model.initialization_formulation(build_solver=False, dynamics_convention=mode, **options)
    return model, refs


class RegressionTests(unittest.TestCase):
    def test_total_time_is_divided_once_in_actual_interior_equations(self):
        for mode in ("legacy", "euler_consistent"):
            model, _ = setup(mode)
            n = model.num_waypoints
            q = np.tile(reference_state([0, 0, 1])[:7, None], (1, n))
            q[0] = 0.5 * (0.1 * np.arange(n)) ** 2  # Constant acceleration 1 m/s^2.
            seed = np.asarray(model.guess_state).ravel().copy()
            seed[0] = 1.0
            seed[1 : 1 + 5 * n] = 0
            seed[1 + 5 * n : 1 + 12 * n] = q.ravel(order="F")
            nlp = model.nlp_def
            evaluate = ca.Function("time_regression", [nlp["x"], nlp["p"]], [nlp["g"]])
            values = np.asarray(evaluate(seed, model.pack_parameters())).ravel()
            # These are the full assembled original equations, not a surrogate dEL.
            np.testing.assert_allclose(
                values[24 : 24 + 7 * (n - 2) : 7], -model.total_mass * 0.1, atol=1e-10
            )

    def test_initial_guess_does_not_mutate_terminal_or_caller(self):
        model, refs = setup()
        self.assertFalse(np.shares_memory(refs, model.Wp_ref))
        np.testing.assert_array_equal(refs[-1, 7:], np.zeros(7))
        np.testing.assert_array_equal(model.Wp_ref, refs)
        self.assertTrue(np.isfinite(np.asarray(model.guess_state)).all())

    def test_waypoint_bounds_and_finite_center_jacobian(self):
        model, _ = setup(minimum_pass_speed=0.3)
        lbx, ubx, lbg, ubg = build_solver_bounds(model)
        self.assertEqual(len(lbx), model.nlp_def["x"].numel())
        self.assertEqual(len(lbg), model.nlp_def["g"].numel())
        np.testing.assert_allclose(ubx[-2 * 10 :], 0.05**2)
        nlp = model.nlp_def
        derivative = ca.Function(
            "finite_waypoint_jac", [nlp["x"], nlp["p"]], [ca.jacobian(nlp["g"], nlp["x"])]
        )
        self.assertTrue(
            np.isfinite(np.asarray(derivative(model.guess_state, model.pack_parameters()))).all()
        )

    def test_reverse_order_and_negative_progress_are_rejected(self):
        model, _ = setup()
        n, m = model.num_waypoints, model.num_def_point
        seed = np.asarray(model.guess_state).ravel().copy()
        lambda_offset = 1 + 12 * n
        lam = np.ones((m, n))
        lam[1, 2:] = 0
        lam[0, 5:] = 0
        seed[lambda_offset : lambda_offset + m * n] = lam.ravel(order="F")
        nlp = model.nlp_def
        evaluate = ca.Function("order_regression", [nlp["x"], nlp["p"]], [nlp["g"]])
        values = np.asarray(evaluate(seed, model.pack_parameters())).ravel()
        order_start = model.equal_constraint_length + 7 * (n - 1) + m * (n - 1)
        self.assertGreater(values[order_start : order_start + n * (m - 1)].max(), 0.5)
        lam[0, 3] = 0.0
        lam[0, 4] = 1.0
        seed[lambda_offset : lambda_offset + m * n] = lam.ravel(order="F")
        values = np.asarray(evaluate(seed, model.pack_parameters())).ravel()
        mono_start = model.equal_constraint_length + 7 * (n - 1)
        self.assertLess(values[mono_start:order_start].min(), -0.5)

    def test_runtime_reference_changes_are_effective(self):
        model, refs = setup()
        params = model.pack_parameters()
        changed = refs.copy()
        changed[0, 0] += 0.5
        nlp = model.nlp_def
        evaluate = ca.Function("param_regression", [nlp["x"], nlp["p"]], [nlp["g"]])
        before = np.asarray(evaluate(model.guess_state, params))
        after = np.asarray(evaluate(model.guess_state, model.pack_parameters(waypoints=changed)))
        self.assertGreater(np.max(np.abs(after - before)), 0.01)

    def test_correct_internal_motor_power_and_euler_rates(self):
        model, _ = setup("euler_consistent")
        q = np.array([0, 0, 1, 0.3, -0.2, 0.4, 1.2])
        rates = np.array([0, 0, 0, 0.2, -0.3, 0.4, 0.7])
        force = np.asarray(model.generalized_force(q, [0, 0, 0, 0, 1])).ravel()
        self.assertAlmostEqual(force @ rates, 0.7)
        np.testing.assert_allclose(
            np.asarray(
                model.difference_rpy(ca.DM(q[3:6]), ca.DM(q[3:6] + 0.01 * rates[3:6]), 0.01)
            ).ravel(),
            rates[3:6],
        )

    def test_physical_rotor_virtual_power(self):
        from dmcc_model.derivation_uav_manipulator_pos_state_force import DerivationUAV

        model, _ = setup("euler_consistent")
        _, force, _ = DerivationUAV().get_Lagrangian_casadi()
        q = np.array([0, 0, 1, 0.3, -0.2, 0.4, 1.2])
        u = [5, 4.7, 5.2, 4.9, 0]
        rates = np.array([0.2, -0.3, 0.4])
        phi, theta = q[3:5]
        T = np.array(
            [
                [1, 0, -np.sin(theta)],
                [0, np.cos(phi), np.sin(phi) * np.cos(theta)],
                [0, -np.sin(phi), np.cos(phi) * np.cos(theta)],
            ]
        )
        raw = np.asarray(force(q, u)).ravel()
        corrected = np.asarray(model.generalized_force(q, u)).ravel()
        self.assertAlmostEqual(corrected[3:6] @ rates, raw[3:6] @ (T @ rates))

    def test_contact_static_target_has_finite_jacobian(self):
        pos, vel = fixed_target([0, 0, 0.85])
        model = DMCC_UAV_Manipulator(
            has_contact_target=True,
            manipulator_length=0.1,
            montage_offset_b=[0, 0, -0.05],
            inertia_moment=[0.03, 0.04, 0.05],
            trajectory_function=pos,
            vel_trajectory_function=vel,
        )
        refs = np.stack([reference_state([0, 0, 1]), reference_state([1, 0, 1])])
        with contextlib.redirect_stdout(io.StringIO()):
            model.get_path_waypoints(
                reference_state([0, 0, 1]),
                refs,
                avg_vel=1,
                num_waypoints=11,
                i_switch=np.array([3, 10]),
            )
            model.initialization_formulation(
                build_solver=False, dynamics_convention="euler_consistent"
            )
        nlp = model.nlp_def
        jac = ca.Function(
            "static_contact_jac", [nlp["x"], nlp["p"]], [ca.jacobian(nlp["g"], nlp["x"])]
        )
        seed = np.asarray(model.guess_state).ravel().copy()
        seed[1 + 5 * 11 : 1 + 12 * 11] = np.tile(reference_state([0, 0, 1])[:7], 11)
        self.assertTrue(np.isfinite(np.asarray(jac(seed, model.pack_parameters()))).all())
        lbx, ubx, lbg, ubg = build_solver_bounds(model)
        self.assertEqual(len(lbg), nlp["g"].numel())

    def test_heading_is_inactive_for_static_targets_and_rejects_reverse(self):
        error = DMCC_UAV_Manipulator.heading_error
        self.assertEqual(float(error(ca.DM([0, 1, 0]), ca.DM.zeros(3))), 0.0)
        self.assertAlmostEqual(float(error(ca.DM([1, 0, 0]), ca.DM([1, 0, 0]))), 0.0)
        self.assertGreater(float(error(ca.DM([-1, 0, 0]), ca.DM([1, 0, 0]))), 3.9)

    def test_fractional_contact_progress_seed_satisfies_evolution(self):
        pos, vel = fixed_target([0, 0, 0.85])
        model = DMCC_UAV_Manipulator(
            has_contact_target=True,
            manipulator_length=0.1,
            montage_offset_b=[0, 0, -0.05],
            inertia_moment=[0.03, 0.04, 0.05],
            trajectory_function=pos,
            vel_trajectory_function=vel,
        )
        refs = np.stack([reference_state([0, 0, 1]), reference_state([1, 0, 1])])
        with contextlib.redirect_stdout(io.StringIO()):
            model.get_path_waypoints(
                reference_state([0, 0, 1]), refs, num_waypoints=11, i_switch=np.array([3, 10])
            )
            model.initialization_formulation(kappa0=1.5, build_solver=False)
        x = np.asarray(model.guess_state).ravel()
        epsilon = x[1 + 12 * 11 : 1 + 12 * 11 + 10]
        kappa = x[1 + 12 * 11 + 10 : 1 + 12 * 11 + 10 + 11]
        np.testing.assert_allclose(np.diff(kappa) + epsilon, 0, atol=1e-12)
        self.assertEqual(kappa[0], 1.5)
        self.assertEqual(kappa[-1], 0)
        self.assertTrue(np.all((epsilon >= 0) & (epsilon <= 1)))

    def test_small_or_duplicate_grid_is_rejected(self):
        base = DMCC_base()
        start = reference_state([0, 0, 1])
        refs = np.stack([reference_state([1, 0, 1]), reference_state([2, 0, 1])])
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                base.get_path_waypoints(start, refs, num_waypoints=1)
            with self.assertRaises(ValueError):
                base.get_path_waypoints(start, refs, num_waypoints=11, i_switch=np.array([5, 5]))

    def test_full_state_translation_velocity_uses_position_indices(self):
        base = DMCC_base()
        first, second = reference_state([0, 0, 1]), reference_state([0.02, 0.04, 1.06])
        second[7:] = 99
        value = np.asarray(
            base.average_full_state(ca.DM(first), ca.DM(second), 0.1, num_state=7)
        ).ravel()
        np.testing.assert_allclose(value[7:10], [0.2, 0.4, 0.6])

    def test_numeric_rotation_and_legacy_export_are_supported(self):
        base = DMCC_base()
        R = np.asarray(base.rotation_matrix(ca.DM([0.2, 0.3, 0.4])))
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        model, _ = setup()
        model.result_tn = np.array([1.0])
        model.X_state = np.tile(reference_state([0, 0, 1])[:7], (model.num_waypoints, 1))
        model.get_velocity(num_state=7)
        self.assertTrue(np.isfinite(model.velocity_list).all())

    def test_demo_node_count_and_pure_time_objective(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model, _ = build_demo("drone_racing", nodes=31)
        self.assertEqual(model.num_waypoints, 31)
        seed = np.asarray(model.guess_state).ravel().copy()
        seed[1:] += 0.1
        nlp = model.nlp_def
        f = ca.Function("time_only", [nlp["x"], nlp["p"]], [nlp["f"]])
        self.assertEqual(float(f(seed, model.pack_parameters())), seed[0])

    def test_invalid_inertia_and_model_flag_are_explicit(self):
        with self.assertRaises(ValueError):
            DMCC_UAV_Manipulator(has_manipulator=False)
        model = DMCC_UAV_Manipulator()
        with self.assertRaises(ValueError):
            model.initialization_formulation(dynamics_convention="euler_consistent")


if __name__ == "__main__":
    unittest.main()
