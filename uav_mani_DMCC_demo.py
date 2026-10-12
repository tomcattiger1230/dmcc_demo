#!/usr/bin/env python
"""UAV/manipulator DMCC demos with explicit model conventions and validation.

Original demo by Wei Luo; original task positions and race parameters retained.
"""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import casadi as ca
import numpy as np
from dmcc_model.bounds import build_solver_bounds
from dmcc_model.uav_manipulator_DMCC_3D_discrete_pos_force import DMCC_UAV_Manipulator


def reference_state(position):
    return np.r_[position, [0.0, 0.0, 0.0, np.pi / 2], np.zeros(7)]


def fixed_target(position):
    h, k = ca.SX.sym("h"), ca.SX.sym("k")
    return (
        ca.Function("fixed_position", [h, k], [ca.DM(position)]),
        ca.Function("fixed_velocity", [h, k], [ca.DM.zeros(3)]),
    )


def build_demo(
    task,
    *,
    dynamics_convention="euler_consistent",
    objective_mode="minimum_time",
    nodes=None,
    pass_speed=0.1,
):
    """Keep the original positions and the demo's physically valid race inertia."""
    if task not in {"drone_racing", "fixed_point_contact"}:
        raise ValueError("Unknown task")
    contact = task == "fixed_point_contact"
    positions = (
        [(0.0, 0.0, 0.65), (1.0, 0.0, 0.54), (2.5, 0.0, 0.65)]
        if contact
        else [(0.0, 0.0, 0.55), (0.55, 0.2, 0.4), (0.95, -0.1, 0.5), (2.5, 0.0, 0.55)]
    )
    refs = np.stack([reference_state(p) for p in positions])
    target_functions = fixed_target([1.0, 0.0, 0.4]) if contact else (None, None)
    model = DMCC_UAV_Manipulator(
        mass_quadrotor=1.659,
        has_contact_target=contact,
        has_manipulator=True,
        manipulator_length=0.1 if contact else 0.21,
        montage_offset_b=[0, 0, -0.05 if contact else -0.1],
        # Reuse the admissible inertia already specified by the race example.
        # The invalid historical class defaults remain only in legacy mode.
        inertia_moment=[0.01558, 0.01523, 0.01926],
        manipulator_inertia_moment=[0, 0.0019, 0] if contact else [2.054e-5, 0.0014302, 0.0014264],
        mass_manipulator=0.36 if contact else 0.136,
        g=9.8066 if contact else 9.7,
        trajectory_function=target_functions[0],
        vel_trajectory_function=target_functions[1],
    )
    model.get_path_waypoints(
        refs[0],
        refs[1:],
        avg_vel=1.5 if contact else 0.5,
        waypoint_distance=0.02,
        dt_waypoint=0.06 if contact else 0.1,
        num_waypoints=nodes,
    )
    model.initialization_formulation(
        build_solver=False,
        dynamics_convention=dynamics_convention,
        objective_mode=objective_mode,
        waypoint_radius=0.05,
        minimum_pass_speed=None if contact else pass_speed,
    )
    bounds = build_solver_bounds(
        model,
        velocity_limits=(1.3, 1.3, 1.15, 8, 8, 2, np.pi / 2)
        if contact
        else (0.25, 0.25, 0.15, 8, 8, 2, np.pi / 8),
        joint_torque_limit=2.0 if contact else 1.0,
    )
    return model, bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=["drone_racing", "fixed_point_contact"], default="drone_racing"
    )
    parser.add_argument(
        "--dynamics-convention", choices=["legacy", "euler_consistent"], default="euler_consistent"
    )
    parser.add_argument(
        "--objective", choices=["regularized", "minimum_time"], default="minimum_time"
    )
    parser.add_argument("--nodes", type=int, help="Optional node count")
    parser.add_argument(
        "--pass-speed", type=float, default=0.1, help="Minimum intermediate interval speed [m/s]"
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        model, bounds = build_demo(
            args.task,
            dynamics_convention=args.dynamics_convention,
            objective_mode=args.objective,
            nodes=args.nodes,
            pass_speed=args.pass_speed,
        )
    opts = dict(model.solver_options)
    opts.update({"ipopt.print_level": 0, "ipopt.sb": "yes", "ipopt.tol": 1e-7})
    model.solver = ca.nlpsol("demo_solver", "ipopt", model.nlp_def, opts)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model.get_results(model.pack_parameters(), *bounds)
    elapsed = time.perf_counter() - start
    report = dict(
        task=args.task,
        dynamics_convention=model.dynamics_convention,
        objective_mode=model.objective_mode,
        nodes=model.num_waypoints,
        total_time=float(model.result_tn[0]),
        solve_seconds=elapsed,
        status=model.solver_stats["return_status"],
        success=model.solver_stats["success"],
        max_constraint_violation=model.constraint_violation,
        terminal_velocity_reference=model.Wp_ref[-1, 7:10].tolist(),
        continuous_replay_performed=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_final_result_as_npy(args.output / f"{args.task}.npy")
    (args.output / f"{args.task}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not args.no_plot:
        model.showPath_XYplane()
        model.showPath_XZplane()
        model.showPath_3D()


if __name__ == "__main__":
    main()
