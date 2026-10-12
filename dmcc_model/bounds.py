"""Shared bounds for contact and ordered-waypoint branches; explicit units."""

import casadi as ca
import numpy as np


def build_solver_bounds(
    model,
    *,
    velocity_limits=(1.3, 1.3, 1.15, 8.0, 8.0, 2.0, np.pi / 2),
    joint_torque_limit=2.0,
    contact_speed_tolerance=0.01,
    heading_tolerance=0.01,
    contact_distance=0.02,
):
    limits = np.asarray(velocity_limits)
    if limits.shape != (7,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("Need seven positive velocity limits")
    if model.minimum_pass_speed is not None and model.minimum_pass_speed > np.linalg.norm(
        limits[:3]
    ):
        raise ValueError("Pass speed exceeds the available translational velocity limits")
    tolerances = [joint_torque_limit, contact_speed_tolerance, heading_tolerance, contact_distance]
    if not np.isfinite(tolerances).all() or np.any(np.asarray(tolerances) <= 0):
        raise ValueError("Torque limit and contact tolerances must be positive and finite")
    n, m = model.num_waypoints, model.num_def_point
    lbg = [0.0] * model.equal_constraint_length
    ubg = lbg.copy()
    for _ in range(n - 1):
        lbg.extend(-limits)
        ubg.extend(limits)
    if model.has_contact_target:
        for _ in range(n - 1):
            lbg.extend([0.0, 0.0])
            ubg.extend([contact_speed_tolerance, heading_tolerance**2])
        lbg.extend([-0.01] * (n - 1))
        ubg.extend([ca.inf] * (n - 1))
    else:
        for low, high in model.waypoint_inequality_bounds:
            lbg.append(low)
            ubg.append(high)
    lbx, ubx = [0.01], [ca.inf]
    hover = model.total_mass * model.g_ / 4
    for _ in range(n):
        lbx.extend([0.5 * hover] * 4 + [-joint_torque_limit])
        ubx.extend([1.5 * hover] * 4 + [joint_torque_limit])
    for _ in range(n):
        lbx.extend([-np.inf, -np.inf, 0.0, -np.pi / 4, -np.pi / 4, -np.pi, np.pi / 3])
        ubx.extend([np.inf, np.inf, 1.5, np.pi / 4, np.pi / 4, np.pi, 2 * np.pi / 3])
    if model.has_contact_target:
        lbx.extend([0.0] * (n - 1 + n + n - 1))
        ubx.extend([1.0] * (n - 1) + [model.kappa_0] * n + [contact_distance] * (n - 1))
    else:
        lbx.extend([0.0] * (m * n + m * (n - 1)))
        ubx.extend([1.0] * (m * n) + [model.waypoint_radius**2] * (m * (n - 1)))
    assert len(lbx) == len(ubx) == model.nlp_def["x"].numel()
    assert len(lbg) == len(ubg) == model.nlp_def["g"].numel()
    return lbx, ubx, lbg, ubg
