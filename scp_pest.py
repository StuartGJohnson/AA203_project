"""
Lifted/Adapted from AA203: cartpole_swingup_constrained.py: Original banner:
Starter code for the problem "Cart-pole swing-up with limited actuation".
Autonomous Systems Lab (ASL), Stanford University
"""
import copy
import json
from typing import Callable

import numpy as np
from functools import partial
import cvxpy as cvx
import jax
import jax.numpy as jnp
from tqdm import tqdm
import pest_pde
import os
import inspect
import datetime
import io, json, jsons
from dataclasses import dataclass, make_dataclass, asdict
from enum import Enum, StrEnum
from time import perf_counter, sleep
import cv2


class OptimizationObjective(StrEnum):
    Convex = 'Convex'
    Linear = 'Linear'


@dataclass
class SCPEnv:
    P_wt: float = 1e-1
    c_wt: float = 1.0
    c_target: float = 1.0
    c_target_compute: bool = True
    p_wt: float = 10
    p_target: float = 0.0
    w_wt: float = 1e-3
    w_target: float = 0.0
    Q_wt: float = 1e2
    R_wt: float = 1e-1
    rho: float = 0.5
    rho_min: float = 0.01
    u_max: float = 10
    eps: float = .001
    eps_spatial: float = 0.01
    max_iters: int = 100
    max_iters_spatial: int = 4
    solver: cvx.settings = cvx.ECOS
    objective: OptimizationObjective = OptimizationObjective.Convex
    verbose_solver: bool = False
    beta: float = 2.0
    eta_1: float = 0.5
    eta_2: float = 0.99
    n_spatial_init: int = 5
    n_spatial_inc: int = 5
    n_spatial_fac: int = 0

    def __post_init__(self):
        if self.solver == 'ECOS':
            self.solver = cvx.ECOS
        if self.solver == 'MOSEK':
            self.solver = cvx.MOSEK
        if self.solver == 'OSQP':
            self.solver = cvx.OSQP
        if self.objective == 'Convex':
            self.objective = OptimizationObjective.Convex
        if self.objective == 'Linear':
            self.objective = OptimizationObjective.Linear


def n_spatial_vec(se: SCPEnv) -> list[int]:
    n_vec = []
    for i in range(0, se.max_iters_spatial):
        n_vec.append(n_spatial(se, i))
    return n_vec


def n_spatial(se: SCPEnv, i: int) -> int:
    """
    Given a refinement iteration #, return the multiscale refinement grid n.
    Args:
        se:
        i:

    Returns:

    """
    if se.n_spatial_inc != 0 and se.n_spatial_fac != 0:
        # one must be zero
        raise ValueError("Bad arguments for spatial refinement in SCPEnv!")
    n = (se.n_spatial_inc > 0) * (i * se.n_spatial_inc + se.n_spatial_init) + \
        (se.n_spatial_fac > 0) * (se.n_spatial_fac ** i * se.n_spatial_init)
    return n


def regrid_solution(e: pest_pde.Env, s: np.ndarray, u: np.ndarray, n_new: int) -> (pest_pde.Env, np.ndarray, np.ndarray):
    """
    regrid - at a new grid resolution - a solution of the pde via interpolation.
    Args:
        e:
        s:
        u:
        n_new:

    Returns:
        updated pest_pde.Env
        regridded s
        regridded u
    """
    e_out = copy.deepcopy(e)
    e_out.n = n_new
    s_out = np.zeros((s.shape[0], 3*n_new**2), dtype=s.dtype)
    if e.u_mode == pest_pde.ControlMode.Spot:
        u_out = np.zeros((u.shape[0], n_new**2), dtype=u.dtype)
    else:
        u_out = np.copy(u)
    c, p, w = np.split(s, 3, axis=1)
    for t in range(0, c.shape[0]):
        # reshape time slice to 2d
        c2 = np.reshape(c[t], (e.n, e.n))
        p2 = np.reshape(p[t], (e.n, e.n))
        w2 = np.reshape(w[t], (e.n, e.n))
        # interpolate
        c3 = cv2.resize(c2, (e_out.n, e_out.n), interpolation=cv2.INTER_LINEAR)
        p3 = cv2.resize(p2, (e_out.n, e_out.n), interpolation=cv2.INTER_LINEAR)
        w3 = cv2.resize(w2, (e_out.n, e_out.n), interpolation=cv2.INTER_LINEAR)
        # flatten
        c4 = np.reshape(c3, (e_out.n*e_out.n,))
        p4 = np.reshape(p3, (e_out.n*e_out.n,))
        w4 = np.reshape(w3, (e_out.n*e_out.n,))
        # repackage
        s_out[t] = np.concatenate([c4, p4, w4], axis=0)
    # it is possible u is one shorter in t than s, so do this separately
    # also, this does not deal with an underactuated u - that is
    # with less spatial resolution than s, but not constant
    # at the time of writing this, most if not all other code does
    # not deal with an underactuated u either.
    if e.u_mode == pest_pde.ControlMode.Spot:
        for t in range(0, u.shape[0]):
            # reshape time slice to 2d
            u2 = np.reshape(u[t], (e.n, e.n))
            # interpolate
            u3 = cv2.resize(u2, (e_out.n, e_out.n), interpolation=cv2.INTER_LINEAR)
            # flatten
            u4 = np.reshape(u3, (e_out.n*e_out.n,))
            # repackage
            u_out[t] = u4
    return e_out, s_out, u_out

@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def affinize(f, s, u):
    """Affinize the function `f(s, u)` around `(s, u)`.

    Arguments
    ---------
    f : callable
        A nonlinear function with call signature `f(s, u)`.
    s : numpy.ndarray
        The state (1-D).
    u : numpy.ndarray
        The control input (1-D).

    Returns
    -------
    A : jax.numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `s`.
    B : jax.numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `u`.
    c : jax.numpy.ndarray
        The offset term in the first-order Taylor expansion of `f` at `(s, u)`
        that sums all vector terms strictly dependent on the nominal point
        `(s, u)` alone.
    """
    A, B = jax.jacobian(f, (0, 1))(s, u)
    # note this assumes we correct for s and u when using A,B and c !
    c = f(s, u)

    return A, B, c


def discretize(f, dt):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""
    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return integrator


def solve_scp(se: SCPEnv, f, s0, s_goal, N, P, Q, R, u_max, rho, eps, max_iters):
    """Solve via SCP.

    Arguments
    ---------
    se: SCPEnv
        settings and parameters of SCP run.
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    N : int
        The time horizon of the LQR cost function.
    P : numpy.ndarray
        The terminal state cost matrix (2-D).
    Q : numpy.ndarray
        The state stage cost matrix (2-D).
    R : numpy.ndarray
        The control stage cost matrix (2-D).
    u_max : float
        The bound defining the control set `[-u_max, u_max]`.
    rho : float
        Trust region radius.
    eps : float
        Termination threshold for SCP.
    max_iters : int
        Maximum number of SCP iterations.

    Returns
    -------
    s : numpy.ndarray
        A 2-D array where `s[k]` is the system state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u : numpy.ndarray
        A 2-D array where `u[k]` is the system control at time step `k`,
        for `k = 0, 1, ..., N-1`
    J : numpy.ndarray
        A 1-D array where `J[i]` is the SCP sub-problem cost after the i-th
        iteration, for `i = 0, 1, ..., (iteration when convergence occurred)`
    """
    n = Q.shape[0]  # state dimension
    if np.isscalar(R):
        m = 1
    else:
        m = R.shape[0]  # control dimension

    # Initialize dynamically feasible nominal trajectories
    u = np.zeros((N, m))
    s = np.zeros((N + 1, n))
    s[0] = s0
    for k in range(N):
        s[k + 1] = f(s[k], u[k])

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    J = np.zeros(max_iters + 1)
    # initialize J to the value of the objective at the feasible initial guess

    J[0] = np.inf
    for i in (prog_bar := tqdm(range(max_iters))):
        rho_current = rho
        s, u, J[i + 1] = scp_iteration(se, f, s0, s_goal, s, u, N, P, Q, R, u_max, rho_current)
        dJ = np.abs(J[i + 1] - J[i])
        prog_bar.set_postfix({"objective change": "{:.5f}".format(dJ)})
        if dJ < eps:
            converged = True
            print("SCP converged after {} iterations.".format(i))
            break
    if not converged:
        raise RuntimeError("SCP did not converge!")
    J = J[1 : i + 1]
    return s, u, J


def solve_scp_plus(se: SCPEnv, f, s0, s_goal, N, P, Q, R, u_max,
                   rho_init, eps, max_iters, s_init: np.ndarray = None, u_init: np.ndarray = None):
    """Solve via SCP. Use adaptive trust region.

    Arguments
    ---------
    se: SCPEnv
        settings and parameters of SCP run.
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    N : int
        The time horizon of the LQR cost function.
    P : numpy.ndarray
        The terminal state cost matrix (2-D).
    Q : numpy.ndarray
        The state stage cost matrix (2-D).
    R : numpy.ndarray
        The control stage cost matrix (2-D).
    u_max : float
        The bound defining the control set `[-u_max, u_max]`.
    rho : float
        Trust region radius.
    eps : float
        Termination threshold for SCP.
    max_iters : int
        Maximum number of SCP iterations.
    s_init: numpy.ndarray
        state from external source (e.g;. previous SCP iterations)
    u_init: numpy.ndarray
        control from external source (e.g;. previous SCP iterations)

    Returns
    -------
    s : numpy.ndarray
        A 2-D array where `s[k]` is the system state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u : numpy.ndarray
        A 2-D array where `u[k]` is the system control at time step `k`,
        for `k = 0, 1, ..., N-1`
    J : numpy.ndarray
        A 1-D array where `J[i]` is the SCP sub-problem cost after the i-th
        iteration, for `i = 0, 1, ..., (iteration when convergence occurred)`

    """
    n = Q.shape[0]  # state dimension
    if np.isscalar(R):
        m = 1
    else:
        m = R.shape[0]  # control dimension

    if u_init is None:
        # Initialize dynamically feasible nominal trajectories
        u = np.zeros((N, m))
    else:
        u = u_init
    if s_init is None:
        # Initialize dynamically feasible nominal trajectories
        s = integrate_dynamics(N, m, n, s0, u, f)
    else:
        s = s_init

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    # the linearized approximation ... leading to J
    J = np.zeros((max_iters + 1,), dtype=float)
    # the exact J
    J_ref = np.zeros((max_iters + 1,), dtype=float)
    # relative error in J
    dJ_rel = np.ones((max_iters + 1,), dtype=float) * np.nan
    # the trust region (this is adaptive)
    rho = np.zeros((max_iters + 1,), dtype=float)
    # the linearity ratio
    lin_ratio = np.ones((max_iters + 1,), dtype=float) * np.nan
    # wall time of compute
    scp_time = np.zeros(max_iters + 1)
    rho[0] = rho_init
    # initialize all J to the value of the objective at the initial values
    J_ref[0] = objective_eval(s, u, s_goal, N, P, Q, R)
    J[0] = J_ref[0]
    # setup up clock
    rho_current = rho[0]
    J_ref_current = J_ref[0]
    J_current = J[0]
    for i in (prog_bar := tqdm(range(max_iters))):
        start_time = perf_counter()
        s_prop, u_prop, J_prop = scp_iteration(se, f, s0, s_goal, s, u, N, P, Q, R, u_max, rho_current)
        s_ref_prop = integrate_dynamics(N, m, n, s0, u_prop, f)
        J_ref_prop = objective_eval(s_ref_prop, u_prop, s_goal, N, P, Q, R)
        J_prop = objective_eval(s_prop, u_prop, s_goal, N, P, Q, R)
        dJ_linear = J_ref_current - J_prop
        dJ_ref = J_ref_current - J_ref_prop
        r = dJ_ref / dJ_linear
        good_update = False
        if r >= se.eta_1:
            # accept update
            s = s_prop
            u = u_prop
            J_ref_current = J_ref_prop
            J_current = J_prop
            good_update = True
        # conf. region updates:
        if r > se.eta_2:
            # expand conf. region
            rho_current = max(rho_current * se.beta, se.rho_min)
        elif r < se.eta_1:
            # contract conf. region
            rho_current = max(rho_current / se.beta, se.rho_min)
        # update fields
        J_ref[i+1] = J_ref_current
        J[i+1] = J_current
        scp_time[i+1] = perf_counter() - start_time
        lin_ratio[i+1] = r
        rho[i+1] = rho_current
        current_error = np.abs(dJ_ref/J_ref_current)
        dJ_rel[i+1] = current_error
        prog_bar.set_postfix(
            {"current rho": "{:.5f}".format(rho_current),
             "objective change": "{:.5f}".format(dJ_linear),
             "r": "{:.5f}".format(r)})
        if current_error < eps and good_update:
            converged = True
            print("SCP converged after {} iterations.".format(i))
            break
    if not converged:
        # todo: could this also be used for trust region adjustment?
        raise RuntimeError("SCP did not converge!")
    J = J[0: i + 2]
    J_ref = J_ref[0: i + 2]
    dJ_rel = dJ_rel[0: i + 2]
    rho = rho[0: i + 2]
    scp_time = scp_time[0: i + 2]
    lin_ratio = lin_ratio[0: i + 2]
    return s, u, J, J_ref, dJ_rel, lin_ratio, rho, scp_time


def objective_eval(s: np.ndarray, u: np.ndarray, s_goal: np.ndarray, N: int, P: np.ndarray, Q: np.ndarray, R):
    """compute the value of the objective function. Note this should follow the definition in the cvxpy solver."""
    # for large state spaces, the matmuls below trigger a bug in numpy. I have flattened these with
    # element-wise multiplies. Too bad I can figure out how to do this in cvxpy. One solution might be to
    # split the problem up into 3 state spaces (c, p, and w).
    p_diag = np.diagonal(P)
    q_diag = np.diagonal(Q)
    s_diff = s[N] - s_goal
    #p_factor = s_diff.T @ P @ s_diff
    p_factor = np.sum(s_diff * p_diag * s_diff)
    q_factor = 0
    u_factor = 0
    if np.isscalar(R):
        R = np.eye(1) * R
        r_diag = R
    else:
        r_diag = np.diagonal(R)
    for i in range(0, N):
        s_diff = s[i] - s_goal
        #q_factor += s_diff.T @ Q @ s_diff
        q_factor += np.sum(s_diff * q_diag * s_diff)
        u_i = u[i]
        #u_factor += u_i.T @ R @ u_i
        u_factor += np.sum(u_i * r_diag * u_i)
    return p_factor + u_factor + q_factor + u_factor

def integrate_dynamics(N: int, m: int, n:int, s0:np.ndarray, u:np.ndarray, f:Callable):
    """ Integrate full accuracy dynamics.

    Args:
        N:
        m:
        n:
        s0:
        u:
        f:

    Returns:

    """
    s = np.zeros((N + 1, n))
    s[0] = s0
    for k in range(N):
        s[k + 1] = f(s[k], u[k])
    return s


def scp_iteration(se: SCPEnv, f, s0, s_goal, s_prev, u_prev, N, P, Q, R, u_max, rho):
    """Solve a single SCP sub-problem.

    Arguments
    ---------
    se: SCPEnv
        Settings and parameters of SCP run.
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    s_prev : numpy.ndarray
        The state trajectory around which the problem is convexified (2-D).
    u_prev : numpy.ndarray
        The control trajectory around which the problem is convexified (2-D).
    N : int
        The time horizon of the LQR cost function.
    P : numpy.ndarray
        The terminal state cost matrix (2-D).
    Q : numpy.ndarray
        The state stage cost matrix (2-D).
    R : numpy.ndarray
        The control stage cost matrix (2-D).
    u_max : float
        The bound defining the control set `[-u_max, u_max]`.
    rho : float
        Trust region radius.

    Returns
    -------
    s : numpy.ndarray
        A 2-D array where `s[k]` is the system state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u : numpy.ndarray
        A 2-D array where `u[k]` is the system control at time step `k`,
        for `k = 0, 1, ..., N-1`
    J : float
        The SCP sub-problem cost.
    """
    A, B, c = affinize(f, s_prev[:-1], u_prev)
    A, B, c = np.array(A), np.array(B), np.array(c)
    n = Q.shape[0]
    if np.isscalar(R):
        m = 1
        R = np.eye(1) * R
    else:
        m = R.shape[0]  # control dimension
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))
    # cvxpy doesn't seem to like big matrices in the quad_forms below, but I
    # am unable to do the diagonal vector product sum in cvxpy (see objective below).
    # p_diag = np.diagonal(P)
    # q_diag = np.diagonal(Q)
    # r_diag = np.diagonal(R)
    # Construct the convex SCP sub-problem for cvxpy.
    if se.objective == OptimizationObjective.Convex:
        objective = (cvx.quad_form((s_cvx[N] - s_goal), P) +
                     cvx.sum([cvx.quad_form(s_cvx[i1] - s_goal, Q) + cvx.quad_form(u_cvx[i1], R) for i1 in range(N)]))
        # objective = (cvx.sum(cvx.multiply(cvx.multiply(s_cvx[N] - s_goal, p_diag), s_cvx[N] - s_goal)) +
        #              cvx.sum([cvx.sum(cvx.multiply(cvx.multiply(s_cvx[i1] - s_goal, q_diag), s_cvx[i1] - s_goal)) +
        #                       cvx.sum(cvx.multiply(cvx.multiply(u_cvx[i1], r_diag), u_cvx[i1])) for i1 in range(N)]))
        # objective = (cvx.quad_form((s_cvx[N] - s_goal), p_diag) +
        #              cvx.sum([cvx.quad_form(s_cvx[i1] - s_goal, q_diag) + cvx.quad_form(u_cvx[i1], r_diag) for i1 in range(N)]))
    elif se.objective == OptimizationObjective.Linear:
        # todo: this does not seem to work well.
        objective = (cvx.norm(P@(s_cvx[N] - s_goal), 1) +
                     cvx.sum([cvx.norm(Q@(s_cvx[i1] - s_goal), 1) + cvx.norm(R @ u_cvx[i1], 1) for i1 in range(N)]))
    else:
        raise RuntimeError("SCP objective settings incorrect!")
    # dynamics
    constraints = [s_cvx[i2 + 1] == c[i2] + A[i2] @ (s_cvx[i2] - s_prev[i2]) +
                   B[i2] @ (u_cvx[i2] - u_prev[i2]) for i2 in range(N)]
    # initial state
    constraints += [s_cvx[0] == s0]
    # control bounds
    constraints += [cvx.min(u_cvx) >= 0]
    constraints += [cvx.max(u_cvx) <= u_max]
    # trust region
    constraints += [cvx.max(cvx.abs(s_cvx - s_prev)) <= rho]
    constraints += [cvx.max(cvx.abs(u_cvx - u_prev)) <= rho]
    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve(se.solver, verbose=se.verbose_solver)
    if prob.status != "optimal":
        raise RuntimeError("SCP solve failed. Problem status: " + prob.status)
    s = s_cvx.value
    u = u_cvx.value
    J = prob.objective.value
    return s, u, J


def scp_iteration2(se: SCPEnv, f, s0, s_goal, s_prev, u_prev, N, P, Q, R, u_max, rho):
    """Solve a single SCP sub-problem.

    Arguments
    ---------
    se: SCPEnv
        Settings and parameters of SCP run.
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    s_prev : numpy.ndarray
        The state trajectory around which the problem is convexified (2-D).
    u_prev : numpy.ndarray
        The control trajectory around which the problem is convexified (2-D).
    N : int
        The time horizon of the LQR cost function.
    P : numpy.ndarray
        The terminal state cost matrix (2-D).
    Q : numpy.ndarray
        The state stage cost matrix (2-D).
    R : numpy.ndarray
        The control stage cost matrix (2-D).
    u_max : float
        The bound defining the control set `[-u_max, u_max]`.
    rho : float
        Trust region radius.

    Returns
    -------
    s : numpy.ndarray
        A 2-D array where `s[k]` is the system state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u : numpy.ndarray
        A 2-D array where `u[k]` is the system control at time step `k`,
        for `k = 0, 1, ..., N-1`
    J : float
        The SCP sub-problem cost.
    """
    A, B, c = affinize(f, s_prev[:-1], u_prev)
    A, B, c = np.array(A), np.array(B), np.array(c)
    n = Q.shape[0]
    if np.isscalar(R):
        m = 1
        R = np.eye(1) * R
    else:
        m = R.shape[0]  # control dimension
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))
    # Construct the convex SCP sub-problem for cvxpy.
    if se.objective == OptimizationObjective.Convex:
        objective = (cvx.quad_form((s_cvx[N] + s_prev[N] - s_goal), P) +
                     cvx.sum([cvx.quad_form(s_cvx[i1] + s_prev[i1] - s_goal, Q) + cvx.quad_form(u_cvx[i1] + u_prev[i1], R) for i1 in range(N)]))
    elif se.objective == OptimizationObjective.Linear:
        # todo: this does not seem to work well.
        objective = (cvx.norm(P@(s_cvx[N] - s_goal), 1) +
                     cvx.sum([cvx.norm(Q@(s_cvx[i1] - s_goal), 1) + cvx.norm(R @ u_cvx[i1], 1) for i1 in range(N)]))
    else:
        raise RuntimeError("SCP objective settings incorrect!")
    # dynamics
    constraints = [s_cvx[i2 + 1] == c[i2] - s_prev[i2] + A[i2] @ s_cvx[i2] +
                   B[i2] @ u_cvx[i2] for i2 in range(N)]
    # initial state
    constraints += [s_cvx[0] == s0 - s_prev[0]]
    # control bounds
    constraints += [cvx.min(u_cvx + u_prev) >= 0]
    constraints += [cvx.max(u_cvx + u_prev) <= u_max]
    # trust region
    constraints += [cvx.max(cvx.abs(s_cvx)) <= rho]
    constraints += [cvx.max(cvx.abs(u_cvx)) <= rho]
    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve(se.solver, verbose=se.verbose_solver)
    if prob.status != "optimal":
        raise RuntimeError("SCP solve failed. Problem status: " + prob.status)
    s = s_cvx.value + s_prev
    u = u_cvx.value + u_prev
    J = prob.objective.value
    return s, u, J


def serialize_scp_run(s: np.ndarray, u: np.ndarray, J: np.ndarray, sim: pest_pde.PestSim, scp_env: SCPEnv) -> str:
    sleep(1.1)  # I only have HHMMSS, so ...
    now = datetime.datetime.now()
    rdir = 'scp_' + now.strftime('%y%m%d-%H%M%S')
    os.makedirs(rdir, exist_ok=True)
    np.save(os.path.join(rdir, 'pest_s.npy'), s)
    np.save(os.path.join(rdir, 'pest_J.npy'), J)
    if sim.e.u_mode == pest_pde.ControlMode.Aerial:
        u_save = sim.u_pattern * u
        np.save(os.path.join(rdir, 'pest_u.npy'), u_save)
    else:
        np.save(os.path.join(rdir, 'pest_u.npy'), u)

    # serialize env
    file_env = os.path.join(rdir, 'pest_pde.env.json')
    with io.open(file_env, 'w', encoding='utf-8') as f:
        f.write(json.dumps(jsons.dump(sim.e), ensure_ascii=False))

    # serialize scp env
    file_env = os.path.join(rdir, 'scp.env.json')
    with io.open(file_env, 'w', encoding='utf-8') as f:
        f.write(json.dumps(jsons.dump(scp_env), ensure_ascii=False))

    # serialize do_scp - how loss is encoded is important
    file_do = os.path.join(rdir, 'do_scp.txt')
    with io.open(file_do, 'w', encoding='utf-8') as f:
        f.write(inspect.getsource(do_scp))

    return rdir


def serialize_scp_run_plus(s: np.ndarray, u: np.ndarray, J: np.ndarray, J_ref: np.ndarray, dJ_rel: np.ndarray,
                           lin_ratio: np.ndarray,
                           rho: np.ndarray, scp_time: np.ndarray, sim: pest_pde.PestSim, scp_env: SCPEnv,
                           n_spatial: np.ndarray = None, iter_count: np.ndarray = None,
                           outer_rel_error: np.ndarray = None) -> str:
    sleep(1.1) # I only have HHMMSS
    now = datetime.datetime.now()
    rdir = 'scp_' + now.strftime('%y%m%d-%H%M%S')
    os.makedirs(rdir, exist_ok=True)
    np.save(os.path.join(rdir, 'pest_s.npy'), s)
    np.save(os.path.join(rdir, 'pest_J.npy'), J)
    np.save(os.path.join(rdir, 'pest_rho.npy'), rho)
    np.save(os.path.join(rdir, 'pest_scp_time.npy'), scp_time)
    np.save(os.path.join(rdir, 'pest_J_ref.npy'), J_ref)
    np.save(os.path.join(rdir, 'pest_dJ_rel.npy'), dJ_rel)
    np.save(os.path.join(rdir, 'pest_lin_ratio.npy'), lin_ratio)
    if n_spatial is not None:
        np.save(os.path.join(rdir, 'n_spatial.npy'), n_spatial)
    if iter_count is not None:
        np.save(os.path.join(rdir, 'iter_count.npy'), iter_count)
        np.savetxt(os.path.join(rdir, 'iter_count.txt'), iter_count)
    if outer_rel_error is not None:
        np.save(os.path.join(rdir, 'outer_rel_error.npy'), outer_rel_error)
        np.savetxt(os.path.join(rdir, 'outer_rel_error.txt'), outer_rel_error)
    if sim.e.u_mode == pest_pde.ControlMode.Aerial:
        u_save = sim.u_pattern * u
        np.save(os.path.join(rdir, 'pest_u.npy'), u_save)
    else:
        np.save(os.path.join(rdir, 'pest_u.npy'), u)

    # serialize env
    file_env = os.path.join(rdir, 'pest_pde.env.json')
    with io.open(file_env, 'w', encoding='utf-8') as f:
        f.write(json.dumps(jsons.dump(sim.e), ensure_ascii=False))

    # serialize scp env
    file_env = os.path.join(rdir, 'scp.env.json')
    with io.open(file_env, 'w', encoding='utf-8') as f:
        f.write(json.dumps(jsons.dump(scp_env), ensure_ascii=False))

    # serialize do_scp - how loss is encoded is important
    file_do = os.path.join(rdir, 'do_scp.txt')
    with io.open(file_do, 'w', encoding='utf-8') as f:
        f.write(inspect.getsource(do_scp))

    return rdir


def deserialize_scp(rdir: str):
    s, u, env = pest_pde.deserialize_sim(rdir)
    J = np.load(os.path.join(rdir, 'pest_J.npy'))
    J_ref = np.load(os.path.join(rdir, 'pest_J_ref.npy'))
    lin_ratio = np.load(os.path.join(rdir, 'pest_lin_ratio.npy'))
    rho = np.load(os.path.join(rdir, 'pest_rho.npy'))
    scp_time = np.load(os.path.join(rdir, 'pest_scp_time.npy'))
    # env
    file_env = os.path.join(rdir, 'scp.env.json')
    with io.open(file_env, 'r', encoding='utf-8') as f:
        env_dict = json.load(f)
    scp_env = SCPEnv(**env_dict)
    if os.path.exists(os.path.join(rdir, 'n_spatial.npy')):
        n_spatial = np.load(os.path.join(rdir, 'n_spatial.npy'))
    else:
        n_spatial = None
    if os.path.exists(os.path.join(rdir, 'iter_count.npy')):
        iter_count = np.load(os.path.join(rdir, 'iter_count.npy'))
    else:
        iter_count = None
    if os.path.exists(os.path.join(rdir, 'pest_dJ_rel.npy')):
        dJ_rel = np.load(os.path.join(rdir, 'pest_dJ_rel.npy'))
    else:
        dJ_rel = None
    return s, u, env, J, J_ref, dJ_rel, lin_ratio, rho, scp_time, scp_env, n_spatial, iter_count


def do_scp(pp_env: pest_pde.Env, scp_env: SCPEnv) -> str:
    # Define constants
    #pp_env = pest_pde.Env()
    n_s = pp_env.n**2
    n = 3*n_s  # state dimension
    if pp_env.u_mode == pest_pde.ControlMode.Aerial:
        m = 1 # control dimension
    else:
        m = n_s # control dimension
    s0, u0 = pest_pde.init_state(pp_env)
    dt = pp_env.dt  # discrete time resolution
    T = pest_pde.get_T(pp_env)  # total simulation time
    # we want crops at what we can expect at time t
    # we want pest at 0
    # we want pesticide at 0
    if scp_env.c_target_compute:
        crop_target = pest_pde.crop_function(pp_env, T)
    else:
        crop_target = scp_env.c_target
    print('crop_target: ' + str(crop_target))
    # scale all by FD cell area. note we all scale by area below
    h = pp_env.D / (pp_env.n - 1)
    h2 = h * h
    # hmm, should we weight by this, or...
    #area_mat = pest_pde.build_area_matrix(pp_env)
    s_goal = np.concatenate([crop_target * np.ones((n_s,)),
                             scp_env.p_target*np.ones((n_s,)),
                             scp_env.w_target*np.ones((n_s,))])  # desired field state
    goal_weights = np.concatenate([scp_env.c_wt*np.ones((n_s,)), scp_env.p_wt*np.ones((n_s,)), scp_env.w_wt*np.ones((n_s,))])
    P = h2 * scp_env.P_wt * np.diag(goal_weights)
    #P = 1e2 * np.eye(n)  # terminal state cost matrix
    Q = h2 * scp_env.Q_wt * np.diag(goal_weights)
    #Q = 1e3 * np.eye(n) # state cost matrix
    #Q = 0.0 * np.eye(n) # state cost matrix
    if pp_env.u_mode == pest_pde.ControlMode.Aerial:
        # note we need to adjust this by the length of
        # the u_pattern used in the pde. I am assuming
        # it is a constant here!
        R = h2 * scp_env.R_wt * pp_env.n**2 # control cost
    else:
        R = h2 * scp_env.R_wt * np.eye(m)  # control cost matrix
    rho = scp_env.rho  # trust region parameter
    u_max = scp_env.u_max  # control effort bound
    eps = scp_env.eps  # convergence tolerance
    max_iters = scp_env.max_iters  # maximum number of SCP iterations

    pest = pest_pde.PestSim(pp_env)
    # Initialize the discrete-time dynamics
    if pp_env.u_mode == pest_pde.ControlMode.Spot:
        fd = jax.jit(discretize(pest.pests_wrapper_su_jax, dt))
    else:
        fd = jax.jit(discretize(pest.pests_wrapper_aerial_su_jax, dt))

    t = np.arange(0.0, T + dt, dt)
    N = t.size - 1
    s, u, J = solve_scp(scp_env, fd, s0, s_goal, N, P, Q, R, u_max, rho, eps, max_iters)

    rdir = serialize_scp_run(s, u, J, pest, scp_env)
    return rdir


def do_scp_plus(pp_env: pest_pde.Env, scp_env: SCPEnv) -> str:
    # Define constants
    #pp_env = pest_pde.Env()
    n_s = pp_env.n**2
    n = 3*n_s  # state dimension
    if pp_env.u_mode == pest_pde.ControlMode.Aerial:
        m = 1 # control dimension
    else:
        m = n_s # control dimension
    s0, u0 = pest_pde.init_state(pp_env)
    dt = pp_env.dt  # discrete time resolution
    T = pest_pde.get_T(pp_env)  # total simulation time
    # we want crops at what we can expect at time t
    # we want pest at 0
    # we want pesticide at 0
    if scp_env.c_target_compute:
        crop_target = pest_pde.crop_function(pp_env, T)
    else:
        crop_target = scp_env.c_target
    print('crop_target: ' + str(crop_target))
    # scale all by FD cell area. note we all scale by area below
    h = pp_env.D / (pp_env.n - 1)
    h2 = h * h
    # hmm, should we weight by this, or...
    area_mat = pest_pde.build_area_matrix(pp_env)
    s_goal = np.concatenate([crop_target * np.ones((n_s,)),
                             scp_env.p_target*np.ones((n_s,)),
                             scp_env.w_target*np.ones((n_s,))])  # desired field state
    #goal_weights = np.concatenate([scp_env.c_wt*np.ones((n_s,)), scp_env.p_wt*np.ones((n_s,)), scp_env.w_wt*np.ones((n_s,))])
    goal_weights = np.concatenate(
        [scp_env.c_wt * area_mat, scp_env.p_wt * area_mat, scp_env.w_wt * area_mat])
    P = h2 * scp_env.P_wt * np.diag(goal_weights)
    #P = 1e2 * np.eye(n)  # terminal state cost matrix
    Q = h2 * scp_env.Q_wt * np.diag(goal_weights)
    #Q = 1e3 * np.eye(n) # state cost matrix
    #Q = 0.0 * np.eye(n) # state cost matrix
    if pp_env.u_mode == pest_pde.ControlMode.Aerial:
        # note we need to adjust this by the length of
        # the u_pattern used in the pde. I am assuming
        # it is a constant here!
        R = h2 * scp_env.R_wt * pp_env.n**2 # control cost
    else:
        R = h2 * scp_env.R_wt * np.eye(m)  # control cost matrix
    rho = scp_env.rho  # trust region parameter
    u_max = scp_env.u_max  # control effort bound
    eps = scp_env.eps  # convergence tolerance
    max_iters = scp_env.max_iters  # maximum number of SCP iterations

    pest = pest_pde.PestSim(pp_env)
    # Initialize the discrete-time dynamics
    if pp_env.u_mode == pest_pde.ControlMode.Spot:
        fd = jax.jit(discretize(pest.pests_wrapper_su_jax, dt))
    else:
        fd = jax.jit(discretize(pest.pests_wrapper_aerial_su_jax, dt))

    t = np.arange(0.0, T + dt, dt)
    N = t.size - 1
    s, u, J, J_ref, dJ_rel, lin_ratio, rho, scp_time = solve_scp_plus(scp_env, fd, s0, s_goal, N, P, Q, R, u_max, rho, eps, max_iters)

    rdir = serialize_scp_run_plus(s, u, J, J_ref, dJ_rel, lin_ratio, rho, scp_time, pest, scp_env)
    return rdir


def do_scp_plus_plus(pp_env_ref: pest_pde.Env, scp_env: SCPEnv) -> str:
    """ Perform SCP using spatial refinement (outer loop) and adaptive trust region (inner loop). """
    # obtain the grid refinement sequence from scp_env
    n_grid = n_spatial_vec(scp_env)

    first_pass = True
    pp_env = copy.deepcopy(pp_env_ref)
    u_init = None
    s_init = None
    J_ref_outer = []
    n_accum = []
    iter_accum = []
    outer_rel_error = []
    for ng in n_grid:
        print(f"starting outer iteration:  n = {ng}")
        if first_pass:
            # just set grid size
            pp_env.n = ng
        else:
            # set grid size and interpolate previous solution
            pp_env, s_init, u_init = regrid_solution(pp_env, s_current, u_current, ng)
        # Define constants
        n_s = pp_env.n**2
        n = 3*n_s  # state dimension
        if pp_env.u_mode == pest_pde.ControlMode.Aerial:
            m = 1 # control dimension
        else:
            m = n_s # control dimension
        s0, u0 = pest_pde.init_state(pp_env)
        dt = pp_env.dt  # discrete time resolution
        T = pest_pde.get_T(pp_env)  # total simulation time
        # we want crops at what we can expect at time t
        # we want pest at 0
        # we want pesticide at 0
        if scp_env.c_target_compute:
            crop_target = pest_pde.crop_function(pp_env, T)
        else:
            crop_target = scp_env.c_target
        #print('crop_target: ' + str(crop_target))
        # scale all by FD cell area. note we all scale by area below
        h = pp_env.D / (pp_env.n - 1)
        h2 = h * h
        # hmm, should we weight by this, or...
        area_mat = pest_pde.build_area_matrix(pp_env)
        s_goal = np.concatenate([crop_target * np.ones((n_s,)),
                                 scp_env.p_target*np.ones((n_s,)),
                                 scp_env.w_target*np.ones((n_s,))])  # desired field state
        #goal_weights = np.concatenate([scp_env.c_wt*np.ones((n_s,)), scp_env.p_wt*np.ones((n_s,)), scp_env.w_wt*np.ones((n_s,))])
        goal_weights = np.concatenate(
            [scp_env.c_wt * area_mat, scp_env.p_wt * area_mat, scp_env.w_wt * area_mat])
        P = h2 * scp_env.P_wt * np.diag(goal_weights)
        #P = 1e2 * np.eye(n)  # terminal state cost matrix
        Q = h2 * scp_env.Q_wt * np.diag(goal_weights)
        #Q = 1e3 * np.eye(n) # state cost matrix
        #Q = 0.0 * np.eye(n) # state cost matrix
        if pp_env.u_mode == pest_pde.ControlMode.Aerial:
            # note we need to adjust this by the length of
            # the u_pattern used in the pde. I am assuming
            # it is a constant here!
            R = h2 * scp_env.R_wt * pp_env.n**2 # control cost
        else:
            R = h2 * scp_env.R_wt * np.eye(m)  # control cost matrix
        rho_current = scp_env.rho  # trust region parameter
        u_max = scp_env.u_max  # control effort bound
        eps = scp_env.eps  # convergence tolerance
        max_iters = scp_env.max_iters  # maximum number of SCP iterations

        pest = pest_pde.PestSim(pp_env)
        # Initialize the discrete-time dynamics
        if pp_env.u_mode == pest_pde.ControlMode.Spot:
            fd = jax.jit(discretize(pest.pests_wrapper_su_jax, dt))
        else:
            fd = jax.jit(discretize(pest.pests_wrapper_aerial_su_jax, dt))

        t = np.arange(0.0, T + dt, dt)
        N = t.size - 1

        s_current, u_current, J, J_ref, dJ_rel, lin_ratio, rho, scp_time = solve_scp_plus(scp_env, fd, s0, s_goal,
                                                                  N, P, Q, R, u_max, rho_current, eps, max_iters,
                                                                  None, u_init)

        # check convergence - for now we'll just use the same number as for the internal solve
        if first_pass:
            j_ref_current = J_ref[0]

        # current best J is the last value returned by the internal solver
        dj_ref = np.abs(J_ref[-1] - j_ref_current)
        dj_ref_rel = np.abs(dj_ref)/J_ref[-1]
        j_ref_current = J_ref[-1]

        # accumulate arrays - let's flatten these, but include an index
        # to the first element of each array for each spatial n
        if first_pass:
            #J, J_ref, lin_ratio, rho, scp_time
            j_accum = copy.deepcopy(J)
            j_ref_accum = copy.deepcopy(J_ref)
            dj_rel_accum = copy.deepcopy(dJ_rel)
            lin_ratio_accum = copy.deepcopy(lin_ratio)
            rho_accum = copy.deepcopy(rho)
            scp_time_accum = copy.deepcopy(scp_time)
        else:
            j_accum = np.append(j_accum, J)
            j_ref_accum = np.append(j_ref_accum, J_ref)
            dj_rel_accum = np.append(dj_rel_accum, dJ_rel)
            lin_ratio_accum = np.append(lin_ratio_accum, lin_ratio)
            rho_accum = np.append(rho_accum, rho)
            scp_time_accum = np.append(scp_time_accum, scp_time)

        n_accum.extend([ng] * len(J))
        # note the length of all these accumulated arrays is the same
        iter_accum.append(len(J))
        outer_rel_error.append(dj_ref_rel)
        first_pass = False
        # convergence check of relative change in target cost
        print(f"relative error(refinement): {dj_ref_rel}")
        if dj_ref_rel <= scp_env.eps:
            break
    n_accum = np.array(n_accum)
    iter_accum = np.array(iter_accum)
    outer_rel_error = np.array(outer_rel_error)
    rdir = serialize_scp_run_plus(s_current, u_current, j_accum, j_ref_accum, dj_rel_accum,
                                  lin_ratio_accum, rho_accum, scp_time_accum,
                                  pest, scp_env, n_accum, iter_accum, outer_rel_error)
    return rdir

if __name__ == '__main__':
    pp_env = pest_pde.Env()
    scp_env = SCPEnv()
    pp_env.n = 5
    do_scp(pp_env, scp_env)
