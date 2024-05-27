import json

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
    # PART (b) ################################################################
    # INSTRUCTIONS: Use JAX to affinize `f` around `(s, u)` in two lines.
    c = jax.jit(f)(s, u)
    A, B = map(jnp.array, jax.jit(jax.jacfwd(f, (0, 1)))(s, u))
    # END PART (b) ############################################################
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


def solve_swingup_scp(f, s0, s_goal, N, P, Q, R, u_max, ρ, eps, max_iters):
    """Solve the cart-pole swing-up problem via SCP.

    Arguments
    ---------
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
    ρ : float
        Trust region radius.
    eps : float
        Termination threshold for SCP.
    max_iters : int
        Maximum number of SCP iterations.

    Returns
    -------
    s : numpy.ndarray
        A 2-D array where `s[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u : numpy.ndarray
        A 2-D array where `u[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    J : numpy.ndarray
        A 1-D array where `J[i]` is the SCP sub-problem cost after the i-th
        iteration, for `i = 0, 1, ..., (iteration when convergence occured)`
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
    J[0] = np.inf
    for i in (prog_bar := tqdm(range(max_iters))):
        s, u, J[i + 1] = scp_iteration(f, s0, s_goal, s, u, N, P, Q, R, u_max, ρ)
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


def scp_iteration(f, s0, s_goal, s_prev, u_prev, N, P, Q, R, u_max, ρ):
    """Solve a single SCP sub-problem for the cart-pole swing-up problem.

    Arguments
    ---------
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
    ρ : float
        Trust region radius.

    Returns
    -------
    s : numpy.ndarray
        A 2-D array where `s[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u : numpy.ndarray
        A 2-D array where `u[k]` is the open-loop state at time step `k`,
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
    # PART (c) ################################################################
    # INSTRUCTIONS: Construct the convex SCP sub-problem.
    objective = cvx.quad_form((s_cvx[N] - s_goal), P) + cvx.sum(
        [cvx.quad_form(s_cvx[i1] - s_goal, Q) + cvx.quad_form(u_cvx[i1], R) for i1 in range(N)])
    #objective = cvx.quad_form((s_cvx[N] - s_goal), P) + cvx.sum(
    #    [cvx.quad_form(u_cvx[i1], R) for i1 in range(N)])
    constraints = [s_cvx[i2 + 1] == c[i2] + A[i2] @ (s_cvx[i2] - s_prev[i2]) +
                   B[i2] @ (u_cvx[i2] - u_prev[i2]) for i2 in range(N)]
    constraints += [s_cvx[0] == s0]
    constraints += [cvx.min(u_cvx) >= 0]
    constraints += [cvx.max(u_cvx) <= u_max]
    constraints += [cvx.max(cvx.abs(s_cvx - s_prev)) <= ρ]
    constraints += [cvx.max(cvx.abs(u_cvx - u_prev)) <= ρ]
    # END PART (c) ############################################################
    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve(solver=cvx.ECOS)
    #prob.solve()
    #prob.solve(verbose=True)
    if prob.status != "optimal":
        raise RuntimeError("SCP solve failed. Problem status: " + prob.status)
    s = s_cvx.value
    u = u_cvx.value
    J = prob.objective.value
    return s, u, J


def serialize_scp_run(s: np.ndarray, u: np.ndarray, J: np.ndarray, sim: pest_pde.PestSim):
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

    # serialize do_scp
    file_do = os.path.join(rdir, 'do_scp.txt')
    with io.open(file_do, 'w', encoding='utf-8') as f:
        f.write(inspect.getsource(do_scp))


def do_scp(pp_env: pest_pde.Env):
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
    T = pp_env.T  # total simulation time
    # we want crops at what we can expect at time t
    # we want pest at 0
    # we want pesticide at 0
    crop_target = pest_pde.crop_function(pp_env, T)
    print('crop_target: ' + str(crop_target))
    s_goal = np.concatenate([crop_target * np.ones((n_s,)), np.zeros((n_s,)), np.zeros((n_s,))])  # desired field state
    goal_weights = np.concatenate([np.ones((n_s,)), 10*np.ones((n_s,)), 0.001*np.ones((n_s,))])
    P = 1e-1 * np.diag(goal_weights)
    #P = 1e2 * np.eye(n)  # terminal state cost matrix
    Q = 1e2 * np.diag(goal_weights)
    #Q = 1e3 * np.eye(n) # state cost matrix
    #Q = 0.0 * np.eye(n) # state cost matrix
    if pp_env.u_mode == pest_pde.ControlMode.Aerial:
        R = 1e-1  # control cost
    else:
        R = 1e-1 * np.eye(m)  # control cost matrix
    ρ = 0.5  # trust region parameter
    u_max = 10.0  # control effort bound
    eps = 5e-2  # convergence tolerance
    max_iters = 100  # maximum number of SCP iterations
    animate = False  # flag for animation

    pest = pest_pde.PestSim(pp_env)
    # Initialize the discrete-time dynamics
    if pp_env.u_mode == pest_pde.ControlMode.Spot:
        fd = jax.jit(discretize(pest.pests_wrapper_su_jax, dt))
    else:
        fd = jax.jit(discretize(pest.pests_wrapper_aerial_su_jax, dt))

    # Solve the swing-up problem with SCP
    t = np.arange(0.0, T + dt, dt)
    N = t.size - 1
    s, u, J = solve_swingup_scp(fd, s0, s_goal, N, P, Q, R, u_max, ρ, eps, max_iters)

    serialize_scp_run(s, u, J, pest)

    # Simulate open-loop control
    #for k in range(N):
    #    s[k + 1] = fd(s[k], u[k])


if __name__ == '__main__':
    env = pest_pde.Env()
    env.n = 5
    do_scp(env)
