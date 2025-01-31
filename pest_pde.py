""" numpy pde implementation. Follows the py-pde implementation closely. """
import equinox.internal
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, make_dataclass, asdict

import scipy.ndimage as nd
from scipy.integrate import odeint
from enum import Enum, StrEnum

import os
import inspect
import datetime
from time import sleep
import io, json, jsons

import jax
import jax.numpy as jnp


class BC(StrEnum):
    Dirichlet = 'Dirichlet'
    Neumann = 'Neumann'


class ControlMode(StrEnum):
    Aerial = 'Aerial'
    Spot = 'Spot'


@dataclass
class Env:
    D: float = 4  # km
    n: int = 5 # so, 1km x 1km grid spacing (see D)
    k_cp: float = 0.2
    k_pw: float = 0.3
    k_pc: float = 0.2
    d_p: float = 1.0
    k_p: float = 0.025
    k_w: float = 0.05
    k_c: float = 0.1
    K_c: float = 1.0
    bc: BC = BC.Neumann
    c0: float = 0.5
    w0: float = 0.0
    u0: float = 0.0
    flux: float = 0.1  # amplitude of flux bc
    flux_source_dist: float = 1 # in same units as D (km)
    flux_source_angle_rad: float = -np.pi/4
    pulse: float = 0.0  # amplitude of central pulse
    u_mode: ControlMode = ControlMode.Spot
    spot_resolution: int = 1 # grid points per control "spot"
    dt: float = 0.1
    T: float = 10.0

    def __post_init__(self):
        if self.u_mode == 'Aerial':
            self.u_mode = ControlMode.Aerial
        if self.u_mode == 'Spot':
            self.u_mode = ControlMode.Spot
        if self.bc == 'Neumann':
            self.bc = BC.Neumann
        if self.bc == 'Dirichlet':
            self.bc = BC.Dirichlet

def get_T(env: Env):
    """ bump time for certain plotting purposes (so the last time point is sampled)."""
    return env.T + 2*env.dt

def build_fd_lap_matrix(e: Env):
    n = e.n
    bc = e.bc
    # the spatial discretization
    h = e.D / (e.n - 1)
    h2 = h * h
    if bc == BC.Dirichlet:
        """ build a homogeneous Dirichlet BC Laplacian fd matrix for a square domain of size nxn. This
         array is of size n^2 x n^2."""
        d = np.ones((n-2,))
        d1 = np.ones((n-3,))
        t_mat = np.diag(-4*d,k=0) + np.diag(d1,k=1) + np.diag(d1, k=-1)
        t_mat = np.pad(t_mat,1,'constant', constant_values=0)
        i_mat = np.diag(d,k=0)
        i_mat = np.pad(i_mat,1,'constant', constant_values=0)
        # insert into the final matrix (I don't see a block tridiag method)
        fd_mat = np.zeros((n**2, n**2))
        for i in range(n,n*n-n,n):
            fd_mat[i:i+n, i:i+n] = t_mat
        for i in range(n,n*(n-1)-n,n):
            fd_mat[i+n:i+2*n, i:i+n] = i_mat
        for i in range(n,n*(n-1)-n,n):
            fd_mat[i:i+n, i+n:i+2*n] = i_mat
        return fd_mat / h2
    elif bc == BC.Neumann:
        """ build a Neumann BC Laplacian fd matrix for a square domain of size nxn. This
         array is of size n^2 x n^2."""
        d = np.ones((n,))
        d1 = np.ones((n-1,))
        t_mat_ext = np.diag(-3*d,k=0) + np.diag(d1,k=1) + np.diag(d1, k=-1)
        t_mat_ext[0, 0] = -2
        t_mat_ext[-1, -1] = -2
        t_mat_int = np.diag(-4*d,k=0) + np.diag(d1,k=1) + np.diag(d1, k=-1)
        t_mat_int[0, 0] = -3
        t_mat_int[-1, -1] = -3
        i_mat = np.diag(d, k=0)
        # insert into the final matrix (I don't see a block tridiag method)
        fd_mat = np.zeros((n**2, n**2))
        for i in range(0,n*n,n):
            if i==0 or i == n*(n-1):
                fd_mat[i:i + n, i:i + n] = t_mat_ext
            else:
                fd_mat[i:i+n, i:i+n] = t_mat_int
        for i in range(0,n*(n-1),n):
            fd_mat[i+n:i+2*n, i:i+n] = i_mat
        for i in range(0,n*(n-1),n):
            fd_mat[i:i+n, i+n:i+2*n] = i_mat
        return fd_mat / h2
    else:
        raise ValueError("bc must be either Dirichlet or Neumann.")


def build_p_mask(n):
    """ a handy mask to kill off time integration errors in the boundary conditions of p."""
    p_mask = np.ones((n-2, n-2))
    p_mask = np.pad(p_mask, 1, 'constant')
    p_mask = np.reshape(p_mask, (n**2,))
    return p_mask

def build_u_pattern(e):
    """control pattern for underactuated control (aerial or lower resolution spraying)"""
    if e.u_mode == 'Aerial':
        u_pat = np.ones((e.n, e.n))
        u_pat = np.reshape(u_pat, (e.n**2,))
        return u_pat
    # for loop over components of matrix
    if e.spot_resolution == 1:
        # we don't need this thing
        return []
    if np.remainder(e.n, e.spot_resolution) != 0:
        raise ValueError("spot_resolution is not an integer factor of grid size!")
    n_u = int(e.n / e.spot_resolution)
    u_pat = np.zeros((e.n**2, n_u**2))
    u_col = 0
    for i in range(0, n_u):
        for j in range(0, n_u):
            tmp = np.zeros(shape=(n_u, n_u))
            tmp[i, j] = 1
            tmp_row1 = tmp.repeat(e.spot_resolution, axis=0).repeat(e.spot_resolution, axis=1)
            u_pat[:, u_col] = np.reshape(tmp_row1, (e.n**2,))
            u_col += 1
    return u_pat

def build_flux_matrix(e: Env):
    # flux BC
    # the spatial discretization
    h = e.D / (e.n - 1)
    source_pos = e.flux_source_dist * np.array([np.cos(e.flux_source_angle_rad),np.sin(e.flux_source_angle_rad)])
    ctr = h * np.array([(e.n-1)/2, (e.n-1)/2])
    bposx, bposy = np.meshgrid(h * np.arange(0,e.n), h * np.arange(0,e.n))
    bdiffx = np.abs(bposx - ctr[0] - source_pos[0])
    bdiffy = np.abs(bposy - ctr[1] - source_pos[1])
    bdist = bdiffx*bdiffx + bdiffy*bdiffy
    flux_pattern = 1.0/bdist
    b = np.zeros((e.n-2, e.n-2), dtype=float)
    b = np.pad(b, 1, 'constant', constant_values=1.0)
    b *= flux_pattern
    # note the flux varies with the length of the boundary
    b *= e.flux / h / np.max(b)
    b = np.reshape(b, (e.n ** 2,))
    return b


def build_area_matrix(e: Env):
    # compensate for partial cells at the domain boundaries
    b = np.ones((e.n-2, e.n-2), dtype=float)
    b = np.pad(b, 1, 'constant', constant_values=0.5)
    b[0, 0] = 0.25
    b[0, -1] = 0.25
    b[-1, 0] = 0.25
    b[-1, -1] = 0.25
    b = np.reshape(b, (e.n ** 2,))
    return b

def pests(s, u, e: Env, L:np.ndarray, b:np.ndarray):
    """compute state time derivative for the pest ODE."""
    # unpack s
    c, p, w = np.split(s, 3)
    dc = -e.k_cp * p * c + e.k_c*(1 - c/e.K_c) * c
    dp = e.d_p * L @ p + b - e.k_pw * w * p + e.k_pc * c * p - e.k_p * p
    dw = u - e.k_w * w
    ds = np.concatenate([dc, dp, dw])
    return ds

def pests_aerial(s, u, e: Env, L:np.ndarray, b:np.ndarray, u_pattern:np.ndarray):
    """compute state time derivative for the pest ODE. Aerial spraying mode."""
    # unpack s
    c, p, w = np.split(s, 3)
    dc = -e.k_cp * p * c + e.k_c*(1 - c/e.K_c) * c
    dp = e.d_p * L @ p + b - e.k_pw * w * p + e.k_pc * c * p - e.k_p * p
    dw = u * u_pattern - e.k_w * w
    ds = np.concatenate([dc, dp, dw])
    return ds

def pests_jax_orig(s, u, e: Env, L:np.ndarray):
    """compute state time derivative for the pest ODE."""
    # unpack s
    i = 0
    j = e.n**2
    c = s[i:j]
    p = s[i+j:j+j]
    w = s[i+2*j:j+2*j]
    dc = -e.k_cp * p * c + e.k_c*(1 - c/e.K_c) * c
    dp = e.d_p * L @ p - e.k_pw * w * p + e.k_pc * c * p
    dw = u - e.k_w * w
    ds = jnp.concatenate([dc, dp, dw])
    return ds

def pests_jax(s, u, e: Env, L:jnp.ndarray, b:jnp.ndarray):
    """compute state time derivative for the pest ODE."""
    # unpack s
    c, p, w = jnp.split(s, 3)
    dc = -e.k_cp * jnp.multiply(p,c) + e.k_c* jnp.multiply((1 - c/e.K_c),c)
    dp = e.d_p * jnp.dot(L,p) + b - e.k_pw * jnp.multiply(w,p) + e.k_pc * jnp.multiply(c, p)
    dw = u - e.k_w * w
    ds = jnp.concatenate([dc, dp, dw])
    return ds


def pests_aerial_jax(s, u, e: Env, L:jnp.ndarray, b:jnp.ndarray, u_pattern:np.ndarray):
    """compute state time derivative for the pest ODE."""
    # unpack s
    c, p, w = jnp.split(s, 3)
    dc = -e.k_cp * jnp.multiply(p,c) + e.k_c* jnp.multiply((1 - c/e.K_c),c)
    dp = e.d_p * jnp.dot(L,p) + b - e.k_pw * jnp.multiply(w,p) + e.k_pc * jnp.multiply(c, p)
    dw = u * u_pattern - e.k_w * w
    ds = jnp.concatenate([dc, dp, dw])
    return ds


def init_state(e):
    """initialize state to match py-pde simulation."""
    # start init in 2d image space, then reshape
    # crop is 1.0
    c = e.c0 * np.ones((e.n, e.n), dtype=float)
    p = np.zeros((e.n, e.n), dtype=float)
    #p[e.n//2, e.n//2] = 100 * (e.n/32)**2
    p[e.n // 2, e.n // 2] = e.pulse
    # smooth
    p = nd.gaussian_filter(p, sigma=e.n/32)

    # parabolic in y pesticide
    #wv = np.expand_dims(np.arange(0, e.n), axis=0)
    #w1 = np.ones((e.n,1))
    #wv2 = wv * wv / (e.n**2/4)
    #w = wv2.T @ w1.T
    w = e.w0 * np.ones((e.n, e.n),dtype=float)
    # reshape into state
    c = np.reshape(c, (e.n ** 2,))
    p = np.reshape(p, (e.n ** 2,))
    w = np.reshape(w, (e.n ** 2,))
    s = np.concatenate([c, p, w])
    if e.u_mode == ControlMode.Spot:
        u = e.u0 * np.ones((e.n**2,))
    else:
        u = e.u0
    return s, u

def unpack_frame(k, e, ca, pa, wa, u):
    cf = np.reshape(ca[k],(e.n,e.n))
    pf = np.reshape(pa[k],(e.n,e.n))
    wf = np.reshape(wa[k],(e.n,e.n))
    uf = np.reshape(u[k],(e.n,e.n))
    return cf, pf, wf, uf

def animate_states(e, s_rec, u_rec):
    def update(kf):
        cf, pf, wf, uf = unpack_frame(kf, e, ca, pa, wa, u_rec)
        c_im.set_data(cf)
        p_im.set_data(pf)
        w_im.set_data(wf)
        u_im.set_data(uf)
    p_mask = build_p_mask(e.n)
    numplots = 1
    numframes = s_rec.shape[0]
    k = 1
    ca, pa, wa = np.split(s_rec, 3, axis=1)
    cmax = np.max(ca)
    pmax = np.max(pa)
    wmax = np.max(wa)
    umax = np.max(u_rec)
    c, p, w, u = unpack_frame(k, e, ca, pa, wa, u_rec)
    fig = plt.figure()
    plt.subplot(numplots, 4, 1)
    c_im = plt.imshow(c, origin='lower',vmin=0, vmax=cmax)
    plt.title('c')
    #plt.ylabel(f"t={k}")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(numplots, 4, 2)
    p_im = plt.imshow(p, origin='lower',vmin=0, vmax=pmax)
    plt.title('p')
    plt.axis('off')
    plt.subplot(numplots, 4, 3)
    w_im = plt.imshow(w, origin='lower',vmin=0, vmax=wmax)
    plt.title('w')
    plt.axis('off')
    plt.subplot(numplots, 4, 4)
    u_im = plt.imshow(u, origin='lower',vmin=0, vmax=umax)
    plt.title('u')
    plt.axis('off')

    ani = animation.FuncAnimation(fig=fig, func=update, frames=numframes-1)
    return ani
    #ani.save(filename="ffmpeg_example.mp4", writer="ffmpeg")


def plot_states(e, s_rec, u_rec, mode='strided', step_count=10):
    interval = u_rec.shape[0] // step_count
    kint = 0
    if mode == 'strided':
        plot_k = range(0, u_rec.shape[0], interval)
    elif mode == 'early':
        plot_k = range(0, step_count)
    else:
        raise ValueError("mode must be either 'strided' or 'early'.")
    numplots = len(plot_k)
    #plt.figure(figsize=(8.5, 11))
    pfig = plt.figure()
    ca, pa, wa = np.split(s_rec, 3, axis=1)
    cmax = np.max(ca)
    pmax = np.max(pa)
    wmax = np.max(wa)
    umax = np.max(u_rec)
    for k in plot_k:
        t = e.dt * k
        #s = s_rec[k]
        u = u_rec[k]
        # unpack s
        i = 0
        j = e.n**2
        c = np.reshape(ca[k],(e.n,e.n))
        p = np.reshape(pa[k],(e.n,e.n))
        w = np.reshape(wa[k],(e.n,e.n))
        # u can be different sizes if not reconstituted in deserialization
        if u.size == 1:
            u = np.reshape(u * np.ones((e.n, e.n), dtype=float), (e.n, e.n))
        else:
            u = np.reshape(u, (e.n, e.n))
        plt.subplot(numplots, 4, kint*4+1)
        plt.imshow(c, origin='lower',vmin=0, vmax=cmax)
        if k == 0:
            plt.title('c')
        plt.ylabel(f"t={t}")
        plt.xticks([])
        plt.yticks([])
        #plt.axis('off')
        plt.subplot(numplots, 4, kint*4+2)
        plt.imshow(p, origin='lower',vmin=0, vmax=pmax)
        if k == 0:
            plt.title('p')
        plt.axis('off')
        plt.subplot(numplots, 4, kint*4+3)
        plt.imshow(w, origin='lower',vmin=0, vmax=wmax)
        if k == 0:
            plt.title('w')
        plt.axis('off')
        plt.subplot(numplots, 4, kint*4+4)
        plt.imshow(u, origin='lower',vmin=0, vmax=umax)
        if k == 0:
            plt.title('u')
        plt.axis('off')
        kint += 1
        #print([np.min(c), np.max(c), np.min(p), np.max(p), np.min(w), np.max(w), np.min(u), np.max(u)])
    plt.show()
    return pfig


class PestSim:
    def __init__(self, e: Env):
        self.e = e
        self.L = build_fd_lap_matrix(e)
        self.Ljax = jnp.array(self.L)
        self.u_pattern = build_u_pattern(e)
        self.b = build_flux_matrix(e)
        self.bjax = jnp.array(self.b)

    def pests_wrapper_su_jax(self, s, u):
        return pests_jax(s, u, self.e, self.Ljax, self.bjax)

    def pests_wrapper_aerial_su_jax(self, s, u):
        return pests_aerial_jax(s, u, self.e, self.Ljax, self.bjax, self.u_pattern)

    def pests_wrapper(self, s, t, u):
        return pests(s, u, self.e, self.L, self.b)

    def pests_wrapper_aerial(self, s, t, u):
        return pests_aerial(s, u, self.e, self.L, self.b, self.u_pattern)

    def simulate(self):
        s_init, u_init = init_state(self.e)
        t = np.arange(0.0, get_T(self.e), self.e.dt)
        num_timesteps = len(t)
        if self.e.u_mode == ControlMode.Aerial:
            m_sim = 1
            u = np.zeros(shape=(num_timesteps, m_sim))
        else:
            m_sim = len(u_init)
            u = np.zeros(shape=(num_timesteps, m_sim))
        n_sim = len(s_init)
        s = np.zeros(shape=(num_timesteps, n_sim))
        s[0] = s_init
        u[0] = u_init
        for k in range(0, num_timesteps):
            u[k] = u_init
            # this conditional prevents the last value in u from being
            # unset
            if k < num_timesteps - 1:
                # continuous
                if self.e.u_mode == ControlMode.Spot:
                    s[k + 1] = odeint(self.pests_wrapper, s[k], t[k:k + 2], (u[k],))[1]
                else:
                    s[k + 1] = odeint(self.pests_wrapper_aerial, s[k], t[k:k + 2], (u[k],))[1]
        return s, u

    def resimulate(self, s_ref, u):
        """ reprocess scp (or sim!) output into a high fidelity state sequence"""
        #s_init, u_init = init_state(self.e)
        t = np.arange(0.0, get_T(self.e), self.e.dt)
        num_timesteps = len(t)
        u_vec=[]
        if self.e.u_mode == ControlMode.Aerial:
            # I am going to coerce this back to a vector
            # so that we are reprocessing with the same f(s,u)
            u_vec = np.median(u, axis=1)
        n_sim = s_ref.shape[1]
        s = np.zeros(shape=(num_timesteps, n_sim))
        s[0] = s_ref[0]
        for k in range(0, num_timesteps):
            # this conditional prevents the last value in u from being
            # unset
            if k < num_timesteps - 1:
                if self.e.u_mode == ControlMode.Spot:
                    s[k + 1] = odeint(self.pests_wrapper, s[k], t[k:k + 2], (u[k],))[1]
                else:
                    s[k + 1] = odeint(self.pests_wrapper_aerial, s[k], t[k:k + 2], (u_vec[k],))[1]
        return s, u

    def save_simulation(self, s, u):
        np.save('pest_s.npy', s)
        if self.e.u_mode == ControlMode.Aerial:
            usave = self.u_pattern * u
            np.save('pest_u.npy', usave)
        else:
            np.save('pest_u.npy', u)


def serialize_sim(s: np.ndarray, u: np.ndarray, sim: PestSim, override_dir: str = '') -> str:
    if override_dir == '':
        sleep(1.1)  # I only have HHMMSS, so ...
        now = datetime.datetime.now()
        rdir = 'sim_' + now.strftime('%y%m%d-%H%M%S')
    else:
        rdir = override_dir
    os.makedirs(rdir, exist_ok=True)
    np.save(os.path.join(rdir, 'pest_s.npy'), s)
    if sim.e.u_mode == ControlMode.Aerial:
        u_save = sim.u_pattern * u
        np.save(os.path.join(rdir, 'pest_u.npy'), u_save)
    else:
        np.save(os.path.join(rdir, 'pest_u.npy'), u)

    # serialize env
    file_env = os.path.join(rdir, 'pest_pde.env.json')
    with io.open(file_env, 'w', encoding='utf-8') as f:
        f.write(json.dumps(jsons.dump(sim.e), ensure_ascii=False))

    # serialize ... code
    # file_do = os.path.join(rdir, 'do_scp.txt')
    # with io.open(file_do, 'w', encoding='utf-8') as f:
    #     f.write(inspect.getsource(do_scp))
    return rdir


def deserialize_sim(rdir: str) -> (np.ndarray, np.ndarray, Env):
    """inverse of obove"""
    # backwards compat
    if os.path.exists(os.path.join(rdir, 'scp_pest_s.npy')):
        s = np.load(os.path.join(rdir, 'scp_pest_s.npy'))
        u = np.load(os.path.join(rdir, 'scp_pest_u.npy'))
    else:
        s = np.load(os.path.join(rdir, 'pest_s.npy'))
        u = np.load(os.path.join(rdir, 'pest_u.npy'))
    # env
    file_env = os.path.join(rdir, 'pest_pde.env.json')
    with io.open(file_env, 'r', encoding='utf-8') as f:
        env_dict = json.load(f)
    env = Env(**env_dict)
    return s, u, env


def animate_sim(rdir: str):
    s, u, e = deserialize_sim(rdir)
    ani = animate_states(e, s, u)
    ani.save(filename=os.path.join(rdir,"pest.mp4"), writer="ffmpeg")


def crop_function(env: Env, t: float) -> float:
    """compute crop evolution with no pests to time T"""
    c_t = env.K_c * env.c0 / (env.c0 + (env.K_c - env.c0) * np.exp(-env.k_c * t))
    return c_t
