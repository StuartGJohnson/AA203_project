""" numpy pde implementation. Follows the py-pde implementation closely. """

import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass

import scipy.ndimage as nd
from scipy.integrate import odeint
from enum import Enum

import jax
import jax.numpy as jnp

class BC(Enum):
    Dirichlet = 1
    Neumann = 2

@dataclass
class Env:
    n = 9
    k_cp = 0.2
    k_pw = 0.3
    k_pc = 0.2
    d_p = 0.25
    k_w = 0.01
    k_c = 0.1
    K_c = 1.0
    bc = BC.Neumann




def build_fd_lap_matrix(n, bc: BC):
    if bc == BC.Dirichlet:
        """ build a homogeneous Dirichlet BC Laplacian fd matrix for a square domain of size nxn. This
         array is of size n^2 x n^2."""
        d = np.ones((n-2,))
        d1 = np.ones((n-3,))
        t_mat = np.diag(-4*d,k=0) + np.diag(d1,k=1) + np.diag(d1, k=-1)
        t_mat = np.pad(t_mat,1,'constant')
        i_mat = np.diag(d,k=0)
        i_mat = np.pad(i_mat,1,'constant')
        # insert into the final matrix (I don't see a block tridiag method)
        fd_mat = np.zeros((n**2, n**2))
        for i in range(n,n*n-n,n):
            fd_mat[i:i+n, i:i+n] = t_mat
        for i in range(n,n*(n-1)-n,n):
            fd_mat[i+n:i+2*n, i:i+n] = i_mat
        for i in range(n,n*(n-1)-n,n):
            fd_mat[i:i+n, i+n:i+2*n] = i_mat
        return fd_mat
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
        return fd_mat
    else:
        raise ValueError("bc must be either Dirichlet or Neumann.")


def build_p_mask(n):
    """ a handy mask to kill of time integration errors in the boundary conditions of p."""
    p_mask = np.ones((n-2, n-2))
    p_mask = np.pad(p_mask, 1, 'constant')
    p_mask = np.reshape(p_mask, (n**2,))
    return p_mask

def pests(s, u, e: Env, L:np.ndarray):
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

def pests_jax(s, u, e: Env, L:np.ndarray):
    """compute state time derivative for the pest ODE."""
    # unpack s
    c, p, w = jnp.split(s, 3)
    dc = -e.k_cp * jnp.multiply(p,c) + e.k_c* jnp.multiply((1 - c/e.K_c),c)
    dp = e.d_p * jnp.dot(L,p) - e.k_pw * jnp.multiply(w,p) + e.k_pc * jnp.multiply(c, p)
    dw = u - e.k_w * w
    ds = jnp.concatenate([dc, dp, dw])
    return ds

def init_state(e):
    """initialize state to match py-pde simulation."""
    # start init in 2d image space, then reshape
    # crop is 1.0
    c = 0.5 * np.ones((e.n, e.n), dtype=float)
    p = np.zeros((e.n, e.n), dtype=float)
    p[e.n//2, e.n//2] = 100 * (e.n/32)**2
    # smooth
    p = nd.gaussian_filter(p, sigma=e.n/32)
    # parabolic in y pesticide
    #wv = np.expand_dims(np.arange(0, e.n), axis=0)
    #w1 = np.ones((e.n,1))
    #wv2 = wv * wv / (e.n**2/4)
    #w = wv2.T @ w1.T
    w = np.zeros((e.n, e.n))
    # reshape into state
    c = np.reshape(c, (e.n ** 2,))
    p = np.reshape(p, (e.n ** 2,))
    w = np.reshape(w, (e.n ** 2,))
    s = np.concatenate([c, p, w])
    u = np.zeros((e.n**2,))
    return s, u


def plot_states(e, s_rec, u_rec, mode='strided'):
    p_mask = build_p_mask(e.n)
    interval = u_rec.shape[0] // 10
    kint = 0
    if mode == 'strided':
        plot_k = range(0, u_rec.shape[0], interval)
    elif mode == 'early':
        plot_k = range(0, 10)
    else:
        raise ValueError("mode must be either 'strided' or 'early'.")
    numplots = len(plot_k)
    plt.figure(figsize=(8.5, 11))
    for k in plot_k:
        s = s_rec[k]
        u = u_rec[k]
        # unpack s
        i = 0
        j = e.n**2
        c = np.reshape(s[i:j],(e.n,e.n))
        p = np.reshape(s[i+j:j+j],(e.n,e.n))
        w = np.reshape(s[i+2*j:j+2*j],(e.n,e.n))
        u = np.reshape(u,(e.n,e.n))
        plt.subplot(numplots, 4, kint*4+1)
        plt.imshow(c, origin='lower',vmin=0, vmax=1)
        if k == 0:
            plt.title('c')
        plt.ylabel(f"t={k}")
        plt.xticks([])
        plt.yticks([])
        #plt.axis('off')
        plt.subplot(numplots, 4, kint*4+2)
        plt.imshow(p, origin='lower',vmin=0, vmax=1)
        if k == 0:
            plt.title('p')
        plt.axis('off')
        plt.subplot(numplots, 4, kint*4+3)
        plt.imshow(w, origin='lower',vmin=0, vmax=1)
        if k == 0:
         plt.title('w')
        plt.axis('off')
        plt.subplot(numplots, 4, kint*4+4)
        plt.imshow(u, origin='lower',vmin=0, vmax=1)
        if k == 0:
         plt.title('u')
        plt.axis('off')
        kint += 1
        print([np.min(c), np.max(c), np.min(p), np.max(p), np.min(w), np.max(w), np.min(u), np.max(u)])
    plt.show()
        #print(np.max([np.max(p[0,:]),np.max(p[:,0]),np.max(p[-1,:]),np.max(p[:,-1])]))
        #print(np.max([np.max(p[0, :]), np.max(p[:, 0])]))



class PestSim:
    def __init__(self, e: Env):
        self.e = e
        self.L = build_fd_lap_matrix(e.n, e.bc)
        self.Ljax = jnp.array(self.L)

    def pests_wrapper_su_jax(self, s, u):
        return pests_jax(s, u, self.e, self.Ljax)

    def pests_wrapper(self, s, t, u):
        return pests(s, u, self.e, self.L)

    def simulate(self):
        s_init, u_init = init_state(self.e)
        t = np.arange(0.0, 60.0, 1 / 10)
        num_timesteps = len(t)
        m_sim = len(u_init)
        n_sim = len(s_init)
        u = np.zeros(shape=(num_timesteps, m_sim))
        s = np.zeros(shape=(num_timesteps, n_sim))
        s[0] = s_init
        u[0] = u_init
        for k in range(0, num_timesteps):
            u[k] = u_init
            # this conditional prevents the last value in u from being
            # unset
            if k < num_timesteps - 1:
                # continuous
                s[k + 1] = odeint(self.pests_wrapper, s[k], t[k:k + 2], (u[k],))[1]
        return s, u






