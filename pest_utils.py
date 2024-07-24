import copy

import pest_pde as pp
import numpy as np
import matplotlib.pyplot as plt
import scp_pest
import os, io, json, jsons
from dataclasses import dataclass, make_dataclass, asdict
import inspect
import typing
import cv2


def plot_stats(t, f):
    plt.plot(t, np.min(f, axis=1), c='b')
    plt.plot(t, np.median(f, axis=1), c='g')
    plt.plot(t, np.max(f, axis=1), c='r')
    plt.legend(['min', 'median', 'max'], prop={'size':8})


def time_plots(rdir):
    s, u, env = pp.deserialize_sim(rdir)
    # the spatial discretization
    h = env.D / (env.n - 1)
    h2 = h * h
    area_mat = pp.build_area_matrix(env)
    time = env.dt * np.arange(0, len(s))
    time_u = env.dt * np.arange(0, len(u))
    tmax = time[-1]
    crop_max = pp.crop_function(env, tmax)
    crop_max_sum = crop_max * env.D**2
    pfig = plt.figure(figsize=(8.5, 5))
    c, p, w = np.split(s, 3, axis=1)
    plt.subplot(2, 4, 1)
    plot_stats(time, c)
    plt.axhline(y=crop_max, color='r', linestyle='--')
    #plt.axhline(y=0.73, color='r', linestyle='--')
    plt.title('crop density c')
    plt.subplot(2, 4, 2)
    plot_stats(time, p)
    #plt.axhline(y=0.73, color='r', linestyle='--')
    plt.title('pest density p')
    plt.subplot(2, 4, 3)
    plot_stats(time, w)
    #plt.axhline(y=0.73, color='r', linestyle='--')
    plt.title('p-cide density w')
    plt.subplot(2, 4, 4)
    plot_stats(time_u, u)
    #plt.axhline(y=0.73, color='r', linestyle='--')
    plt.title('control rate u')
    plt.subplot(2, 4, 5)
    #plt.plot(time, h2 * np.sum(c, axis=1))
    plt.plot(time, h2 * c @ area_mat)
    plt.axhline(y=crop_max_sum, color='r', linestyle='--')
    plt.title('total crop')
    plt.xlabel('time')
    plt.subplot(2, 4, 6)
    #plt.plot(time, h2 * np.sum(p, axis=1))
    plt.plot(time, h2 * p @ area_mat)
    plt.title('total pest')
    plt.xlabel('time')
    plt.subplot(2, 4, 7)
    #plt.plot(time, h2 * np.sum(w, axis=1))
    plt.plot(time, h2 * w @ area_mat)
    plt.title('total p-cide')
    plt.xlabel('time')
    plt.subplot(2, 4, 8)
    #plt.plot(time_u, env.dt * h2 * np.cumsum(np.sum(u, axis=1)))
    plt.plot(time_u, env.dt * h2 * np.cumsum(u @ area_mat))
    plt.title('total control')
    plt.xlabel('time')
    pfig.tight_layout()
    plt.show()
    return pfig


def plot_scp(scp_env: scp_pest.SCPEnv, J: np.ndarray, J_ref: np.ndarray, dJ_rel: np.ndarray,
             lin_ratio: np.ndarray, rho: np.ndarray, scp_time: np.ndarray,
             n_spatial: np.ndarray, iter_count: np.ndarray):
    # collect some data from the settings
    eta_1 = scp_env.eta_1
    eta_2 = scp_env.eta_2
    plen = len(J)-1
    pfig = plt.figure(figsize=(6, 8.5))
    plt.subplot(6, 1, 1)
    plt.semilogy(J, color='g', linestyle='-', label='J')
    plt.semilogy(J_ref, color='r', linestyle='-', label='J_ref')
    plt.xlim(0, plen)
    plt.legend()
    plt.title('log SCP objective')
    plt.subplot(6, 1, 2)
    plt.semilogy(dJ_rel, color='g', linestyle='-', label='dJ_rel')
    plt.title('log SCP objective relative change')
    plt.xlim(0, plen)
    plt.subplot(6, 1, 3)
    plt.plot(lin_ratio, color='b', linestyle='-', label='lin_ratio')
    plt.axhline(y=eta_1, color='r', linestyle='--')
    plt.axhline(y=eta_2, color='g', linestyle='--')
    plt.title('SCP error ratio')
    plt.xlim(0, plen)
    plt.ylim([-0.5, 1.5])
    plt.subplot(6, 1, 4)
    plt.plot(rho, color='b', linestyle='-', label='rho')
    plt.title('SCP trust region')
    plt.xlim(0, plen)
    plt.subplot(6, 1, 5)
    plt.plot(scp_time, color='b', linestyle='-', label='scp_time')
    plt.title('SCP iteration solution time')
    plt.xlabel('SCP iteration')
    plt.xlim(0, plen)
    plt.subplot(6, 1, 6)
    plt.plot(n_spatial, color='b', linestyle='-', label='n')
    plt.title('PDE spatial grid n')
    plt.xlabel('SCP iteration')
    plt.xlim(0, plen)
    pfig.tight_layout()
    plt.show()
    return pfig


def collect_data(rdir, annot):
    s, u, env = pp.deserialize_sim(rdir)
    # the spatial discretization
    h = env.D / (env.n - 1)
    h2 = h * h
    area_mat = pp.build_area_matrix(env)
    time = env.dt * np.arange(0, len(s))
    time_u = env.dt * np.arange(0, len(u))
    tmax = time[-1]
    crop_max = pp.crop_function(env, tmax)
    crop_max_sum = crop_max * env.D**2
    c, p, w = np.split(s, 3, axis=1)
    csum = h2 * c @ area_mat
    psum = h2 * p @ area_mat
    wsum = h2 * w @ area_mat
    ucumsum = env.dt * h2 * np.cumsum(u @ area_mat)
    print(annot)
    print(f"raw: {csum[-1]:.2f}, {psum[-1]:.2f}, {wsum[-1]:.2f}, {ucumsum[-1]:.2f}")
    print(f"norm: {csum[-1]/csum[-1]:.2f}, {psum[-1]/csum[-1]:.2f}, {wsum[-1]/csum[-1]:.2f}, {ucumsum[-1]/csum[-1]:.2f}")


def collect_run_time(rdir, annot):
    s, u, env, J, J_ref, dJ_rel, lin_ratio, rho, scp_time, scp_env, n_spatial, iter_count \
        = scp_pest.deserialize_scp(rdir)
    print(annot)
    print(f"cumulative time(secs): {np.sum(scp_time)}")

def compute_pde_error(e_ref: pp.Env, s_ref: np.ndarray, u_ref: np.ndarray, e: pp.Env, s: np.ndarray, u: np.ndarray):
    """compute error in pde approximation"""
    # use opencv to area-interpolate to the lower resolution
    c, p, w = np.split(s, 3, axis=1)
    c_ref, p_ref, w_ref = np.split(s_ref, 3, axis=1)
    # compute error for each time point
    c_err_max = 0
    p_err_max = 0
    w_err_max = 0
    c_max = np.max(c_ref)
    p_max = np.max(p_ref)
    w_max = np.max(w_ref)
    n_ref = e_ref.n
    n = e.n
    dt_ref = e_ref.dt
    dt = e.dt
    # assume dt increments are integer factors
    tf = int(dt / dt_ref)
    for t in range(0, c.shape[0]):
        c2_ref = np.reshape(c_ref[t*tf], (e_ref.n, e_ref.n))
        p2_ref = np.reshape(p_ref[t*tf], (e_ref.n, e_ref.n))
        w2_ref = np.reshape(w_ref[t*tf], (e_ref.n, e_ref.n))
        c2 = np.reshape(c[t], (e.n, e.n))
        p2 = np.reshape(p[t], (e.n, e.n))
        w2 = np.reshape(w[t], (e.n, e.n))
        c_cmp = cv2.resize(c2_ref, (e.n, e.n), interpolation=cv2.INTER_AREA)
        p_cmp = cv2.resize(p2_ref, (e.n, e.n), interpolation=cv2.INTER_AREA)
        w_cmp = cv2.resize(w2_ref, (e.n, e.n), interpolation=cv2.INTER_AREA)
        c_err = np.max(np.abs(c_cmp - c2))
        c_err_max = max(c_err_max, c_err)
        p_err = np.max(np.abs(p_cmp - p2))
        p_err_max = max(p_err_max, p_err)
        w_err = np.max(np.abs(w_cmp - w2))
        w_err_max = max(w_err_max, w_err)
    c_rel = 0
    if c_max > 0:
        c_rel = c_err_max / c_max
    p_rel = 0
    if p_max > 0:
        p_rel = p_err_max / p_max
    w_rel = 0
    if w_max > 0:
        w_rel = w_err_max / w_max
    return dt_ref, dt, n_ref, n, c_max, c_err_max, c_rel, p_max, p_err_max, p_rel, w_max, w_err_max, w_rel


def read_json_report(filename):
    with io.open(filename, 'r', encoding='utf-8') as f:
        dir_list = json.load(f)
    return dir_list
