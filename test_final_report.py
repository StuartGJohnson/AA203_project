import unittest
import pest_pde as pp
import numpy as np
import matplotlib.pyplot as plt
import scp_pest
import os, io, json, dataclasses, jsons
import inspect
import typing


def plot_stats(t, f):
    plt.plot(t, np.min(f, axis=1), c='b')
    plt.plot(t, np.median(f, axis=1), c='g')
    plt.plot(t, np.max(f, axis=1), c='r')
    plt.legend(['min', 'median', 'max'], prop={'size':8})

def time_plots(rdir):
    s, u, env = pp.deserialize_sim(rdir)
    time = env.dt * np.arange(0, len(s))
    time_u = time[:-1]
    tmax = time[-1]
    crop_max = pp.crop_function(env,tmax)
    crop_max_sum = crop_max * env.n**2
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
    plt.plot(time, np.sum(c, axis=1))
    plt.axhline(y=crop_max_sum, color='r', linestyle='--')
    plt.title('total crop')
    plt.xlabel('time')
    plt.subplot(2, 4, 6)
    plt.plot(time, np.sum(p, axis=1))
    plt.title('total pest')
    plt.xlabel('time')
    plt.subplot(2, 4, 7)
    plt.plot(time, np.sum(w, axis=1))
    plt.title('total p-cide')
    plt.xlabel('time')
    plt.subplot(2, 4, 8)
    plt.plot(time_u, env.dt * np.cumsum(np.sum(u, axis=1)))
    plt.title('total control')
    plt.xlabel('time')
    pfig.tight_layout()
    plt.show()
    return pfig

def collect_data(rdir, annot):
    s, u, env = pp.deserialize_sim(rdir)
    time = env.dt * np.arange(0, len(s))
    time_u = time[:-1]
    tmax = time[-1]
    crop_max = pp.crop_function(env,tmax)
    crop_max_sum = crop_max * env.n**2
    c, p, w = np.split(s, 3, axis=1)
    csum = np.sum(c, axis=1)
    psum = np.sum(p, axis=1)
    wsum = np.sum(w, axis=1)
    ucumsum = env.dt * np.cumsum(np.sum(u, axis=1))
    print(annot)
    print(f"raw: {csum[-1]:.2f}, {psum[-1]:.2f}, {wsum[-1]:.2f}, {ucumsum[-1]:.2f}")
    print(f"norm: {csum[-1]/csum[-1]:.2f}, {psum[-1]/csum[-1]:.2f}, {wsum[-1]/csum[-1]:.2f}, {ucumsum[-1]/csum[-1]:.2f}")

class MyTestCase2(unittest.TestCase):
    def test_fr_sim_no_control(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Aerial
        e.u0 = 0.0
        e.k_w = 0.2
        e.d_p = 0.3
        ps = pp.PestSim(e)
        s, u = ps.simulate()
        pp.serialize_sim(s, u, ps)

    def test_fr_anim_sim_no_control(self):
        rdir = 'sim_240603-103842'
        pp.animate_sim(rdir)

    def test_fr_plot_sim_not_control(self):
        rdir = 'sim_240603-103842'
        s, u, env = pp.deserialize_sim(rdir)
        pfig = pp.plot_states(env, s, u, mode='strided', step_count=5)
        pfig.savefig(os.path.join(rdir,'slices.png'))

    def test_fr_time_plots_sim_no_control(self):
        rdir = 'sim_240603-103842'
        pfig = time_plots(rdir)
        pfig.savefig(os.path.join(rdir, 'time.png'))


    #### first aerial scp
    def test_fr_scp_aerial_fastw_slowdp(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Aerial
        e.k_w = 0.2
        e.d_p = 0.3
        e.T = 10 + 2 * e.dt
        se = scp_pest.SCPEnv()
        se.rho = 0.5
        se.w_wt = 1e-2
        scp_pest.do_scp(e, se)

    def test_fr_anim_scp_aerial_fastw_slowdp(self):
        rdir = 'scp_240603-152427'
        rdir = 'scp_240603-181903'
        pp.animate_sim(rdir)

    def test_fr_plot_scp_aerial_fastw_slowdp(self):
        rdir = 'scp_240603-152427'
        rdir = 'scp_240603-181903'
        s, u, env = pp.deserialize_sim(rdir)
        pfig = pp.plot_states(env, s, u, mode='strided', step_count=5)
        pfig.savefig(os.path.join(rdir,'slices.png'))

    def test_fr_time_plots_scp_aerial_fastw_slowdp(self):
        rdir = 'scp_240603-152427'
        rdir = 'scp_240603-181903'
        pfig = time_plots(rdir)
        pfig.savefig(os.path.join(rdir, 'time.png'))


    #### second aerial scp - flip P/Q weights
    def test_fr_scp_aerial_fastw_slowdp_flipped(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Aerial
        e.k_w = 0.2
        e.d_p = 0.3
        e.T = 10 + 2 * e.dt
        se = scp_pest.SCPEnv()
        se.rho = 0.5
        se.P_wt = 1e2
        se.Q_wt = 1e-1
        se.w_wt = 1e-2
        scp_pest.do_scp(e, se)

    def test_fr_anim_scp_aerial_fastw_slowdp_flipped(self):
        rdir = 'scp_240603-155025'
        rdir = 'scp_240603-180508'
        rdir = 'scp_240603-181411'
        rdir = 'scp_240603-182214'
        pp.animate_sim(rdir)

    def test_fr_plot_scp_aerial_fastw_slowdp_flipped(self):
        rdir = 'scp_240603-155025'
        rdir = 'scp_240603-180508'
        rdir = 'scp_240603-181411'
        rdir = 'scp_240603-182214'
        s, u, env = pp.deserialize_sim(rdir)
        pfig = pp.plot_states(env, s, u, mode='strided', step_count=5)
        pfig.savefig(os.path.join(rdir,'slices.png'))

    def test_fr_time_plots_scp_aerial_fastw_slowdp_flipped(self):
        rdir = 'scp_240603-155025'
        rdir = 'scp_240603-180508'
        rdir = 'scp_240603-181411'
        rdir = 'scp_240603-182214'
        pfig = time_plots(rdir)
        pfig.savefig(os.path.join(rdir, 'time.png'))



    #### first spot scp
    def test_fr_scp_spot_fastw_slowdp(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Spot
        e.k_w = 0.2
        e.d_p = 0.3
        e.T = 10.0 + 2 * e.dt
        se = scp_pest.SCPEnv()
        se.rho = 0.5
        se.w_wt = 1e-2
        scp_pest.do_scp(e, se)


    def test_fr_anim_scp_spot_fastw_slowdp(self):
        rdir = 'scp_240603-164825'
        rdir = 'scp_240603-183047'
        pp.animate_sim(rdir)

    def test_fr_plot_scp_spot_fastw_slowdp(self):
        rdir = 'scp_240603-164825'
        rdir = 'scp_240603-183047'
        s, u, env = pp.deserialize_sim(rdir)
        pfig = pp.plot_states(env, s, u, mode='strided', step_count=5)
        pfig.savefig(os.path.join(rdir,'slices.png'))

    def test_fr_time_plots_scp_spot_fastw_slowdp(self):
        rdir = 'scp_240603-164825'
        rdir = 'scp_240603-183047'
        pfig = time_plots(rdir)
        pfig.savefig(os.path.join(rdir, 'time.png'))

    #### second spot scp - flip P/Q weights
    def test_fr_scp_spot_fastw_slowdp_flipped(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Spot
        e.k_w = 0.2
        e.d_p = 0.3
        e.T = 10 + 2 * e.dt
        se = scp_pest.SCPEnv()
        se.rho = 0.5
        se.P_wt = 1e2
        se.Q_wt = 1e-1
        se.w_wt = 1e-2
        scp_pest.do_scp(e, se)

    def test_fr_anim_scp_spot_fastw_slowdp_flipped(self):
        rdir = 'scp_240603-165736'
        rdir = 'scp_240603-183633'
        pp.animate_sim(rdir)

    def test_fr_plot_scp_spot_fastw_slowdp_flipped(self):
        rdir = 'scp_240603-165736'
        rdir = 'scp_240603-183633'
        s, u, env = pp.deserialize_sim(rdir)
        pfig = pp.plot_states(env, s, u, mode='strided', step_count=5)
        pfig.savefig(os.path.join(rdir,'slices.png'))

    def test_fr_time_plots_scp_spot_fastw_slowdp_flipped(self):
        rdir = 'scp_240603-165736'
        rdir = 'scp_240603-183633'
        pfig = time_plots(rdir)
        pfig.savefig(os.path.join(rdir, 'time.png'))

    def test_collect_final_data(self):
        # collect final raw and yield-adjusted tables
        # aerial
        rdir = 'scp_240603-181903'
        collect_data(rdir, "aerial, Q heavy")
        # aerial, N heavy
        rdir = 'scp_240603-182214'
        collect_data(rdir, "aerial, P heavy")
        # spot
        rdir = 'scp_240603-183047'
        collect_data(rdir, "spot, Q heavy")
        # spot, N heavy
        rdir = 'scp_240603-183633'
        collect_data(rdir, "spot, P heavy")


if __name__ == '__main__':
    unittest.main()
