import unittest
import pest_pde as pp
import numpy as np
import matplotlib.pyplot as plt
import scp_pest
import os, io, json, jsons
from dataclasses import dataclass, make_dataclass, asdict
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
    time_u = env.dt * np.arange(0, len(u))
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
    # the ampersands are for a latex table!
    print(f"raw: {csum[-1]:.2f} & {psum[-1]:.2f} & {wsum[-1]:.2f} & {ucumsum[-1]:.2f}")
    print(f"norm: {csum[-1]/csum[-1]:.2f} & {psum[-1]/csum[-1]:.2f} & {wsum[-1]/csum[-1]:.2f} & {ucumsum[-1]/csum[-1]:.2f}")


def record_rdirs(record_file, dir_list):
    file_name = record_file
    with io.open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(jsons.dump(dir_list), ensure_ascii=False))


def run_scp(control_mode: pp.ControlMode, record_file):
    e = pp.Env()
    e.n = 5
    e.u_mode = control_mode
    e.k_w = 0.2
    e.d_p = 0.15
    se = scp_pest.SCPEnv()
    se.solver = scp_pest.cvx.MOSEK
    se.objective = scp_pest.OptimizationObjective.Convex
    se.rho = 0.5
    se.P_wt = 1e2
    se.Q_wt = 1e-1
    se.R_wt = 1e-3
    se.w_wt = 1e-2
    se.eps = .05
    rdir = scp_pest.do_scp(e, se)
    pp.animate_sim(rdir)
    s, u, env = pp.deserialize_sim(rdir)
    pfig = pp.plot_states(env, s, u, mode='strided', step_count=5)
    pfig.savefig(os.path.join(rdir, 'slices.png'))
    pfig = time_plots(rdir)
    pfig.savefig(os.path.join(rdir, 'time.png'))
    record_rdirs(record_file,[rdir])



class MyTestCase2(unittest.TestCase):

    def test_no_control(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Aerial
        e.k_w = 0.2
        e.d_p = 0.15
        e.T = 30
        ps = pp.PestSim(e)
        s, u = ps.simulate()
        rdir = pp.serialize_sim(s, u, ps)
        pp.animate_sim(rdir)
        pfig = pp.plot_states(e, s, u, mode='strided', step_count=5)
        pfig.savefig(os.path.join(rdir, 'slices.png'))
        pfig = time_plots(rdir)
        pfig.savefig(os.path.join(rdir, 'time.png'))
        record_rdirs('final_report_no_control.json', [rdir])
        # rdir for original final reports
        # "sim_240604-091905"

    def test_combo_control(self):
        run_scp(pp.ControlMode.Aerial, 'control_aerial_final_report.json')
        run_scp(pp.ControlMode.Spot, 'control_spot_final_report.json')

    def test_collect_table(self):
        # see the relevant json files (or use pest_utils.read_json_report)
        # original report
        rdir = "scp_240604-091225"
        # updated code:
        #rdir = "scp_240719-171420"
        collect_data(rdir, "aerial")
        # original report
        rdir = "scp_240604-085334"
        # updated code:
        #rdir = "scp_240719-171513"
        collect_data(rdir, "spot")
        # note these results are a bit different than the original code
        # for the final report.


if __name__ == '__main__':
    unittest.main()
