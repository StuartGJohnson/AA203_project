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
    plot_stats(time, u)
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
    plt.plot(time, np.cumsum(np.sum(u, axis=1)))
    plt.title('total control')
    plt.xlabel('time')
    pfig.tight_layout()
    plt.show()
    return pfig


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

    def test_fr_scp_aerial_fastw_slowdp(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Aerial
        e.k_w = 0.2
        e.d_p = 0.3
        e.T = 15
        se = scp_pest.SCPEnv()
        scp_pest.do_scp(e, se)

    def test_scp_aerial(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Aerial
        se = scp_pest.SCPEnv()
        scp_pest.do_scp(e, se)

    def test_scp_aerial_fastw(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Aerial
        e.k_w = 0.2
        se = scp_pest.SCPEnv()
        scp_pest.do_scp(e, se)

    def test_scp_spot(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Spot
        se = scp_pest.SCPEnv()
        scp_pest.do_scp(e, se)

    def test_scp_spot_fastw(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Spot
        e.k_w = 0.2
        se = scp_pest.SCPEnv()
        scp_pest.do_scp(e, se)

    def test_scp_spot_fastw_slowdp(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Spot
        e.k_w = 0.2
        e.d_p = 0.3
        se = scp_pest.SCPEnv()
        scp_pest.do_scp(e, se)

    def test_scp_aerial_fastw_slowdp(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Aerial
        e.k_w = 0.2
        e.d_p = 0.3
        se = scp_pest.SCPEnv()
        scp_pest.do_scp(e, se)


    def test_animate_scp(self):
        #rdir = 'scp_240525-223941'
        #rdir = 'scp_240526-104304'
        #rdir = 'scp_240526-111421'
        rdir = 'scp_240526-121034'
        pp.animate_sim(rdir)

    def test_plot_scp_xxx(self):
        #rdir = 'scp_240525-230011'
        #rdir = 'scp_240525-235048'
        #rdir = 'scp_240526-000742'
        #rdir = 'scp_240526-001527'
        #rdir = 'scp_240526-104304'
        #rdir = 'scp_240526-110830'
        #rdir = 'scp_240526-111421'
        #rdir = 'scp_240526-113023'
        #rdir = 'scp_240526-121034'
        #rdir = 'scp_240526-122752'
        #rdir = 'scp_240526-124526'
        #rdir = 'scp_240526-130010'
        #rdir = 'scp_240526-142214'
        #rdir = 'scp_240526-145809'
        #rdir = 'scp_240526-150344'
        #rdir = 'scp_240526-152845'
        #rdir = 'scp_240526-160118'
        #rdir = 'scp_240526-160731'
        #rdir = 'resim_240526-160731'
        #rdir = 'resim_240526-160118'
        #rdir = 'scp_240527-113133'
        #rdir = 'scp_240527-121309'
        rdir = 'scp_240529-151517'
        s, u, env = pp.deserialize_sim(rdir)
        plt.figure()
        plt.plot(np.sum(u, axis=1))
        plt.title('sum u')
        plt.show()
        plt.figure()
        plt.plot(np.cumsum(np.sum(u, axis=1)))
        plt.title('cumulative u')
        plt.show()
        plt.figure()
        plt.plot(np.median(u, axis=1), c='b')
        plt.plot(np.min(u, axis=1), c='g')
        plt.plot(np.max(u, axis=1), c='r')
        plt.title('stats u')
        plt.show()
        c,p,w = np.split(s,3, axis=1)
        plt.figure()
        plt.plot(np.median(c, axis=1))
        plt.axhline(y=0.73, color='r', linestyle='--')
        plt.plot()
        plt.title('median c')
        plt.show()
        plt.plot(np.sum(c, axis=1))
        plt.axhline(y=0.73*25, color='r', linestyle='--')
        plt.plot()
        plt.title('sum c')
        plt.show()
        plt.figure()
        plt.plot(np.sum(p, axis=1))
        plt.title('sum p')
        plt.show()
        plt.figure()
        plt.plot(np.median(p, axis=1), c='b')
        plt.plot(np.min(p, axis=1), c='g')
        plt.plot(np.max(p, axis=1), c='r')
        plt.title('stats p')
        plt.show()
        plt.figure()
        plt.plot(np.sum(w, axis=1))
        plt.title('sum w')
        plt.show()
        plt.figure()
        plt.plot(np.median(w, axis=1), c='b')
        plt.plot(np.min(w, axis=1), c='g')
        plt.plot(np.max(w, axis=1), c='r')
        plt.title('stats w')
        plt.show()


if __name__ == '__main__':
    unittest.main()
