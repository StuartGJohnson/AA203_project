"""
New checks after introduce proper domain scaling/units. This is essential to
make sure problem is not fundamentally changing with spatial resolution.
"""
import unittest
import pest_pde as pp
import pest_utils as pu
import numpy as np
import matplotlib.pyplot as plt
import scp_pest
import os, io, json, dataclasses, jsons
import inspect
import typing
import cv2
import copy
import pandas as pd



class MyTestCase(unittest.TestCase):
    def test_base_sim(self):
        e = pp.Env()
        e.n = 10
        e.u_mode = pp.ControlMode.Spot
        e.k_w = 0.2
        e.d_p = 0.15
        ps = pp.PestSim(e)
        s, u = ps.simulate()
        sim_dir = pp.serialize_sim(s, u, ps)
        # animation
        pp.animate_sim(sim_dir)
        # slices
        pfig = pp.plot_states(e, s, u, mode='strided', step_count=5)
        pfig.savefig(os.path.join(sim_dir, 'slices.png'))
        # time series
        pfig = pu.time_plots(sim_dir)
        pfig.savefig(os.path.join(sim_dir, 'time.png'))

    def test_upscale(self):
        tmp = np.array([[0,1,0],[1,0,1]])
        array = tmp.repeat(2, axis=0).repeat(2, axis=1)
        print(array)

    def test_umat(self):
        # check the underactuated to full u matrix
        e = pp.Env()
        e.n = 8
        e.spot_resolution = 4
        u_pat = pp.build_u_pattern(e)
        print(u_pat)
        print(np.reshape(u_pat @ np.array([1,0,0,0]), (e.n, e.n)))
        print(np.reshape(u_pat @ np.array([0,1,0,0]), (e.n, e.n)))

    def test_downsample(self):
        # reducing array resolution
        p = np.random.rand(10, 10)
        p2 = p + 0.1
        pr = cv2.resize(p, (2, 2), interpolation=cv2.INTER_AREA)
        pr2 = cv2.resize(p2, (2, 2), interpolation=cv2.INTER_AREA)
        print(pr)
        print(pr2)
        print(np.max(np.abs(pr2-pr)))

    def test_upsample(self):
        # increasing array resolution - handy for warm-starting high res SCP
        p = np.random.rand(10, 10)
        p = cv2.GaussianBlur(p, (5, 5), sigmaX=5, sigmaY=5)
        # first, downsample
        p_down = cv2.resize(p, (7, 7), interpolation=cv2.INTER_AREA)
        # upsample to original
        p_down_up = cv2.resize(p_down, (10, 10), interpolation=cv2.INTER_LINEAR)
        print(np.median(np.abs(p_down_up-p)/np.abs(p)))
        plt.imshow(p_down_up)
        plt.title('p_down_up')
        plt.colorbar()
        plt.show()
        plt.imshow(p_down)
        plt.title('p_down')
        plt.colorbar()
        plt.show()
        plt.imshow(p)
        plt.title('p')
        plt.colorbar()
        plt.show()
        plt.imshow(np.abs(p_down_up - p)/np.abs(p))
        plt.title('diff')
        plt.colorbar()
        plt.show()


    def test_json_list(self):
        list = ['a','b','c','d']
        file_env = 'test_list.json'
        with io.open(file_env, 'w', encoding='utf-8') as f:
            f.write(json.dumps(jsons.dump(list), ensure_ascii=False))
        with io.open(file_env, 'r', encoding='utf-8') as f:
            list2 = json.load(f)
        print(list)
        print(list2)
        self.assertListEqual(list, list2)

    def test_convergence(self):
        # run through a set of simulations to check grid dimensions
        # note the first dim is the reference dim
        # also save the list of rdirs for future reference
        dim_list = [32, 28, 24, 20, 16, 12, 8, 4]
        # odeint solves the problem of convergence in time,
        # so I am not going to worry about this (I did check it)
        dt_list = [0.1]
        # run settings - note that these should
        # be the pde settings used in the sims
        e = pp.Env()
        e.u_mode = pp.ControlMode.Spot
        e.k_w = 0.2
        #e.d_p = 0.15
        e.d_p = 0.2
        sim_dirs = []
        err_table = []
        err_tablec = []
        err_tablep = []
        err_tablew = []
        first_pass = True
        for d in dt_list:
            for dim in dim_list:
                e.n = dim
                e.dt = d
                ps = pp.PestSim(e)
                s, u = ps.simulate()
                if first_pass:
                    s_ref = np.copy(s)
                    u_ref = np.copy(u)
                    e_ref = copy.deepcopy(e)
                    first_pass = False
                sim_dir = pp.serialize_sim(s, u, ps)
                sim_dirs.append(sim_dir)
                dt_ref, dt, n_ref, n, c_max, c_err_max, c_rel, p_max, p_err_max, p_rel, w_max, w_err_max, w_rel =\
                    pu.compute_pde_error(e_ref, s_ref, u_ref, e, s, u)
                err_table.append([n_ref, n, c_max, c_err_max, c_rel, p_max, p_err_max, p_rel, w_max, w_err_max, w_rel])
                err_tablec.append([n_ref, n, c_max, c_err_max, c_rel])
                err_tablep.append([n_ref, n, p_max, p_err_max, p_rel])
                err_tablew.append([n_ref, n, w_max, w_err_max, w_rel])
        file_name = 'convergence_test2.json'
        with io.open(file_name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(jsons.dump(sim_dirs), ensure_ascii=False))
        df = pd.DataFrame(err_table,
                          columns=('n_ref', 'n', 'c_max', 'c_err_max', 'c_rel', 'p_max', 'p_err_max', 'p_rel', 'w_max', 'w_err_max', 'w_rel'))
        df.to_excel('convergence_test2' + '.xlsx', index=False)
        dfc = pd.DataFrame(err_tablec,
                          columns=(r'$n_{ref}$', r'n', r'$\| c_{ref} \|_{\infty}$', r'$\| c-c_{ref} \|_{\infty}$',
                                   r'$\frac{\| c-c_{ref} \|_{\infty}}{\| c_{ref} \|_{\infty}}$'))
        dfc.to_latex('convergence_test_c2' + '.tex', index=False, float_format="{:0.3f}".format)
        dfp = pd.DataFrame(err_tablep,
                          columns=(r'$n_{ref}$', r'n',
                                   r'$\| p_{ref} \|_{\infty}$', r'$\| p-p_{ref} \|_{\infty}$',
                                   r'$\frac{\| p-p_{ref} \|_{\infty}}{\| p_{ref} \|_{\infty}}$'))
        dfp.to_latex('convergence_test_p2' + '.tex', index=False, float_format="{:0.3f}".format)
        dfw = pd.DataFrame(err_tablew,
                          columns=(r'$n_{ref}$', r'n',
                                   r'$\| w_{ref} \|_{\infty}$', r'$\| w-w_{ref} \|_{\infty}$',
                                   r'$\frac{\| w-w_{ref} \|_{\infty}}{\| w_{ref} \|_{\infty}}$'))
        dfw.to_latex('convergence_test_w2' + '.tex', index=False, float_format="{:0.3f}".format)


if __name__ == '__main__':
    unittest.main()
