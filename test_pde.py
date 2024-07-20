import unittest
import pest_pde as pp
import numpy as np
import matplotlib.pyplot as plt
import scp_pest
import os, io, json, dataclasses, jsons
import inspect
import typing

class MyTestCase(unittest.TestCase):
    def test_sim(self):
        e = pp.Env()
        e.bc = pp.B
        tmp = pp.build_fd_lap_matrix(e)
        plt.figure()
        plt.imshow(tmp)
        plt.colorbar()
        plt.show()
        #self.assertEqual(True, False)  # add assertion here

    def test_pde_mat2(self):
        e = pp.Env()
        e.bc = pp.BC.Neumann
        tmp = pp.build_fd_lap_matrix(e)
        plt.figure()
        plt.imshow(tmp)
        plt.colorbar()
        plt.show()

    def test_diag(self):
        n=5
        d = np.ones((n,))
        d1 = np.ones((n-1,))
        a = np.diag(-3 * d, k=0)
        b = np.diag(d1, k=1)
        t_mat_ext = np.diag(-3*d,k=0) + np.diag(d1,k=1) + np.diag(d1, k=-1)

    def test_np_unpack(self):
        tmp = np.random.rand(5, 1)
        a,b,c,d,e = tmp
        print(a,b,c,d,e)

    def test_np_index(self):
        tmp = np.array(np.arange(0,30,1))
        tmp2 = tmp * tmp
        print(tmp[0:5])
        print(tmp[0+5:5+5])
        print(tmp2[0:5])
        tmp_new = np.concatenate([tmp[0:5],tmp[0+5:5+5],tmp2[0:5]])
        print(tmp_new)

    def test_np_reshape(self):
        tmp = np.random.rand(3, 3)
        tmp2 = tmp.reshape(9,)
        print(tmp)
        print(tmp2)
        print(16//2)

    def test_init(self):
        e = pp.Env()
        s,u,b = pp.init_state(e)
        self.assertEqual(s.shape,(e.n**2*3,))
        self.assertEqual(b.shape,(e.n**2,))
        self.assertEqual(u.shape, (e.n ** 2,))
        bp = np.reshape(b, (e.n, e.n))
        plt.figure()
        plt.imshow(bp, origin='lower')
        plt.show()

    def test_init2(self):
        e = pp.Env()
        e.u_mode = pp.ControlMode.Aerial
        s,u,b = pp.init_state(e)
        self.assertEqual(s.shape,(e.n**2*3,))
        self.assertEqual(b.shape,(e.n**2,))
        self.assertEqual(u, e.u0)
        bp = np.reshape(b, (e.n, e.n))
        plt.figure()
        plt.imshow(bp, origin='lower')
        plt.show()

    def test_dict(self):
        d = {'a':1,'b':2,'c':3,'d':4,'e':5}
        dlist = [d[k] for k in list(d.keys())]
        print(dlist)
        print(list(d.keys()))

    def test_simulate(self):
        e = pp.Env()
        ps = pp.PestSim(e)
        s, u = ps.simulate()
        pp.serialize_sim(s, u, ps)

    def test_simulate2(self):
        e = pp.Env()
        e.u_mode = pp.ControlMode.Aerial
        ps = pp.PestSim(e)
        s, u = ps.simulate()
        pp.serialize_sim(s, u, ps)

    def test_simulate3(self):
        e = pp.Env()
        e.u_mode = pp.ControlMode.Aerial
        e.u0 = .05
        ps = pp.PestSim(e)
        s, u = ps.simulate()
        pp.serialize_sim(s, u, ps)

    def test_simulate4(self):
        e = pp.Env()
        e.u_mode = pp.ControlMode.Spot
        e.u0 = .05
        ps = pp.PestSim(e)
        s, u = ps.simulate()
        pp.serialize_sim(s, u, ps)

    def test_simulate2(self):
        e = pp.Env()
        e.u_mode = pp.ControlMode.Aerial
        #e.bc = pp.BC.Neumann
        ps = pp.PestSim(e)
        s, u = ps.simulate()
        pp.serialize_sim(s, u, ps)

    def test_plot_simulate(self):
        e = pp.Env()
        s = np.load('pest_s.npy')
        u = np.load('pest_u.npy')
        pp.plot_states(e, s, u)

    def test_plot_simulate_pests(self):
        e = pp.Env()
        s = np.load('pest_s.npy')
        #u = np.load('pest_u.npy')
        p = s[:,e.n**2:2*e.n**2]
        plt.figure()
        plt.plot(np.max(p,axis=1))
        plt.show()


    def test_pmask(self):
        e = pp.Env()
        pm = pp.build_p_mask(e.n)
        print(pm)
        L = pp.build_fd_lap_matrix(e.n, pp.BC.Dirichlet)
        # this should be zero
        print(L@(pm-1))

    def test_np_thing(self):
        tmp = np.eye(1)
        print(tmp)

    def test_np_crap(self):
        tmp = np.ones((10,1))
        m = np.eye(1)
        x = tmp[3].T @ m @ tmp[3]
        print(x)

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

    def test_strrep(self):
        rdir = 'scp_240526-160731'
        rdir2 = rdir.replace('scp', 'resim')
        print(rdir2)

    def test_numpy_quick(self):
        t = np.diag([1,2,3,4])
        print(t)

    def test_plot_scp3(self):
        e = pp.Env()
        e.n = 7
        s = np.load('scp_pest_s.npy')
        u = np.load('scp_pest_u.npy')
        pp.plot_states(e, s, u)

    def test_plot_scp_report(self):
        e = pp.Env()
        e.n = 7
        s = np.load('scp_n7_ECOS_linux/scp_pest_s.npy')
        u = np.load('scp_n7_ECOS_linux/scp_pest_u.npy')
        pp.plot_states(e, s, u)

    def test_plot_scp_report_early(self):
        e = pp.Env()
        e.n = 7
        s = np.load('scp_n7_ECOS_linux/scp_pest_s.npy')
        u = np.load('scp_n7_ECOS_linux/scp_pest_u.npy')
        pp.plot_states(e, s, u, mode='early')

    def test_plot_scp_report_movie(self):
        e = pp.Env()
        e.n = 7
        s = np.load('scp_n7_ECOS_linux/scp_pest_s.npy')
        u = np.load('scp_n7_ECOS_linux/scp_pest_u.npy')
        ani = pp.animate_states(e, s, u)
        ani.save(filename="scp_n7_ECOS_linux/scp_pest.mp4", writer="ffmpeg")

    def test_source_dumper(self):
        """ This could be handy. This code is like a mini-RCS. This
        method is a little too beefy for datalassing it's bits. Well,
        maybe."""
        # serialize do_scp
        file_do = 'test_do_scp.txt'
        with io.open(file_do, 'w', encoding='utf-8') as f:
            f.write(inspect.getsource(scp_pest.do_scp))

    def test_time_stamper(self):
        import datetime
        now = datetime.datetime.now()
        print(now.strftime("%Y%m%d-%H%M%S"))

    def test_write_dataclass(self):
        e = pp.Env()
        file_env = os.path.join('pest_pde.env.json')
        with io.open(file_env, 'w', encoding='utf-8') as f:
            f.write(json.dumps(jsons.dump(e), ensure_ascii=False))

    def test_write_dataclass2(self):
        e = scp_pest.SCPEnv()
        file_env = os.path.join('scp.env.json')
        with io.open(file_env, 'w', encoding='utf-8') as f:
            f.write(json.dumps(jsons.dump(e), ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
