import unittest
import pest_pde as pp
import numpy as np
import matplotlib.pyplot as plt
import scp_pest
import os, io, json, dataclasses, jsons
import inspect

class MyTestCase(unittest.TestCase):
    def test_pde_mat(self):
        tmp = pp.build_fd_lap_matrix(5, pp.BC.Dirichlet)
        plt.figure()
        plt.imshow(tmp)
        plt.colorbar()
        plt.show()
        #self.assertEqual(True, False)  # add assertion here

    def test_pde_mat2(self):
        tmp = pp.build_fd_lap_matrix(5, pp.BC.Neumann)
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

    def test_deserialize(self):
        rdir = 'sim_240525-135920'
        s, u, sim_env = pp.deserialize_sim(rdir)
        print(sim_env)

    def test_animate_simulate(self):
        rdir = 'sim_240525-135920'
        pp.animate_sim(rdir)

    def test_deserialize2(self):
        rdir = 'sim_240525-135940'
        s, u, sim_env = pp.deserialize_sim(rdir)
        print(sim_env)

    def test_animate_simulate2(self):
        rdir = 'sim_240525-135940'
        pp.animate_sim(rdir)

    def test_deserialize3(self):
        rdir = 'sim_240525-143439'
        s, u, sim_env = pp.deserialize_sim(rdir)
        print(sim_env)

    def test_animate_simulate3(self):
        rdir = 'sim_240525-143439'
        pp.animate_sim(rdir)

    def test_animate_simulate4(self):
        rdir = 'sim_240525-143901'
        pp.animate_sim(rdir)

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
        scp_pest.do_scp(e)

    def test_scp_spot(self):
        e = pp.Env()
        e.n = 5
        e.u_mode = pp.ControlMode.Spot
        scp_pest.do_scp(e)

    def test_animate_scp(self):
        rdir = 'scp_240525-223941'
        pp.animate_sim(rdir)

    def test_plot_scp_xxx(self):
        #rdir = 'scp_240525-230011'
        rdir = 'scp_240525-235048'
        s, u, env = pp.deserialize_sim(rdir)
        plt.figure()
        plt.plot(np.median(u, axis=1))
        plt.title('u')
        plt.show()
        c,p,w = np.split(s,3, axis=1)
        plt.figure()
        plt.plot(np.sum(c, axis=1))
        plt.title('c')
        plt.show()
        plt.figure()
        plt.plot(np.sum(p, axis=1))
        plt.title('p')
        plt.show()
        plt.figure()
        plt.plot(np.sum(w, axis=1))
        plt.title('w')
        plt.show()

    def test_plot_scp(self):
        e = pp.Env()
        e.n = 5
        s = np.load('scp_n5/scp_pest_s.npy')
        u = np.load('scp_n5/scp_pest_u.npy')
        pp.plot_states(e, s, u)

    def test_plot_scp2(self):
        e = pp.Env()
        e.n = 5
        s = np.load('scp_n5_linux1/scp_pest_s.npy')
        u = np.load('scp_n5_linux1/scp_pest_u.npy')
        pp.plot_states(e, s, u)

    def test_plot_scpx(self):
        e = pp.Env()
        e.n = 5
        s = np.load('scp_240517-182839/scp_pest_s.npy')
        u = np.load('scp_240517-182839/scp_pest_u.npy')
        pp.plot_states(e, s, u)

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

if __name__ == '__main__':
    unittest.main()
