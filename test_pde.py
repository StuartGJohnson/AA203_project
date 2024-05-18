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
        s,u = pp.init_state(e)
        self.assertEqual(s.shape,(256*3,))

    def test_dict(self):
        d = {'a':1,'b':2,'c':3,'d':4,'e':5}
        dlist = [d[k] for k in list(d.keys())]
        print(dlist)
        print(list(d.keys()))

    def test_simulate(self):
        e = pp.Env()
        e.bc = pp.BC.Neumann
        ps = pp.PestSim(e)
        s, u = ps.simulate()
        np.save('pest_s.npy', s)
        np.save('pest_u.npy', u)

    def test_plot_simulate(self):
        e = pp.Env()
        s = np.load('pest_s.npy')
        u = np.load('pest_u.npy')
        pp.plot_states(e, s, u)

    def test_pmask(self):
        e = pp.Env()
        pm = pp.build_p_mask(e.n)
        print(pm)
        L = pp.build_fd_lap_matrix(e.n, pp.BC.Dirichlet)
        # this should be zero
        print(L@(pm-1))

    def test_scp(self):
        e = pp.Env()
        e.n = 5
        scp_pest.do_scp(e)

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
