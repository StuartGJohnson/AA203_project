import unittest
import pest_pde as pp
import numpy as np
import matplotlib.pyplot as plt
import scp_pest
import os, io, json, jsons
from dataclasses import dataclass, make_dataclass, asdict
import inspect
import typing
import pest_utils as pu

def run_scp(n, d_p, control_mode: pp.ControlMode, record_file: str):
    # wrapper for do_scp - no adaptive trust region
    e = pp.Env()
    e.n = n
    e.u_mode = control_mode
    e.k_w = 0.2
    e.d_p = d_p
    se = scp_pest.SCPEnv()
    se.solver = scp_pest.cvx.MOSEK
    se.objective = scp_pest.OptimizationObjective.Convex
    se.rho = 0.1
    se.P_wt = 1e2
    se.Q_wt = 1e-1
    se.R_wt = 1e-3
    se.w_wt = 1e-2
    rdir = scp_pest.do_scp(e, se)
    pp.animate_sim(rdir)
    s, u, env = pp.deserialize_sim(rdir)
    pfig = pp.plot_states(e, s, u, mode='strided', step_count=5)
    pfig.savefig(os.path.join(rdir, 'slices.png'))
    pfig = pu.time_plots(rdir)
    pfig.savefig(os.path.join(rdir, 'time.png'))
    # now re-simulate
    ps = pp.PestSim(env)
    s_out, u_out = ps.resimulate(s, u)
    rdir_resim = rdir.replace('scp', 'resim')
    pp.serialize_sim(s_out, u_out, ps, override_dir=rdir_resim)
    pp.animate_sim(rdir_resim)
    pfig = pp.plot_states(env, s_out, u_out, mode='strided', step_count=5)
    pfig.savefig(os.path.join(rdir_resim, 'slices.png'))
    pfig = pu.time_plots(rdir_resim)
    pfig.savefig(os.path.join(rdir_resim, 'time.png'))
    dir_list = [rdir, rdir_resim]
    file_name = record_file
    with io.open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(jsons.dump(dir_list), ensure_ascii=False))


def run_scp_plus(n, d_p, control_mode: pp.ControlMode, record_file: str):
    # wrapper for do_scp_plus - adaptive trust region
    e = pp.Env()
    e.n = n
    e.u_mode = control_mode
    e.k_w = 0.2
    e.d_p = d_p
    se = scp_pest.SCPEnv()
    se.solver = scp_pest.cvx.MOSEK
    se.objective = scp_pest.OptimizationObjective.Convex
    se.rho = 0.5
    se.P_wt = 1e2
    se.Q_wt = 1e-1
    se.R_wt = 1e-3
    se.w_wt = 1e-2
    rdir = scp_pest.do_scp_plus(e, se)
    print(rdir)
    # rdir = "scp_240701-131144"
    pp.animate_sim(rdir)
    s, u, env, J, J_ref, lin_ratio, rho, scp_time, scp_env, _, _ = scp_pest.deserialize_scp(rdir)
    # movie snapshots
    pfig = pp.plot_states(e, s, u, mode='strided', step_count=5)
    pfig.savefig(os.path.join(rdir, 'slices.png'))
    # reduced over spatial dims
    pfig = pu.time_plots(rdir)
    pfig.savefig(os.path.join(rdir, 'time.png'))
    # plots of scp convergence and trust region shenanigans
    pfig = pu.plot_scp(scp_env, J, J_ref, lin_ratio, rho, scp_time)
    pfig.savefig(os.path.join(rdir, 'scp.png'))
    # now re-simulate
    ps = pp.PestSim(env)
    s_out, u_out = ps.resimulate(s, u)
    rdir_resim = rdir.replace('scp', 'resim')
    pp.serialize_sim(s_out, u_out, ps, override_dir=rdir_resim)
    pp.animate_sim(rdir_resim)
    pfig = pp.plot_states(env, s_out, u_out, mode='strided', step_count=5)
    pfig.savefig(os.path.join(rdir_resim, 'slices.png'))
    pfig = pu.time_plots(rdir_resim)
    pfig.savefig(os.path.join(rdir_resim, 'time.png'))
    dir_list = [rdir, rdir_resim]
    file_name = record_file
    with io.open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(jsons.dump(dir_list), ensure_ascii=False))


def run_scp_plus_plus(se: scp_pest.SCPEnv, d_p: float, control_mode: pp.ControlMode, record_file: str):
    # wrapper for do_scp_plus_plus - adaptive trust region AND spatial grid refinement
    e = pp.Env()
    e.u_mode = control_mode
    e.k_w = 0.2
    e.d_p = d_p
    e.dt = 0.05
    se.solver = scp_pest.cvx.MOSEK
    se.objective = scp_pest.OptimizationObjective.Convex
    se.rho = 0.5
    se.P_wt = 1e2
    se.Q_wt = 1e-1
    se.R_wt = 1e-3
    se.w_wt = 1e-2
    rdir = scp_pest.do_scp_plus_plus(e, se)
    print(rdir)
    # rdir = "scp_240701-131144"
    pp.animate_sim(rdir)
    # todo: load n
    #s, u, env, J, J_ref, lin_ratio, rho, scp_time, scp_env, n_spatial = scp_pest.deserialize_scp(rdir)
    s, u, env, J, J_ref, dJ_rel, lin_ratio, rho, scp_time, scp_env, n_spatial, iter_count = scp_pest.deserialize_scp(rdir)
    # movie snapshots
    pfig = pp.plot_states(env, s, u, mode='strided', step_count=5)
    pfig.savefig(os.path.join(rdir, 'slices.png'))
    # reduced over spatial dims
    pfig = pu.time_plots(rdir)
    pfig.savefig(os.path.join(rdir, 'time.png'))
    # plots of scp convergence and trust region shenanigans
    pfig = pu.plot_scp(scp_env, J, J_ref, dJ_rel, lin_ratio, rho, scp_time, n_spatial, iter_count)
    pfig.savefig(os.path.join(rdir, 'scp.png'))
    # now re-simulate
    ps = pp.PestSim(env)
    s_out, u_out = ps.resimulate(s, u)
    rdir_resim = rdir.replace('scp', 'resim')
    pp.serialize_sim(s_out, u_out, ps, override_dir=rdir_resim)
    pp.animate_sim(rdir_resim)
    pfig = pp.plot_states(env, s_out, u_out, mode='strided', step_count=5)
    pfig.savefig(os.path.join(rdir_resim, 'slices.png'))
    pfig = pu.time_plots(rdir_resim)
    pfig.savefig(os.path.join(rdir_resim, 'time.png'))
    dir_list = [rdir, rdir_resim]
    file_name = record_file
    with io.open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(jsons.dump(dir_list), ensure_ascii=False))


class MyTestCase2(unittest.TestCase):

    def test_combo_control_plus_plus_0(self):
        se = scp_pest.SCPEnv()
        # cvxpy cannot handle state spaces bigger than those
        # for n = 16 (n=20 or 1200-dim state space seems to have out-of-memory problems)
        se.max_iters_spatial = 3
        se.n_spatial_init = 8
        se.n_spatial_inc = 4
        se.n_spatial_fac = 0
        se.verbose_solver = False
        run_scp_plus_plus(se, 0.10, pp.ControlMode.Aerial, 'test_control_aerial_plus_plus_0.json')
        run_scp_plus_plus(se, 0.10, pp.ControlMode.Spot, 'test_control_spot_plus_plus_0.json')

    def test_combo_control_plus_plus_1(self):
        se = scp_pest.SCPEnv()
        # cvxpy cannot handle state spaces bigger than those
        # for n = 16 (n=20 or 1200-dim state space seems to have out-of-memory problems)
        se.max_iters_spatial = 3
        se.n_spatial_init = 8
        se.n_spatial_inc = 4
        se.n_spatial_fac = 0
        se.verbose_solver = False
        run_scp_plus_plus(se, 0.25, pp.ControlMode.Aerial, 'test_control_aerial_plus_plus_1.json')
        run_scp_plus_plus(se, 0.25, pp.ControlMode.Spot, 'test_control_spot_plus_plus_1.json')

    def test_combo_control_plus_plus_2(self):
        se = scp_pest.SCPEnv()
        # cvxpy cannot handle state spaces bigger than those
        # for n = 16 (n=20 or 1200-dim state space seems to have out-of-memory problems)
        se.max_iters_spatial = 3
        se.n_spatial_init = 8
        se.n_spatial_inc = 4
        se.n_spatial_fac = 0
        se.verbose_solver = False
        run_scp_plus_plus(se, 0.4, pp.ControlMode.Aerial, 'test_control_aerial_plus_plus_2.json')
        run_scp_plus_plus(se, 0.4, pp.ControlMode.Spot, 'test_control_spot_plus_plus_2.json')

    def test_combo_control_plus_plus_3(self):
        se = scp_pest.SCPEnv()
        # cvxpy cannot handle state spaces bigger than those
        # for n = 16 (n=20 or 1200-dim state space seems to have out-of-memory problems)
        se.max_iters_spatial = 3
        se.n_spatial_init = 8
        se.n_spatial_inc = 4
        se.n_spatial_fac = 0
        se.verbose_solver = False
        run_scp_plus_plus(se, 0.05, pp.ControlMode.Aerial, 'test_control_aerial_plus_plus_3.json')
        run_scp_plus_plus(se, 0.05, pp.ControlMode.Spot, 'test_control_spot_plus_plus_3.json')

    def test_combo_control_plus_plus_4(self):
        se = scp_pest.SCPEnv()
        # cvxpy cannot handle state spaces bigger than those
        # for n = 16 (n=20 or 1200-dim state space seems to have out-of-memory problems)
        se.max_iters_spatial = 3
        se.n_spatial_init = 8
        se.n_spatial_inc = 4
        se.n_spatial_fac = 0
        se.verbose_solver = False
        run_scp_plus_plus(se, 0.15, pp.ControlMode.Aerial, 'test_control_aerial_plus_plus_4.json')
        run_scp_plus_plus(se, 0.15, pp.ControlMode.Spot, 'test_control_spot_plus_plus_4.json')

    def test_collect_table_data(self):
        rdir = pu.read_json_report('test_control_aerial_plus_plus_3.json')[0]
        pu.collect_data(rdir, "aerial, dp=0.05")
        rdir = pu.read_json_report('test_control_spot_plus_plus_3.json')[0]
        pu.collect_data(rdir, "spot, dp=0.05")

        rdir = pu.read_json_report('test_control_aerial_plus_plus_0.json')[0]
        pu.collect_data(rdir, "aerial, dp=0.1")
        rdir = pu.read_json_report('test_control_spot_plus_plus_0.json')[0]
        pu.collect_data(rdir, "spot, dp=0.1")

        rdir = pu.read_json_report('test_control_aerial_plus_plus_4.json')[0]
        pu.collect_data(rdir, "aerial, dp=0.15")
        rdir = pu.read_json_report('test_control_spot_plus_plus_4.json')[0]
        pu.collect_data(rdir, "spot, dp=0.15")

        rdir = pu.read_json_report('test_control_aerial_plus_plus_1.json')[0]
        pu.collect_data(rdir, "aerial, dp=0.25")
        rdir = pu.read_json_report('test_control_spot_plus_plus_1.json')[0]
        pu.collect_data(rdir, "spot, dp=0.25")

        rdir = pu.read_json_report('test_control_aerial_plus_plus_2.json')[0]
        pu.collect_data(rdir, "aerial, dp=0.4")
        rdir = pu.read_json_report('test_control_spot_plus_plus_2.json')[0]
        pu.collect_data(rdir, "spot, dp=0.4")

if __name__ == '__main__':
    unittest.main()
