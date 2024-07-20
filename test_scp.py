import unittest
import scp_pest, pest_utils
import numpy as np
import os, io, json, jsons

class MyTestCase(unittest.TestCase):
    def test_sequence_inc(self):
        # test a sequence of n's for the outer loop of SCP
        se = scp_pest.SCPEnv()
        se.n_spatial_fac = 0
        se.n_spatial_inc = 2
        se.n_spatial_init = 4
        n_vec = scp_pest.n_spatial_vec(se)
        print(n_vec)
        self.assertEqual(n_vec, [4, 6, 8, 10])

    def test_sequence_fac(self):
        # test a sequence of n's for the outer loop of SCP
        se = scp_pest.SCPEnv()
        se.n_spatial_fac = 3
        se.n_spatial_inc = 0
        se.n_spatial_init = 3
        n_vec = scp_pest.n_spatial_vec(se)
        print(n_vec)
        self.assertEqual(n_vec, [3, 9, 27, 81])

    def test_read_json_report(self):
        print(pest_utils.read_json_report("test_control_aerial_plus_plus_3.json"))



if __name__ == '__main__':
    unittest.main()
