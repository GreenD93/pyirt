# -*- coding: utf-8 -*-
import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)

import unittest
from pyirt.util import tools, clib 

import math
import numpy as np

tol = 1e-4
delta = 1e-5

class TestIrtFunctions(unittest.TestCase):

    def test_irt_fnc(self):
        # make sure the shuffled sequence does not lose any elements
        prob = tools.irt_fnc(0.0, 0.0, 1.0)
        self.assertEqual(prob, 0.5)
        # alpha should play no role
        prob = tools.irt_fnc(0.0, 0.0, 2.0)
        self.assertEqual(prob, 0.5)
        # higher theta should have higher prob
        prob = tools.irt_fnc(1.0, 0.0, 1.0)
        self.assertEqual(prob, 1.0 / (1.0 + math.exp(-1.0)))
        # cancel out by higher beta
        prob = tools.irt_fnc(1.0, -1.0, 1.0)
        self.assertEqual(prob, 0.5)
        # test for c as limit situation
        prob = tools.irt_fnc(-99, 0.0, 1.0, 0.25)
        self.assertTrue(abs(prob - 0.25) < tol)
        prob = tools.irt_fnc(99, 0.0, 1.0, 0.25)
        self.assertTrue(abs(prob - 1.0) < tol)

    def test_log_likelihood(self):

        # the default model, log likelihood is log(0.5)
        ll = clib.log_likelihood_2PL(1.0, 0.0, 0.0, 1.0, 0.0)
        self.assertEqual(ll, math.log(0.5))
        ll = clib.log_likelihood_2PL(0.0, 1.0, 0.0, 1.0, 0.0)
        self.assertEqual(ll, math.log(0.5))

        # check the different model
        ll = clib.log_likelihood_2PL(1.0, 0.0, 1.0, 1.0, 0.0)
        self.assertEqual(ll, math.log(1.0 / (1.0 + math.exp(-1.0))))

        ll = clib.log_likelihood_2PL(0.0, 1.0, 1.0, 1.0, 0.0)
        self.assertEqual(ll, math.log(1.0 - 1.0 / (1.0 + math.exp(-1.0))))

        # check a real value
        ll = clib.log_likelihood_2PL(0.0, 1.0, -1.1617696779178492, 1.0, 0.0)

        self.assertTrue(abs(ll + 0.27226272946920399) < 0.0000000001)

        # check if it handles c correctly
        ll = clib.log_likelihood_2PL(1.0, 0.0, 0.0, 1.0, 0.0, 0.25)
        self.assertEqual(ll, math.log(0.625))
        ll = clib.log_likelihood_2PL(0.0, 1.0, 0.0, 1.0, 0.0, 0.25)
        self.assertEqual(ll, math.log(0.375))

    def test_log_sum(self):
        # add up a list of small values
        log_prob = np.array([-135, -115, -125, -100])
        approx_sum = tools.logsum(log_prob)
        exact_sum = 0
        for num in log_prob:
            exact_sum += math.exp(num)
        exact_sum = math.log(exact_sum)
        self.assertTrue(abs(approx_sum - exact_sum) < 1e-10)

    def test_log_item_gradient(self):
        y1 = 1.0
        y0 = 2.0
        theta = -2.0
        alpha = 1.0
        beta = 0.0
        # simulate the gradient
        true_gradient_approx_beta = (clib.log_likelihood_2PL(y1, y0, theta, alpha, beta + delta) -
                                     clib.log_likelihood_2PL(y1, y0, theta, alpha, beta)) / delta
        true_gradient_approx_alpha = (clib.log_likelihood_2PL(y1, y0, theta, alpha + delta, beta) -
                                      clib.log_likelihood_2PL(y1, y0, theta, alpha, beta)) / delta
        # calculate
        calc_gradient = clib.log_likelihood_2PL_gradient(y1, y0, theta, alpha, beta)

        self.assertTrue(abs(calc_gradient[0] - true_gradient_approx_beta) < tol)
        self.assertTrue(abs(calc_gradient[1] - true_gradient_approx_alpha) < tol)

        # simulate the gradient with c
        c = 0.25
        true_gradient_approx_beta = (clib.log_likelihood_2PL(y1, y0, theta, alpha, beta + delta, c) -
                                     clib.log_likelihood_2PL(y1, y0, theta, alpha, beta, c)) / delta
        true_gradient_approx_alpha = (clib.log_likelihood_2PL(y1, y0, theta, alpha + delta, beta, c) -
                                      clib.log_likelihood_2PL(y1, y0, theta, alpha, beta, c)) / delta
        # calculate
        calc_gradient = clib.log_likelihood_2PL_gradient(y1, y0, theta, alpha, beta, c)

        self.assertTrue(abs(calc_gradient[0] - true_gradient_approx_beta) < tol)
        self.assertTrue(abs(calc_gradient[1] - true_gradient_approx_alpha) < tol)

#TODO: test for c
class TestProbVect(unittest.TestCase):
    def test_single(self):
        param = np.array([0.5,-1.0])
        theta = np.array([1,0.5])
        self.assertTrue(tools.irt_vec(param, theta)==0.5)
        
        param = np.array([0.5,-1.0])
        theta = np.array([1,1])
        get = tools.irt_vec(param, theta)
        want = 1/(1+np.exp(0.5))
        self.assertTrue(abs(want-get)<tol)
        
    def test_composite(self):
        param = np.array([0.5,-1.0,1.0])
        theta = np.array([1,0.5,1])
        get = tools.irt_vec(param, theta)
        want = 1/(1+np.exp(-1))
        self.assertTrue(abs(want-get)<tol)

class TestLlkVec(unittest.TestCase):
    def test_single(self):
        param = np.array([0.5,-1.0])
        theta = np.array([1,0.5])
        get = tools.llk_vec(1,param, theta)
        want = np.log(0.5) 
        self.assertTrue(abs(want-get)<tol)

        param = np.array([0.5,-1.0])
        theta = np.array([1,1])
        get = tools.llk_vec(1,param, theta)
        want = np.log(1/(1+np.exp(0.5)))
        self.assertTrue(abs(want-get)<tol)
        get = tools.llk_vec(0,param, theta)
        want = np.log(1-1/(1+np.exp(0.5)))
        self.assertTrue(abs(want-get)<tol)
    
    def test_composite(self):

        param = np.array([0.5,-1.0,1.0])
        theta = np.array([1,0.5,1])
        get = tools.llk_vec(1,param, theta)
        want = np.log(1/(1+np.exp(-1)))
        self.assertTrue(abs(want-get)<tol)
        get = tools.llk_vec(0,param, theta)
        want = np.log(1-1/(1+np.exp(-1)))
        self.assertTrue(abs(want-get)<tol)

class TestGradVec(unittest.TestCase):
    def test_single(self):
        param = np.array([0.5,-1.0])
        theta = np.array([1,1])
        res = tools.llk_grad_vec(1, param, theta, "param")
        print(res)
        res = tools.llk_grad_vec(1, param, theta, "theta")
        print(res)
       

class TestCutList(unittest.TestCase):
    def test_no_mod(self):
        test_chunks = tools.cut_list(100,4)
        true_chunks = [(0,25),(25,50),(50,75),(75,100)]
        for i in range(4):
            self.assertTrue(test_chunks[i]==true_chunks[i])
    
    def test_mod(self):
        test_chunks = tools.cut_list(23,4)
        true_chunks = [(0,5),(5,11),(11,17),(17,23)]
        for i in range(4):
            self.assertTrue(test_chunks[i]==true_chunks[i])
        



if __name__ == '__main__':
    unittest.main()
