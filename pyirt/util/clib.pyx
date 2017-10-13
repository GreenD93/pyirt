# -*-coding:utf-8-*-

import cython
import numpy as np
#import scipy.linalg.blas
#from libcpp.vector cimport vector
#cdef vector[double] vect

cdef extern from "math.h":
    double exp(double x)
    double log(double x)

def log_likelihood_2PL(double y1,
                       double y0,
                       double theta,
                       double alpha,
                       double beta,
                       double c=0.0):
    expPos = exp(alpha*theta + beta) ;
    ell =  y1*log((c+expPos)/(1.0+expPos)) + y0*log((1.0-c)/(1.0+expPos)) ;

    return ell


def log_likelihood_2PL_gradient(double y1,
                                double y0,
                                double theta,
                                double alpha,
                                double beta,
                                double c=0.0):
    # It is the gradient of the log likelihood, not the NEGATIVE log likelihood
    cdef extern from "math.h":
        double exp(double x)

    grad = np.zeros(2)

    temp = exp(beta + alpha * theta)
    beta_grad = temp /(1.0+temp) *( y1*(1.0-c)/(c+temp)- y0)
    alpha_grad = theta*beta_grad
    grad[0] = beta_grad
    grad[1] = alpha_grad
    return grad


