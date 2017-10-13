# -*- coding:utf-8 -*-
import numpy as np

def irt_fnc(theta, beta, alpha=1.0, c=0.0):
    # beta is item difficulty
    # theta is respondent capability

    prob = c + (1.0 - c) / (1 + np.exp(-(alpha * theta + beta)))
    return prob


def log_likelihood_factor_gradient(y1, y0, theta, alpha, beta, c=0.0):
    temp = np.exp(beta + alpha * theta)
    grad = alpha * temp / (1.0 + temp) * (y1 * (1.0 - c) / (c + temp ) - y0 )

    return grad

# vectorize
def core_exp(param, theta):
    return np.exp(-np.dot(param, theta))

def irt_vec(param, theta, c=0.0):
    z = core_exp(param, theta) 
    f = c+(1-c)/(1+z)
    return f

def llk_vec(y, param, theta, c=0.0):
    f = irt_vec(param, theta, c)
    llk = y*np.log(f)+(1-y)*np.log(1-f)
    return llk

def llk_grad_vec(y, param, theta, grad_type, c=0.0):
    # partial on core
    z = core_exp(param, theta)
    if grad_type=="param":
        grad_core = -z*theta
    else:
        grad_core = -z*param
    
    prob = irt_vec(param, theta, c)
    grad_prob = -(1-c)/(1+z**2)*grad_core
    grad = (y/prob - (1-y)/(1-prob))*grad_prob
    return grad 



def logsum(logp):
    w = max(logp)
    logSump = w + np.log(sum(np.exp(logp - w)))
    return logSump



def cut_list(list_length, num_chunk):
    chunk_bnd = [0]
    for i in range(num_chunk):
        chunk_bnd.append(int(list_length*(i+1)/num_chunk))
    chunk_bnd.append(list_length)
    chunk_list = [(chunk_bnd[i], chunk_bnd[i+1]) for i in range(num_chunk) ]
    return chunk_list
