import sys
from pathlib import Path
parent_dir = Path('..').resolve() 
sys.path.append(str(parent_dir)) 

import sympy
import argparse
import numpy as np
import pickle
import os
import time

from . import equations
from . import data
from .gp_utils import run_gp
from .interpolate import num_diff, num_diff_gp



def run(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed, T0, T, idx=0):

    '''
    run SR-T and return the building blocks
    RMK. questa versione funziona per un unico seed, per estenderla bisogna allargare la libreria candidata ad ogni seed, controllando però di non inserire doppioni tramite una funzione ausiliaria tipo get_building_blocks(est_gp, ode)
    '''

    np.random.seed(999)

    # specify ODE:
    ode = equations.get_ode(ode_name, ode_param)
    #T = ode.T
    init_low = ode.init_low
    init_high = ode.init_high
    has_coef = ode.has_coef
    noise_sigma = ode.std_base * noise_ratio


    # data simulation: 
    dg = data.DataGenerator(ode, T, freq, n_sample, noise_sigma, init_low, init_high) 
    yt = dg.generate_data()

    if T0: # if T0>0, cut portion [0,T0] 
        yt = yt[T0*freq:, :, :]
    print('Dataset shape: ', np.shape(yt))
    #print('Initial time instant: ', yt[0, 0, 2])
    #print(yt[0:20, 0, 0]) # first trajectory


    # numerical differentiation:
    if noise_sigma == 0:
        dxdt_hat = (yt[1:, :, :] - yt[:-1, :, :]) / (dg.solver.t[1:] - dg.solver.t[:-1])[:, None, None]
    elif alg != 'gp':
        dxdt_hat = num_diff(yt, dg, alg, T0)
    else:
        dxdt_hat, xt_hat = num_diff_gp(yt, dg, ode)
    #print('Numerical differentiation: Done.')


    # build dataset:
    X_train = yt[:-1, :, :]
    X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])
    y_train = dxdt_hat[:, :, x_id].flatten()
    assert X_train.shape[0] == y_train.shape[0]


    # run SR-T:
    for s in range(seed, seed+n_seed): 

        # find the ODE f:
        f_hat, est_gp, building_blocks_lambda, function_names = run_gp(X_train, y_train, ode, x_id, s, idx=idx)

    
    return building_blocks_lambda, function_names 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ode_name", help="name of the ode", type=str)
    parser.add_argument("--ode_param", help="parameters of the ode (default: None)", type=str, default=None)
    parser.add_argument("--x_id", help="ID of the equation to be learned", type=int, default=0) # RMK. in n-dimensional cases, we have a multiple equations system to be learned. D-CODE learns one equation at a time, identified by x_id
    parser.add_argument("--freq", help="sampling frequency", type=float, default=10)
    parser.add_argument("--n_sample", help="number of trajectories", type=int, default=100)
    parser.add_argument("--noise_sigma", help="noise level (default 0)", type=float, default=0.)
    parser.add_argument("--alg", help="name of the benchmark", type=str, default='tv', choices=['tv', 'spline', 'gp'])
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--n_seed", help="random seed", type=int, default=10)

    args = parser.parse_args()
    print('Running with: ', args)

    if args.ode_param is not None:
        param = [float(x) for x in args.ode_param.split(',')]
    else:
        param = None
    if args.freq >= 1:
        freq = int(args.freq)
    else:
        freq = args.freq
    run(args.ode_name, param, args.x_id, freq, args.n_sample, args.noise_sigma, args.alg, seed=args.seed, n_seed=args.n_seed)
