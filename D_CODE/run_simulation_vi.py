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
from .gp_utils import run_gp_ode
from .interpolate import get_ode_data



def run(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, seed, n_seed, T0=0, T=15):

    '''
    run D-CODE and return the building blocks 
    RMK. questa versione funziona per un unico seed, per estenderla bisogna allargare la libreria candidata ad ogni seed, controllando però di non inserire doppioni tramite una funzione ausiliaria tipo get_building_blocks(est_gp, ode)
    '''
    
    np.random.seed(999)

    # specify ODE:
    ode = equations.get_ode(ode_name, ode_param)
    # T = ode.T 
    init_low = ode.init_low
    init_high = ode.init_high
    noise_sigma = ode.std_base * noise_ratio


    # data simulation: 
    dg = data.DataGenerator(ode, T, freq, n_sample, noise_sigma, init_low, init_high) 
    yt = dg.generate_data() # dimensions: (41, 50, 1)
    if T0: # if T0>0, cut portion [0,T0] 
        yt = yt[int(T0*freq):, :, :]
    #print(np.shape(yt))


    # smooth data:
    ode_data, X_ph, y_ph, t_new = get_ode_data(yt, x_id, dg, ode, T0=T0)
    print('Dataset shape: ', np.shape(ode_data['x_hat']))
    #print(ode_data['x_hat'][0:20, 0, 0]) # print first trajectory


    # run D-CODE:
    for s in range(seed, seed+n_seed): 

        # find the ODE f:
        f_hat, est_gp, building_blocks_lambda, function_names = run_gp_ode(ode_data, X_ph, y_ph, ode, x_id, s)


    return building_blocks_lambda, function_names 



def run_HD(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, seed, n_seed, T0=0, T=15, latent_data=None):

    '''
    High Dimensional version
    '''
    
    np.random.seed(999)

    # specify ODE:
    ode = equations.get_ode(ode_name, ode_param)
    init_low = ode.init_low
    init_high = ode.init_high
    noise_sigma = ode.std_base * noise_ratio


    # data simulation: 
    dg = data.DataGenerator(ode, T, freq, n_sample, noise_sigma, init_low, init_high)
    #yt = dg.generate_data()
    #print(np.shape(yt))


    # al posto dei dati simulati, usiamo i dati ricostruiti dall'AE
    yt = latent_data
    print(np.shape(yt))
    

    if T0: # if T0>0, cut portion [0,T0] 
        yt = yt[int(T0*freq):, :, :]


    # smooth data:
    ode_data, X_ph, y_ph, t_new = get_ode_data(yt, x_id, dg, ode, T0=T0)
    print('Dataset shape: ', np.shape(ode_data['x_hat']))
    print(ode_data['x_hat'][0:20, 0, 0]) # print first trajectory


    # run D-CODE:
    for s in range(seed, seed+n_seed): 

        # find the ODE f:
        f_hat, est_gp, building_blocks_lambda, function_names = run_gp_ode(ode_data, X_ph, y_ph, ode, x_id, s)

    
    return building_blocks_lambda, function_names 




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ode_name", help="name of the ode", type=str)
    parser.add_argument("--ode_param", help="parameters of the ode (default: None)", type=str, default=None)
    parser.add_argument("--x_id", help="ID of the equation to be learned", type=int, default=0)
    parser.add_argument("--freq", help="sampling frequency", type=float, default=10)
    parser.add_argument("--n_sample", help="number of trajectories", type=int, default=100)
    parser.add_argument("--noise_sigma", help="noise level (default 0)", type=float, default=0.)
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
    run(args.ode_name, param, args.x_id, freq, args.n_sample, args.noise_sigma, seed=args.seed, n_seed=args.n_seed)
