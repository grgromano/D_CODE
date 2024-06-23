import sys
from pathlib import Path
parent_dir = Path('..').resolve() 
sys.path.append(str(parent_dir)) 

import sympy
import argparse
import numpy as np

import equations
import data
# from dsr_utils import run_dsr
from gp_utils import run_gp
from interpolate import num_diff, num_diff_gp
import pickle
import os
import time

# # set up ODE config
# ode_param = None
# x_id = 0
#
# # data generation config
# freq = 10
# n_sample = 100
# noise_sigma = 0.0
#
# # set up algorithm config
# alg = 'gp'

def run(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed):
    np.random.seed(999)
    #print(freq)

    # specify ODE:
    ode = equations.get_ode(ode_name, ode_param)
    T = ode.T
    init_low = ode.init_low
    print(init_low)
    init_high = ode.init_high
    print(init_high)
    has_coef = ode.has_coef

    noise_sigma = ode.std_base * noise_ratio

    # data simulation: 
    dg = data.DataGenerator(ode, T, freq, n_sample, noise_sigma, init_low, init_high) # RMK. non usiamo il seed per generare i dati!!
    yt = dg.generate_data()
    print(np.shape(yt)) # dimensions: (41, 50, 1/2)
    print(yt[0:20, 0, 0]) # print first sample, variable x 
    #print(yt[0:20, 0, 1]) # print first sample, variable a -> given its definiton, a is constant over time
    #print(yt[0:20, 0, 2])
    #print(yt[0:20, 0, 3])
    # RMK. sono 50 traiettorie, ciascuna campionata con 41 osservazioni temporali delle componenti x e a
    


    # numerical differentiation:
    if noise_sigma == 0:
        dxdt_hat = (yt[1:, :, :] - yt[:-1, :, :]) / (dg.solver.t[1:] - dg.solver.t[:-1])[:, None, None]
    elif alg != 'gp':
        dxdt_hat = num_diff(yt, dg, alg)
    else:
        dxdt_hat, xt_hat = num_diff_gp(yt, dg, ode)

    print('Numerical differentiation: Done.')

    # build dataset:
    X_train = yt[:-1, :, :]
    X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])

    y_train = dxdt_hat[:, :, x_id].flatten()
    assert X_train.shape[0] == y_train.shape[0]

    #print(np.shape(X_train))
    #print(np.shape(y_train)) # RMK. y_train contiene i dati sulle derivate, che non vengono utilizzati da D-CODE
    #print(X_train[:, 0])
    #print(X_train[:, 1])

    # dimensions: X_train: (2000, 1/2), y_train: (2000,)  
    # in questo caso, i dati relativi allo stato e la derivata dei 50 campioni sono unificati in due unici array, risp. X_train e y_train
    # RMK. a differenza del caso D-CODE, in questo caso non viene perfromato lo smoothing dei dati

    # build repository for storing results:
    if alg == 'tv':
        path_base = '../results/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    elif alg == 'gp':
        path_base = '../results_gp/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else:
        path_base = '../results_spline/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)

    if not os.path.isdir(path_base):
        os.makedirs(path_base)


    for s in range(seed, seed+n_seed): # facciamo n_seed applicazioni del metodo, ciascuna con un seed differente
        
        print('Running with seed {}'.format(s))
        if x_id == 0:
            path = path_base + 'grad_seed_{}.pkl'.format(s)
        else:
            path = path_base + 'grad_x_{}_seed_{}.pkl'.format(x_id, s)

        if os.path.isfile(path): # verifica se un determinato percorso si riferisce a un file esistente
            print('Skipping seed {}'.format(s))
            print(' ')
            continue
        print(' ')

        start = time.time()

        # find the ODE f:
        f_hat, est_gp = run_gp(X_train, y_train, ode, x_id, s)

        # evaluate the functional form:
        f_true = ode.get_expression()[x_id]
        if not isinstance(f_true, tuple):
            correct = sympy.simplify(f_hat - f_true) == 0
        else:
            correct_list = [sympy.simplify(f_hat - f) == 0 for f in f_true]
            correct = max(correct_list) == 1

        end = time.time()


        with open(path, 'wb') as f:
            pickle.dump({
                'model': est_gp._program,
                'X_train': X_train,
                'y_train': y_train,
                'seed': s,
                'correct': correct,
                'f_hat': f_hat,
                'ode': ode,
                'noise_ratio': noise_ratio,
                'noise_sigma': noise_sigma,
                'dg': dg,
                'time': end-start,
            }, f)

        print('')
        print(f_hat)
        print(correct)
        print('')


# filter warning: -> non funziona, modifico direttamente il codice perché non ritorni il warning
#import warnings
#warnings.filterwarnings("ignore", message="WARNING - convergence to tolerance not achieved!")
#warnings.filterwarnings("default", message="WARNING - convergence to tolerance not achieved!") # to re-activate warning



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
