import sys
from pathlib import Path
parent_dir = Path('..').resolve() 
sys.path.append(str(parent_dir)) 

import sympy
import argparse
import numpy as np

import equations
import data
from gp_utils import run_gp_ode
from interpolate import get_ode_data
import pickle
import os
import time


def run(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, seed, n_seed):
    np.random.seed(999)

    # specify ODE:
    ode = equations.get_ode(ode_name, ode_param)
    T = ode.T # T = 4 è l'estremo temporale -> l'esperimento avviene nell'intervallo [0, 4]
    init_low = ode.init_low
    init_high = ode.init_high

    noise_sigma = ode.std_base * noise_ratio


    # data simulation: 
    # RMK. n_sample = 50 -> numero di traiettorie generate
    dg = data.DataGenerator(ode, T, freq, n_sample, noise_sigma, init_low, init_high)
    yt = dg.generate_data() # dimensions: (41, 50, 1)
    print(np.shape(yt))
    #print(yt)


    # smooth data:
    # ode_data contiene i dati interpolati, di dimensione (81, 50, 1), più altri dati relativi al metodo di interpolazione
    # X_ph e y_ph sono dei place_holer di dimensione (50, 1), (50,)
    # RMK. abbiamo 50 trajectories in input, ciascuna campionata in 81 istanti temporali
    # RMK. La numerosità del campione di training di D-CODE dipende da 'freq_int': 20 (in selkov_interp_config())

    ode_data, X_ph, y_ph, t_new = get_ode_data(yt, x_id, dg, ode)
    
    print(np.shape(ode_data['x_hat']))
    print(ode_data['x_hat'][0:20, 0, 0]) # print first trajectory
    #print(ode_data['x_hat'][0:20, 0, 1]) # print first trajectory
    #print(ode_data['x_hat'][0:20, 0, 2])
    #print(ode_data['x_hat'][0:20, 0, 3])
    #ode_data = {
    #    'x_hat': X_sample, # smoothed data, dimensions: (81, 50, 1)
    #    'g': g, # basis functions, dimensions: (81, 50) -> testing functions
    #    'c': c, # coefficients, dimensions: (50, 50)
    #    'weights': weight # weights, dimensions: (81,)
    #}
    #X_ph = np.zeros((X_sample.shape[1], X_sample.shape[2])) # placeholder, dimensions: (50, 1)
    #y_ph = np.zeros(X_sample.shape[1]) # placeholder, dimensions: (50,)


    # build repository for storing results:
    path_base = '../results_vi/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    if not os.path.isdir(path_base):
        os.makedirs(path_base)


    for s in range(seed, seed+n_seed): # facciamo n_seed applicazioni del metodo, ciascuna con un sotto_seed differente
        
        print('Running with seed {}'.format(s))
        if x_id == 0:
            path = path_base + 'grad_seed_{}.pkl'.format(s)
        else:
            path = path_base + 'grad_x_{}_seed_{}.pkl'.format(x_id, s)
        if os.path.isfile(path):
            print('Skipping seed {}'.format(s))
            print(' ')
            continue
        print(' ')

        start = time.time()

        # find the ODE f:
        f_hat, est_gp = run_gp_ode(ode_data, X_ph, y_ph, ode, x_id, s)

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
                'ode_data': ode_data,
                'seed': s,
                'correct': correct,
                'f_hat': f_hat,
                'ode': ode,
                'noise_ratio': noise_ratio,
                'noise_sigma': noise_sigma,
                'dg': dg,
                't_new': t_new,
                'time': end - start,
            }, f)

        print('')
        print(f_hat)
        print(correct)
        print('')


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
