import argparse
import functools
import numpy as np
import equations
import data
from scipy.stats import ks_2samp
import pickle
import sys
import os
from sklearn.metrics import root_mean_squared_error

# 1D:
# Standard
def evaluation1D(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed , init_high_0, init_low_0):
    
    # seed:
    np.random.seed(666)

    # upload the results:
    seed_s = seed
    seed_e = n_seed
    if alg == 'diff':
        path_base = '../results/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else: # -> alg == 'vi':
        path_base = '../results_vi/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    
    res_list_1 = []
    for s in range(seed_s, seed_e):
        
        path_1 = path_base + 'grad_seed_{}.pkl'.format(s)

        try:
            with open(path_1, 'rb') as f_1:
                res_1 = pickle.load(f_1)
            res_list_1.append(res_1)
        except Exception:
            pass 
    print(len(res_list_1)) # check

    # percentage of correct identifications:
    correct_list_1 = [res_1['correct'] for res_1 in res_list_1] 
    p_correct_1 = np.mean(correct_list_1) 
    std_correct_1 = np.sqrt(p_correct_1 * (1 - p_correct_1) / len(correct_list_1))
    print(p_correct_1)

    # true trajectories:
    ode = equations.get_ode(ode_name, ode_param)
    dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
    xt_true = dg_true.xt
    #print(np.shape(xt_true))

    # estimate trajectories and compute RMSE:
    est_trajectories = []
    rmse_list = list()
    for res_1 in res_list_1:

        ode_true = res_1['ode']
        dim_x = ode_true.dim_x
        f_hat_1 = res_1['model'].execute 
        
        def f_hat(t, X):
            X = np.reshape(X, (1, 1))
            dxdt = f_hat_1(X)
            dxdt = dxdt.item()
            return [dxdt]
                
        dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
        xt_hat = dg_hat.xt
        #print(np.shape(xt_hat))

        est_trajectories.append(xt_hat)

        # RMSE:
        if xt_true[:, :, :1].squeeze().shape == xt_hat[:, :, :1].squeeze().shape:
            rmse = root_mean_squared_error(xt_true[:, :, :1].squeeze(), xt_hat[:, :, :1].squeeze())
            rmse_list.append(rmse)
        else:
            est_trajectories.pop()
            print('Skipping this seed')
            pass

    print(rmse_list)
    rmse_mean = np.mean(rmse_list)

    return xt_true, est_trajectories, rmse_mean

# Parametrized
def evaluation1D_p(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed , init_high_0, init_low_0, N=1, init_point=None):
    
    # seed:
    np.random.seed(666)

    # upload the results:
    seed_s = seed
    seed_e = n_seed
    if alg == 'diff':
        path_base = '../results/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else: # -> alg == 'vi':
        path_base = '../results_vi/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    
    res_list_1 = []
    for s in range(seed_s, seed_e):
        
        path_1 = path_base + 'grad_seed_{}.pkl'.format(s)

        try:
            with open(path_1, 'rb') as f_1:
                res_1 = pickle.load(f_1)
            res_list_1.append(res_1)
        except Exception:
            pass 
        # RMK. doing so only results available with both methods are added to the list -> res_list_1, res_list_2 have always the same length
    print(len(res_list_1)) # check

    # percentage of correct identifications:
    correct_list_1 = [res_1['correct'] for res_1 in res_list_1] 
    p_correct_1 = np.mean(correct_list_1) 
    std_correct_1 = np.sqrt(p_correct_1 * (1 - p_correct_1) / len(correct_list_1))
    print(p_correct_1)

    if N==1:

        # true trajectory:
        ode = equations.get_ode(ode_name, ode_param)
        dg_true = data.DataGenerator(ode, ode.T, freq, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0) # !!! freq=10
        xt_true = dg_true.xt
        #print(np.shape(xt_true))

        # estimate trajectories and compute RMSE:
        est_trajectories = []
        rmse_list = list()
        for res_1 in res_list_1:

            ode_true = res_1['ode']
            dim_x = ode_true.dim_x
            f_hat_1 = res_1['model'].execute 
            
            def f_hat(t, X):
                X = np.reshape(X, (1, 2))
                dxdt = f_hat_1(X)
                dxdt = dxdt.item()
                drdt = np.zeros_like(dxdt)
                return [dxdt, drdt]
                    
            dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0) # !!! freq=10
            xt_hat = dg_hat.xt
            #print(np.shape(xt_hat))

            est_trajectories.append(xt_hat)

            # RMSE:
            if xt_true[:, :, :1].squeeze().shape == xt_hat[:, :, :1].squeeze().shape:
                rmse = root_mean_squared_error(xt_true[:, :, :1].squeeze(), xt_hat[:, :, :1].squeeze())
                rmse_list.append(rmse)
            else:
                est_trajectories.pop()
                print('Skipping this seed')
                pass

        print(rmse_list)
        rmse_mean = np.mean(rmse_list)

        return xt_true, est_trajectories, rmse_mean

    else: 

        # true trajectories:
        ode = equations.get_ode(ode_name, ode_param)
        dg_true = data.DataGenerator(ode, ode.T, freq, n_sample=1, noise_sigma=0., init_high=init_point[0], init_low=init_point[0])# !!! freq=10
        xt_true = dg_true.xt
        for i in range(1, N):
            dg_true = data.DataGenerator(ode, ode.T, freq, n_sample=1, noise_sigma=0., init_high=init_point[i], init_low=init_point[i])# !!! freq=10
            xt_true_aux = dg_true.xt
            xt_true = np.concatenate((xt_true, xt_true_aux), axis=1)

        #print('true trajectories')
        #print(np.shape(xt_true))
        xt_true = xt_true.reshape(xt_true.shape[0] * xt_true.shape[1], xt_true.shape[2])
        #print(np.shape(xt_true))


        #compute RMSE:
        rmse_list = list()
        for res_1 in res_list_1:

            ode_true = res_1['ode']
            dim_x = ode_true.dim_x
            f_hat_1 = res_1['model'].execute 
            
            def f_hat(t, X):
                X = np.reshape(X, (1, 2))
                dxdt = f_hat_1(X)
                dxdt = dxdt.item()
                drdt = np.zeros_like(dxdt)
                return [dxdt, drdt]

            dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq, n_sample=1, noise_sigma=0., init_high=init_point[0], init_low=init_point[0])# !!! freq=10
            xt_hat = dg_hat.xt
            for i in range(1, N):
                dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq, n_sample=1, noise_sigma=0., init_high=init_point[i], init_low=init_point[i])# !!! freq=10
                xt_hat_aux = dg_hat.xt
                xt_hat = np.concatenate((xt_hat, xt_hat_aux), axis=1)
            #print('estimated trajectories')
            #print(np.shape(xt_hat))
            xt_hat = xt_hat.reshape(xt_hat.shape[0] * xt_hat.shape[1], xt_hat.shape[2])
            #print(np.shape(xt_hat))

            # RMSE:
            if xt_true[:, :1].shape == xt_hat[:, :1].shape:
                rmse = root_mean_squared_error(xt_true[:, :1], xt_hat[:, :1])
                rmse_list.append(rmse)
            else:
                print('Skipping this seed')
                pass
        
        print(rmse_list)
        rmse_mean = np.mean(rmse_list)
        return rmse_mean


def evaluation1D_2p(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed , init_high_0, init_low_0, N=1, init_point=None):
    
    # seed:
    np.random.seed(666)

    # upload the results:
    seed_s = seed
    seed_e = n_seed
    if alg == 'diff':
        path_base = '../results/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else: # -> alg == 'vi':
        path_base = '../results_vi/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    
    res_list_1 = []
    for s in range(seed_s, seed_e):
        
        path_1 = path_base + 'grad_seed_{}.pkl'.format(s)

        try:
            with open(path_1, 'rb') as f_1:
                res_1 = pickle.load(f_1)
            res_list_1.append(res_1)
        except Exception:
            pass 
        # RMK. doing so only results available with both methods are added to the list -> res_list_1, res_list_2 have always the same length
    print(len(res_list_1)) # check

    # percentage of correct identifications:
    correct_list_1 = [res_1['correct'] for res_1 in res_list_1] 
    p_correct_1 = np.mean(correct_list_1) 
    std_correct_1 = np.sqrt(p_correct_1 * (1 - p_correct_1) / len(correct_list_1))
    print(p_correct_1)

    if N==1:

        # true trajectories:
        ode = equations.get_ode(ode_name, ode_param)
        dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
        xt_true = dg_true.xt
        #print(np.shape(xt_true))

        # estimate trajectories and compute RMSE:
        est_trajectories = []
        rmse_list = list()
        for res_1 in res_list_1:

            ode_true = res_1['ode']
            dim_x = ode_true.dim_x
            f_hat_1 = res_1['model'].execute 
            
            def f_hat(t, X):
                X = np.reshape(X, (1, 3))
                dxdt = f_hat_1(X)
                dxdt = dxdt.item()
                dadt = np.zeros_like(dxdt)
                dbdt = np.zeros_like(dxdt)
                return [dxdt, dadt, dbdt]
                    
            dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
            xt_hat = dg_hat.xt
            #print(np.shape(xt_hat))

            est_trajectories.append(xt_hat)

            # RMSE:
            if xt_true[:, :, :1].squeeze().shape == xt_hat[:, :, :1].squeeze().shape:
                rmse = root_mean_squared_error(xt_true[:, :, :1].squeeze(), xt_hat[:, :, :1].squeeze())
                rmse_list.append(rmse)
            else:
                est_trajectories.pop()
                print('Skipping this seed')
                pass

        print(rmse_list)
        rmse_mean = np.mean(rmse_list)

        return xt_true, est_trajectories, rmse_mean
    
    else:

        # true trajectories:
        ode = equations.get_ode(ode_name, ode_param)
        dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_point[0], init_low=init_point[0])
        xt_true = dg_true.xt
        for i in range(1, N):
            dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_point[i], init_low=init_point[i])
            xt_true_aux = dg_true.xt
            xt_true = np.concatenate((xt_true, xt_true_aux), axis=1)
        #print('true trajectories')
        #print(np.shape(xt_true))
        xt_true = xt_true.reshape(xt_true.shape[0] * xt_true.shape[1], xt_true.shape[2])
        #print(np.shape(xt_true))

        #compute RMSE:
        rmse_list = list()
        for res_1 in res_list_1:

            ode_true = res_1['ode']
            dim_x = ode_true.dim_x
            f_hat_1 = res_1['model'].execute 
            
            def f_hat(t, X):
                X = np.reshape(X, (1, 3))
                dxdt = f_hat_1(X)
                dxdt = dxdt.item()
                dadt = np.zeros_like(dxdt)
                dbdt = np.zeros_like(dxdt)
                return [dxdt, dadt, dbdt]
                    
            dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_point[0], init_low=init_point[0])
            xt_hat = dg_hat.xt
            for i in range(1, N):
                dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_point[i], init_low=init_point[i])
                xt_hat_aux = dg_hat.xt
                xt_hat = np.concatenate((xt_hat, xt_hat_aux), axis=1)
            #print('estimated trajectories')
            #print(np.shape(xt_hat))
            xt_hat = xt_hat.reshape(xt_hat.shape[0] * xt_hat.shape[1], xt_hat.shape[2])
            #print(np.shape(xt_hat))

            # RMSE:
            if xt_true[:, :1].shape == xt_hat[:, :1].shape:
                rmse = root_mean_squared_error(xt_true[:, :1], xt_hat[:, :1])
                rmse_list.append(rmse)
            else:
                print('Skipping this seed')
                pass

        print(rmse_list)
        rmse_mean = np.mean(rmse_list)

        return rmse_mean


# 2D:
# Standard
def evaluation2D(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed , init_high_0, init_low_0):
    
    # seed:
    np.random.seed(666)

    # upload the results:
    seed_s = seed
    seed_e = n_seed
    if alg == 'diff':
        path_base = '../results/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else: # -> alg == 'vi':
        path_base = '../results_vi/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    
    res_list_1 = []
    res_list_2 = []
    for s in range(seed_s, seed_e):
        
        path_1 = path_base + 'grad_seed_{}.pkl'.format(s)
        path_2 = path_base + 'grad_x_{}_seed_{}.pkl'.format(1, s)

        try:
            with open(path_1, 'rb') as f_1:
                res_1 = pickle.load(f_1)
            with open(path_2, 'rb') as f_2:
                res_2 = pickle.load(f_2)
            res_list_1.append(res_1)
            res_list_2.append(res_2)
        except Exception:
            pass 
        # RMK. doing so only results available with both methods are added to the list -> res_list_1, res_list_2 have always the same length
    print(len(res_list_1)) # check

    # percentage of correct identifications:
    correct_list_1 = [res_1['correct'] for res_1 in res_list_1] 
    p_correct_1 = np.mean(correct_list_1) 
    std_correct_1 = np.sqrt(p_correct_1 * (1 - p_correct_1) / len(correct_list_1))
    print(p_correct_1)
    correct_list_2 = [res_2['correct'] for res_2 in res_list_2] 
    p_correct_2 = np.mean(correct_list_2) 
    std_correct_2 = np.sqrt(p_correct_2 * (1 - p_correct_2) / len(correct_list_2))
    print(p_correct_2)

    # true trajectories:
    ode = equations.get_ode(ode_name, ode_param)
    dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
    xt_true = dg_true.xt
    #print(np.shape(xt_true))

    # estimate trajectories and compute RMSE:
    est_trajectories = []
    rmse_list = list()
    for res_1, res_2 in zip(res_list_1, res_list_2):

        ode_true = res_1['ode']
        dim_x = ode_true.dim_x
        f_hat_1 = res_1['model'].execute 
        f_hat_2 = res_2['model'].execute 
        
        def f_hat(t, X):
            X = np.reshape(X, (1, 2))
            dxdt = f_hat_1(X)
            dxdt = dxdt.item()
            dydt = f_hat_2(X)
            dydt = dydt.item()
            return [dxdt, dydt]
                
        dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
        xt_hat = dg_hat.xt

        est_trajectories.append(xt_hat)

        # RMSE:
        if xt_true[:, :, :2].squeeze().shape == xt_hat[:, :, :2].squeeze().shape:
            rmse = root_mean_squared_error(xt_true[:, :, :2].squeeze(), xt_hat[:, :, :2].squeeze())
            rmse_list.append(rmse)
        else:
            est_trajectories.pop()
            print('Skipping this seed')
            pass

    print(rmse_list)
    rmse_mean = np.mean(rmse_list)

    return xt_true, est_trajectories, rmse_mean

def evaluation2D_partial(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed , init_high_0, init_low_0):
    
    # seed:
    np.random.seed(666)

    # upload the results:
    seed_s = seed
    seed_e = n_seed
    if alg == 'diff':
        path_base = '../results/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else: # -> alg == 'vi':
        path_base = '../results_vi/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    
    res_list = []
    for s in range(seed_s, seed_e):
        if x_id == 0:
            path = path_base + 'grad_seed_{}.pkl'.format(s)
        else:
            path = path_base + 'grad_x_{}_seed_{}.pkl'.format(x_id, s)

        try:
            with open(path, 'rb') as f:
                res = pickle.load(f)
            res_list.append(res)
        except Exception:
            pass
    print(len(res_list)) # check

    # percentage of correct identifications:
    correct_list = [res['correct'] for res in res_list] # list of booleans: 1 if the model has correctly identified the ODE, 0 otherwise
    p_correct = np.mean(correct_list) # percentage of correct identifications
    std_correct = np.sqrt(p_correct * (1 - p_correct) / len(correct_list)) # standard deviation of the percentage of correct identifications
    print(p_correct)

    # true trajectories:
    ode = equations.get_ode(ode_name, ode_param)
    dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
    xt_true = dg_true.xt
    #print(np.shape(xt_true))

    # estimate trajectories and compute RMSE:
    est_trajectories = []
    rmse_list = list()
    for res in res_list:

        ode_true = res['ode']
        dim_x = ode_true.dim_x
        f_hat_partial = res['model'].execute 
        
        if x_id == 0:
            def f_hat(t, X):
                x = X[0]
                y = X[1]
                X = np.reshape(X, (1, 2))
                dxdt = f_hat_partial(X)
                dxdt = dxdt.item()
                _, dydt = ode_true._dx_dt(x, y)
                dydt = np.asarray(dydt).squeeze()
                return [dxdt, dydt]
        else: # -> x_id = 1
            def f_hat(t, X):
                x = X[0]
                y = X[1]
                X = np.reshape(X, (1, 2))
                dydt = f_hat_partial(X)
                dydt = dydt.item()
                dxdt, _ = ode_true._dx_dt(x, y)
                dxdt = np.asarray(dxdt).squeeze()
                return [dxdt, dydt]


        dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
        xt_hat = dg_hat.xt

        est_trajectories.append(xt_hat)

        # RMSE:
        if xt_true[:, :, :2].squeeze().shape == xt_hat[:, :, :2].squeeze().shape:
            rmse = root_mean_squared_error(xt_true[:, :, :2].squeeze(), xt_hat[:, :, :2].squeeze())
            rmse_list.append(rmse)
        else:
            est_trajectories.pop()
            print('Skipping this seed')

            pass

    print(rmse_list)
    rmse_mean = np.mean(rmse_list)

    return xt_true, est_trajectories, rmse_mean

# Parametrized
def evaluation2D_p(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed , init_high_0, init_low_0, N=1, init_point=None):
    
    # seed:
    np.random.seed(666)

    # upload the results:
    seed_s = seed
    seed_e = n_seed
    if alg == 'diff':
        path_base = '../results/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else: # -> alg == 'vi':
        path_base = '../results_vi/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    
    res_list_1 = []
    res_list_2 = []
    for s in range(seed_s, seed_e):
        
        path_1 = path_base + 'grad_seed_{}.pkl'.format(s)
        path_2 = path_base + 'grad_x_{}_seed_{}.pkl'.format(1, s)

        try:
            with open(path_1, 'rb') as f_1:
                res_1 = pickle.load(f_1)
            with open(path_2, 'rb') as f_2:
                res_2 = pickle.load(f_2)
            res_list_1.append(res_1)
            res_list_2.append(res_2)
        except Exception:
            pass 
        # RMK. doing so only results available with both methods are added to the list -> res_list_1, res_list_2 have always the same length
    print(len(res_list_1)) # check

    # percentage of correct identifications:
    correct_list_1 = [res_1['correct'] for res_1 in res_list_1] 
    p_correct_1 = np.mean(correct_list_1) 
    std_correct_1 = np.sqrt(p_correct_1 * (1 - p_correct_1) / len(correct_list_1))
    print(p_correct_1)
    correct_list_2 = [res_2['correct'] for res_2 in res_list_2] 
    p_correct_2 = np.mean(correct_list_2) 
    std_correct_2 = np.sqrt(p_correct_2 * (1 - p_correct_2) / len(correct_list_2))
    print(p_correct_2)


    if N==1:
        # true trajectories:
        ode = equations.get_ode(ode_name, ode_param)
        dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
        xt_true = dg_true.xt
        #print(np.shape(xt_true))

        # estimate trajectories and compute RMSE:
        est_trajectories = []
        rmse_list = list()
        for res_1, res_2 in zip(res_list_1, res_list_2):

            ode_true = res_1['ode']
            dim_x = ode_true.dim_x
            f_hat_1 = res_1['model'].execute 
            f_hat_2 = res_2['model'].execute 
            
            def f_hat(t, X):
                X = np.reshape(X, (1, 3))
                dxdt = f_hat_1(X)
                dxdt = dxdt.item()
                dydt = f_hat_2(X)
                dydt = dydt.item()
                drdt = np.zeros_like(dydt)
                return [dxdt, dydt, drdt]
                    
            dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
            xt_hat = dg_hat.xt

            est_trajectories.append(xt_hat)

            # RMSE:
            if xt_true[:, :, :2].squeeze().shape == xt_hat[:, :, :2].squeeze().shape:
                rmse = root_mean_squared_error(xt_true[:, :, :2].squeeze(), xt_hat[:, :, :2].squeeze())
                rmse_list.append(rmse)
            else:
                est_trajectories.pop()
                print('Skipping this seed')
                pass

        print(rmse_list)
        rmse_mean = np.mean(rmse_list)

        return xt_true, est_trajectories, rmse_mean
    
    else:
            
            # true trajectories:
            ode = equations.get_ode(ode_name, ode_param)
            dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_point[0], init_low=init_point[0])
            xt_true = dg_true.xt
            for i in range(1, N):
                dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_point[i], init_low=init_point[i])
                xt_true_aux = dg_true.xt
                xt_true = np.concatenate((xt_true, xt_true_aux), axis=1)
            xt_true = xt_true.reshape(xt_true.shape[0] * xt_true.shape[1], xt_true.shape[2])
    
            #compute RMSE:
            rmse_list = list()
            for res_1, res_2 in zip(res_list_1, res_list_2):
    
                ode_true = res_1['ode']
                dim_x = ode_true.dim_x
                f_hat_1 = res_1['model'].execute 
                f_hat_2 = res_2['model'].execute 
                
                def f_hat(t, X):
                    X = np.reshape(X, (1, 3))
                    dxdt = f_hat_1(X)
                    dxdt = dxdt.item()
                    dydt = f_hat_2(X)
                    dydt = dydt.item()
                    drdt = np.zeros_like(dydt)
                    return [dxdt, dydt, drdt]
                        
                dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_point[0], init_low=init_point[0])
                xt_hat = dg_hat.xt
                shape = np.shape(xt_hat)
                for i in range(1, N):
                    dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_point[i], init_low=init_point[i])
                    xt_hat_aux = dg_hat.xt
                    if xt_hat_aux.shape == shape:
                        xt_hat = np.concatenate((xt_hat, xt_hat_aux), axis=1)
                xt_hat = xt_hat.reshape(xt_hat.shape[0] * xt_hat.shape[1], xt_hat.shape[2])

                # RMSE:
                if xt_true[:, :2].shape == xt_hat[:, :2].shape:
                    rmse = root_mean_squared_error(xt_true[:, :2], xt_hat[:, :2])
                    rmse_list.append(rmse)
                else:
                    print('Skipping this seed') # RMK. nel caso in cui, anche solo per uno delle N trajectories, il modello_i trovato non riesca a ricostruire la traiettoria, non verrà riportato il valore di RMSE associato a tale modello.
                                                # per migliorare il metodo si potrebbe skippare solo la traiettoria problematica e calcolare l'RMSE sulle altre, ma vedere se ne vale la pena... 
                                                # RMK. nel caso in cui sia il seed 0 quello problematico non funziona -> in caso sistemare..
                    pass

            print(rmse_list)
            rmse_mean = np.mean(rmse_list)
            return rmse_mean
                                            
def evaluation2D_partial_p(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed , init_high_0, init_low_0):
    
    # seed:
    np.random.seed(666)

    # upload the results:
    seed_s = seed
    seed_e = n_seed
    if alg == 'diff':
        path_base = '../results/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else: # -> alg == 'vi':
        path_base = '../results_vi/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    
    res_list = []
    for s in range(seed_s, seed_e):
        if x_id == 0:
            path = path_base + 'grad_seed_{}.pkl'.format(s)
        else:
            path = path_base + 'grad_x_{}_seed_{}.pkl'.format(x_id, s)

        try:
            with open(path, 'rb') as f:
                res = pickle.load(f)
            res_list.append(res)
        except Exception:
            pass
    print(len(res_list)) # check

    # percentage of correct identifications:
    correct_list = [res['correct'] for res in res_list] # list of booleans: 1 if the model has correctly identified the ODE, 0 otherwise
    p_correct = np.mean(correct_list) # percentage of correct identifications
    std_correct = np.sqrt(p_correct * (1 - p_correct) / len(correct_list)) # standard deviation of the percentage of correct identifications
    print(p_correct)

    # true trajectories:
    ode = equations.get_ode(ode_name, ode_param)
    dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
    xt_true = dg_true.xt
    #print(np.shape(xt_true))

    # estimate trajectories and compute RMSE:
    est_trajectories = []
    rmse_list = list()
    for res in res_list:

        ode_true = res['ode']
        dim_x = ode_true.dim_x
        f_hat_partial = res['model'].execute 
        
        if x_id == 0:
            def f_hat(t, X):
                x = X[0]
                y = X[1]
                rho = X[2]
                X = np.reshape(X, (1, 3))
                dxdt = f_hat_partial(X)
                dxdt = dxdt.item()
                _, dydt, drdt = ode_true._dx_dt(x, y, rho)
                dydt = np.asarray(dydt).squeeze()
                drdt = np.asarray(drdt).squeeze()
                return [dxdt, dydt, drdt]
        else: # -> x_id = 1
            def f_hat(t, X):
                x = X[0]
                y = X[1]
                rho = X[2]
                X = np.reshape(X, (1, 3))
                dydt = f_hat_partial(X)
                dydt = dydt.item()
                dxdt, _, drdt = ode_true._dx_dt(x, y, rho)
                dxdt = np.asarray(dxdt).squeeze()
                drdt = np.asarray(drdt).squeeze()
                return [dxdt, dydt, drdt]


        dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
        xt_hat = dg_hat.xt

        est_trajectories.append(xt_hat)

        #print(xt_true[:, :, :2].squeeze().shape)
        #print(xt_hat[:, :, :2].squeeze().shape)

        # RMSE:
        if xt_true[:, :, :2].squeeze().shape == xt_hat[:, :, :2].squeeze().shape:
            rmse = root_mean_squared_error(xt_true[:, :, :2].squeeze(), xt_hat[:, :, :2].squeeze())
            rmse_list.append(rmse)
        else:
            est_trajectories.pop()
            print('Skipping this seed')
            pass

    print(rmse_list)
    rmse_mean = np.mean(rmse_list)

    return xt_true, est_trajectories, rmse_mean


# 3D:
# Standard
def evaluation3D(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed , init_high_0, init_low_0):
    
    # seed:
    np.random.seed(666)

    # upload the results:
    seed_s = seed
    seed_e = n_seed
    if alg == 'diff':
        path_base = '../results/{}/noise-{}/sample-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else: # -> alg == 'vi':
        path_base = '../results_vi/{}/noise-{}/sample-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    
    res_list_1 = []
    res_list_2 = []
    res_list_3 = []
    for s in range(seed_s, seed_e):

        path_1 = path_base + 'freq-25/' + 'grad_seed_{}.pkl'.format(s)
        path_2 = path_base + 'freq-50/' + 'grad_x_{}_seed_{}.pkl'.format(1, s)
        path_3 = path_base + 'freq-50/' + 'grad_x_{}_seed_{}.pkl'.format(2, s)

        try:
            with open(path_1, 'rb') as f_1:
                res_1 = pickle.load(f_1)
            with open(path_2, 'rb') as f_2:
                res_2 = pickle.load(f_2)
            with open(path_3, 'rb') as f_3:
                res_3 = pickle.load(f_3)
            res_list_1.append(res_1)
            res_list_2.append(res_2)
            res_list_3.append(res_3)
        except Exception:
            pass 
        # RMK. doing so only results available for all three methods are added to the list -> res_list_1, res_list_2, res_list_3 have always the same length
    print(len(res_list_1)) # check

    # percentage of correct identifications:
    correct_list_1 = [res_1['correct'] for res_1 in res_list_1] 
    p_correct_1 = np.mean(correct_list_1) 
    std_correct_1 = np.sqrt(p_correct_1 * (1 - p_correct_1) / len(correct_list_1))
    print(p_correct_1)
    correct_list_2 = [res_2['correct'] for res_2 in res_list_2] 
    p_correct_2 = np.mean(correct_list_2) 
    std_correct_2 = np.sqrt(p_correct_2 * (1 - p_correct_2) / len(correct_list_2))
    print(p_correct_2)
    correct_list_3 = [res_3['correct'] for res_3 in res_list_3] 
    p_correct_3 = np.mean(correct_list_3) 
    std_correct_3 = np.sqrt(p_correct_3 * (1 - p_correct_3) / len(correct_list_3))
    print(p_correct_3)

    # true trajectories:
    ode = equations.get_ode(ode_name, ode_param)
    dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
    xt_true = dg_true.xt
    #print(np.shape(xt_true))

    # estimate trajectories and compute RMSE:
    est_trajectories = []
    rmse_list = list()
    for res_1, res_2, res_3 in zip(res_list_1, res_list_2, res_list_3):

        ode_true = res_1['ode']
        dim_x = ode_true.dim_x
        f_hat_1 = res_1['model'].execute 
        f_hat_2 = res_2['model'].execute 
        f_hat_3 = res_3['model'].execute
        
        def f_hat(t, X):
            X = np.reshape(X, (1, 3))
            dxdt = f_hat_1(X)
            dxdt = dxdt.item()
            dydt = f_hat_2(X)
            dydt = dydt.item()
            dzdt = f_hat_3(X)
            dzdt = dzdt.item()
            return [dxdt, dydt, dzdt]
                
        dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
        xt_hat = dg_hat.xt

        est_trajectories.append(xt_hat)

        # RMSE:
        if xt_true[:, :, :3].squeeze().shape == xt_hat[:, :, :3].squeeze().shape:
            rmse = root_mean_squared_error(xt_true[:, :, :3].squeeze(), xt_hat[:, :, :3].squeeze())
            rmse_list.append(rmse)
        else:
            est_trajectories.pop()
            print('Skipping this seed')
            pass

    print(rmse_list)
    rmse_mean = np.mean(rmse_list)

    return xt_true, est_trajectories, rmse_mean

def evaluation3D_partial(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed , init_high_0, init_low_0):
    
    # seed:
    np.random.seed(666)

    # upload the results:
    seed_s = seed
    seed_e = n_seed
    if alg == 'diff':
        path_base = '../results/{}/noise-{}/sample-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else: # -> alg == 'vi':
        path_base = '../results_vi/{}/noise-{}/sample-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    
    res_list = []
    for s in range(seed_s, seed_e):

        if x_id == 0:
            path = path_base + 'freq-25/' + 'grad_seed_{}.pkl'.format(s)
        elif x_id == 1:
            path = path_base + 'freq-50/' + 'grad_x_{}_seed_{}.pkl'.format(1, s)
        else: # -> x_id == 2
            path = path_base + 'freq-50/' + 'grad_x_{}_seed_{}.pkl'.format(2, s)

        try:
            with open(path, 'rb') as f:
                res = pickle.load(f)
            res_list.append(res)
        except Exception:
            pass
    print(len(res_list)) # check

    # percentage of correct identifications:
    correct_list = [res['correct'] for res in res_list] # list of booleans: 1 if the model has correctly identified the ODE, 0 otherwise
    p_correct = np.mean(correct_list) # percentage of correct identifications
    std_correct = np.sqrt(p_correct * (1 - p_correct) / len(correct_list)) # standard deviation of the percentage of correct identifications
    print(p_correct)

    # true trajectories:
    ode = equations.get_ode(ode_name, ode_param)
    dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
    xt_true = dg_true.xt
    #print(np.shape(xt_true))

    # estimate trajectories and compute RMSE:
    est_trajectories = []
    rmse_list = list()
    for res in res_list:

        ode_true = res['ode']
        dim_x = ode_true.dim_x
        f_hat_partial = res['model'].execute 
        
        if x_id == 0:
            def f_hat(t, X):
                x = X[0]
                y = X[1]
                z = X[2]
                X = np.reshape(X, (1, 3))
                dxdt = f_hat_partial(X)
                dxdt = dxdt.item()
                _, dydt, dzdt = ode_true._dx_dt(x, y, z)
                dydt = np.asarray(dydt).squeeze()
                dzdt = np.asarray(dzdt).squeeze()
                return [dxdt, dydt, dzdt]
        elif x_id == 1: 
            def f_hat(t, X):
                x = X[0]
                y = X[1]
                z = X[2]
                X = np.reshape(X, (1, 3))
                dydt = f_hat_partial(X)
                dydt = dydt.item()
                dxdt, _, dzdt = ode_true._dx_dt(x, y, z)
                dxdt = np.asarray(dxdt).squeeze()
                dzdt = np.asarray(dzdt).squeeze()
                return [dxdt, dydt, dzdt]
        else: # -> x_id == 2
            def f_hat(t, X):
                x = X[0]
                y = X[1]
                z = X[2]
                X = np.reshape(X, (1, 3))
                dzdt = f_hat_partial(X)
                dzdt = dzdt.item()
                dxdt, dydt, _ = ode_true._dx_dt(x, y, z)
                dxdt = np.asarray(dxdt).squeeze()
                dydt = np.asarray(dydt).squeeze()
                return [dxdt, dydt, dzdt]


        dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
        xt_hat = dg_hat.xt
        #print(np.shape(xt_hat))

        est_trajectories.append(xt_hat)

        # RMSE:
        if xt_true[:, :, :3].squeeze().shape == xt_hat[:, :, :3].squeeze().shape:
            rmse = root_mean_squared_error(xt_true[:, :, :3].squeeze(), xt_hat[:, :, :3].squeeze())
            rmse_list.append(rmse)
        else:
            est_trajectories.pop()
            print('Skipping this seed')

            pass

    print(rmse_list)
    rmse_mean = np.mean(rmse_list)

    return xt_true, est_trajectories, rmse_mean

# Parametrized
def evaluation3D_partial_p(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed , init_high_0, init_low_0, N=1, init_point=None):
    
    # seed:
    np.random.seed(666)

    # upload the results:
    seed_s = seed
    seed_e = n_seed
    if alg == 'diff':
        path_base = '../results/{}/noise-{}/sample-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else: # -> alg == 'vi':
        path_base = '../results_vi/{}/noise-{}/sample-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    
    res_list = []
    for s in range(seed_s, seed_e):

        if x_id == 0:
            path = path_base + 'freq-25/' + 'grad_seed_{}.pkl'.format(s)
        elif x_id == 1:
            path = path_base + 'freq-50/' + 'grad_x_{}_seed_{}.pkl'.format(1, s)
        else: # -> x_id == 2
            path = path_base + 'freq-50/' + 'grad_x_{}_seed_{}.pkl'.format(2, s)

        try:
            with open(path, 'rb') as f:
                res = pickle.load(f)
            res_list.append(res)
        except Exception:
            pass
    print(len(res_list)) # check

    # percentage of correct identifications:
    correct_list = [res['correct'] for res in res_list] # list of booleans: 1 if the model has correctly identified the ODE, 0 otherwise
    p_correct = np.mean(correct_list) # percentage of correct identifications
    std_correct = np.sqrt(p_correct * (1 - p_correct) / len(correct_list)) # standard deviation of the percentage of correct identifications
    print(p_correct)

    if N==1:

        # true trajectories:
        ode = equations.get_ode(ode_name, ode_param)
        dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
        xt_true = dg_true.xt
        #print(np.shape(xt_true))

        # estimate trajectories and compute RMSE:
        est_trajectories = []
        rmse_list = list()
        for res in res_list:

            ode_true = res['ode']
            dim_x = ode_true.dim_x
            f_hat_partial = res['model'].execute 
            
            if x_id == 0:
                def f_hat(t, X):
                    x = X[0]
                    y = X[1]
                    z = X[2]
                    rho = X[3]
                    X = np.reshape(X, (1, 4))
                    dxdt = f_hat_partial(X)
                    dxdt = dxdt.item()
                    _, dydt, dzdt, drdt = ode_true._dx_dt(x, y, z, rho)
                    dydt = np.asarray(dydt).squeeze()
                    dzdt = np.asarray(dzdt).squeeze()
                    drdt = np.asarray(drdt).squeeze()
                    return [dxdt, dydt, dzdt, drdt]
            elif x_id == 1:
                def f_hat(t, X):
                    x = X[0]
                    y = X[1]
                    z = X[2]
                    rho = X[3]
                    X = np.reshape(X, (1, 4))
                    dydt = f_hat_partial(X)
                    dydt = dydt.item()
                    dxdt, _, dzdt, drdt = ode_true._dx_dt(x, y, z, rho)
                    dxdt = np.asarray(dxdt).squeeze()
                    dzdt = np.asarray(dzdt).squeeze()
                    drdt = np.asarray(drdt).squeeze()
                    return [dxdt, dydt, dzdt, drdt]
            else: # -> x_id == 2
                def f_hat(t, X):
                    x = X[0]
                    y = X[1]
                    z = X[2]
                    rho = X[3]
                    X = np.reshape(X, (1, 4))
                    dzdt = f_hat_partial(X)
                    dzdt = dzdt.item()
                    dxdt, dydt, _, drdt = ode_true._dx_dt(x, y, z, rho)
                    dxdt = np.asarray(dxdt).squeeze()
                    dydt = np.asarray(dydt).squeeze()
                    drdt = np.asarray(drdt).squeeze()
                    return [dxdt, dydt, dzdt, drdt]


            dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
            xt_hat = dg_hat.xt
            #print(np.shape(xt_hat))

            est_trajectories.append(xt_hat)

            # RMSE:
            if xt_true[:, :, :3].squeeze().shape == xt_hat[:, :, :3].squeeze().shape:
                rmse = root_mean_squared_error(xt_true[:, :, :3].squeeze(), xt_hat[:, :, :3].squeeze())
                rmse_list.append(rmse)
            else:
                est_trajectories.pop()
                print('Skipping this seed')

                pass

        print(rmse_list)
        rmse_mean = np.mean(rmse_list)

        return xt_true, est_trajectories, rmse_mean
    
    else:

        # true trajectories:
        ode = equations.get_ode(ode_name, ode_param)
        dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_point[0], init_low=init_point[0])
        xt_true = dg_true.xt
        for i in range(1, N):
            dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_point[i], init_low=init_point[i])
            xt_true_aux = dg_true.xt
            xt_true = np.concatenate((xt_true, xt_true_aux), axis=1)
        xt_true = xt_true.reshape(xt_true.shape[0] * xt_true.shape[1], xt_true.shape[2])

        #compute RMSE:
        rmse_list = list()
        for res in res_list:

            ode_true = res['ode']
            dim_x = ode_true.dim_x
            f_hat_partial = res['model'].execute 
            
            if x_id == 0:
                def f_hat(t, X):
                    x = X[0]
                    y = X[1]
                    z = X[2]
                    rho = X[3]
                    X = np.reshape(X, (1, 4))
                    dxdt = f_hat_partial(X)
                    dxdt = dxdt.item()
                    _, dydt, dzdt, drdt = ode_true._dx_dt(x, y, z, rho)
                    dydt = np.asarray(dydt).squeeze()
                    dzdt = np.asarray(dzdt).squeeze()
                    drdt = np.asarray(drdt).squeeze()
                    return [dxdt, dydt, dzdt, drdt]
            elif x_id == 1:
                def f_hat(t, X):
                    x = X[0]
                    y = X[1]
                    z = X[2]
                    rho = X[3]
                    X = np.reshape(X, (1, 4))
                    dydt = f_hat_partial(X)
                    dydt = dydt.item()
                    dxdt, _, dzdt, drdt = ode_true._dx_dt(x, y, z, rho)
                    dxdt = np.asarray(dxdt).squeeze()
                    dzdt = np.asarray(dzdt).squeeze()
                    drdt = np.asarray(drdt).squeeze()
                    return [dxdt, dydt, dzdt, drdt]
            else: # -> x_id == 2
                def f_hat(t, X):
                    x = X[0]
                    y = X[1]
                    z = X[2]
                    rho = X[3]
                    X = np.reshape(X, (1, 4))
                    dzdt = f_hat_partial(X)
                    dzdt = dzdt.item()
                    dxdt, dydt, _, drdt = ode_true._dx_dt(x, y, z, rho)
                    dxdt = np.asarray(dxdt).squeeze()
                    dydt = np.asarray(dydt).squeeze()
                    drdt = np.asarray(drdt).squeeze()
                    return [dxdt, dydt, dzdt, drdt]
                

            dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_point[0], init_low=init_point[0])
            xt_hat = dg_hat.xt
            shape = np.shape(xt_hat)
            for i in range(1, N):
                dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_point[i], init_low=init_point[i])
                xt_hat_aux = dg_hat.xt
                if xt_hat_aux.shape == shape:
                    xt_hat = np.concatenate((xt_hat, xt_hat_aux), axis=1)
            xt_hat = xt_hat.reshape(xt_hat.shape[0] * xt_hat.shape[1], xt_hat.shape[2])


            # RMSE:
            if xt_true[:, :3].shape == xt_hat[:, :3].shape:
                rmse = root_mean_squared_error(xt_true[:, :3], xt_hat[:, :3])
                rmse_list.append(rmse)
            else:
                print('Skipping this seed')# RMK. nel caso in cui, anche solo per uno delle N trajectories, il modello_i trovato non riesca a ricostruire la traiettoria, non verrà riportato il valore di RMSE associato a tale modello.
                                           # per migliorare il metodo si potrebbe skippare solo la traiettoria problematica e calcolare l'RMSE sulle altre, ma vedere se ne vale la pena... 
                                           # RMK. nel caso in cui sia il seed 0 quello problematico non funziona -> in caso sistemare..
                pass

        print(rmse_list)
        rmse_mean = np.mean(rmse_list)
        return rmse_mean
    


# dev:
#evaluation3D_p, per visualizzare la ricostruzione totale nel caso parametrizzato, RMK. le tre equazioni sono state stimate con parametrizzazioni diverse, ma ok...  --> unfeasible
#just in case: evaluation2D_partial_p_2, in cu, oltre ad usare l'equazione identificata nel caso parametrizzato, usiamo per le altre due le equazioni stimate nel caso non parametrizzato, anziché le equazioni corrette come in evaluation2D_partial_p --> unfeasible
def evaluation3D_p(ode_name_1, ode_name_2, ode_name_3, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed , init_high_0, init_low_0):
    
    #     # seed:
    #     np.random.seed(666)

    #     # upload the results:
    #     seed_s = seed
    #     seed_e = n_seed
    #     if alg == 'diff':
    #         path_base_1 = '../results/{}/noise-{}/sample-{}/'.format(ode_name_1, noise_ratio, n_sample, freq)
    #         path_base_2 = '../results/{}/noise-{}/sample-{}/'.format(ode_name_2, noise_ratio, n_sample, freq)
    #         path_base_3 = '../results/{}/noise-{}/sample-{}/'.format(ode_name_3, noise_ratio, n_sample, freq)
    #     else: # -> alg == 'vi':
    #         path_base_1 = '../results_vi/{}/noise-{}/sample-{}/'.format(ode_name_1, noise_ratio, n_sample, freq)
    #         path_base_2 = '../results_vi/{}/noise-{}/sample-{}/'.format(ode_name_2, noise_ratio, n_sample, freq)
    #         path_base_3 = '../results_vi/{}/noise-{}/sample-{}/'.format(ode_name_3, noise_ratio, n_sample, freq)
        
    #     res_list_1 = []
    #     res_list_2 = []
    #     res_list_3 = []
    #     for s in range(seed_s, seed_e):

    #         path_1 = path_base_1 + 'freq-25/' + 'grad_seed_{}.pkl'.format(s)
    #         path_2 = path_base_2 + 'freq-50/' + 'grad_x_{}_seed_{}.pkl'.format(1, s)
    #         path_3 = path_base_3 + 'freq-50/' + 'grad_x_{}_seed_{}.pkl'.format(2, s)

    #         try:
    #             with open(path_1, 'rb') as f_1:
    #                 res_1 = pickle.load(f_1)
    #             with open(path_2, 'rb') as f_2:
    #                 res_2 = pickle.load(f_2)
    #             with open(path_3, 'rb') as f_3:
    #                 res_3 = pickle.load(f_3)
    #             res_list_1.append(res_1)
    #             res_list_2.append(res_2)
    #             res_list_3.append(res_3)
    #         except Exception:
    #             pass 
    #         # RMK. doing so only results available for all three methods are added to the list -> res_list_1, res_list_2, res_list_3 have always the same length
    #     print(len(res_list_1)) # check

    #     # percentage of correct identifications:
    #     correct_list_1 = [res_1['correct'] for res_1 in res_list_1] 
    #     p_correct_1 = np.mean(correct_list_1) 
    #     std_correct_1 = np.sqrt(p_correct_1 * (1 - p_correct_1) / len(correct_list_1))
    #     print(p_correct_1)
    #     correct_list_2 = [res_2['correct'] for res_2 in res_list_2] 
    #     p_correct_2 = np.mean(correct_list_2) 
    #     std_correct_2 = np.sqrt(p_correct_2 * (1 - p_correct_2) / len(correct_list_2))
    #     print(p_correct_2)
    #     correct_list_3 = [res_3['correct'] for res_3 in res_list_3] 
    #     p_correct_3 = np.mean(correct_list_3) 
    #     std_correct_3 = np.sqrt(p_correct_3 * (1 - p_correct_3) / len(correct_list_3))
    #     print(p_correct_3)

    #     # true trajectories:
    #     ode = equations.get_ode(ode_name_1, ode_param)
    #     dg_true = data.DataGenerator(ode, ode.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
    #     xt_true = dg_true.xt
    #     #print(np.shape(xt_true))

    #     # estimate trajectories and compute RMSE:
    #     est_trajectories = []
    #     rmse_list = list()
    #     for res_1, res_2, res_3 in zip(res_list_1, res_list_2, res_list_3):

    #         ode_true = res_1['ode']
    #         dim_x = ode_true.dim_x
    #         f_hat_1 = res_1['model'].execute 
    #         f_hat_2 = res_2['model'].execute 
    #         f_hat_3 = res_3['model'].execute
            
    #         def f_hat(t, X):
    #             X = np.reshape(X, (1, 3))
    #             dxdt = f_hat_1(X)
    #             dxdt = dxdt.item()
    #             dydt = f_hat_2(X)
    #             dydt = dydt.item()
    #             drdt = np.zeros_like(dydt)
    #             return [dxdt, dydt, drdt]
                    
    #         dg_hat = data.DataGenerator_p(f_hat, dim_x, ode_true.T, freq=10, n_sample=1, noise_sigma=0., init_high=init_high_0, init_low=init_low_0)
    #         xt_hat = dg_hat.xt

    #         est_trajectories.append(xt_hat)

    #         # RMSE:
    #         if xt_true[:, :, :2].squeeze().shape == xt_hat[:, :, :2].squeeze().shape:
    #             rmse = root_mean_squared_error(xt_true[:, :, :2].squeeze(), xt_hat[:, :, :2].squeeze())
    #             rmse_list.append(rmse)
    #         else:
    #             est_trajectories.pop()
    #             print('Skipping this seed')
    #             pass

    #     print(rmse_list)
    #     rmse_mean = np.mean(rmse_list)

    #     return xt_true, est_trajectories, rmse_mean
    return
