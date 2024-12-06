from . import basis


def get_config_real(x_id=0):
    return real_config


def get_config(ode, x_id=0):
    if ode.name == 'GompertzODE':
        if ode.has_coef:
            return gompertz_config
        else:
            return gompertz_config_no_coef
    elif ode.name == 'GompertzODE_par_a':
        return gompertz_par_a_config
    elif ode.name == 'GompertzODE_par_b':
        return gompertz_par_b_config
    elif ode.name == 'GompertzODE_par_ab':
        return gompertz_par_ab_config
    elif ode.name == 'LogisticODE' or ode.name == 'LogisticODE_a':
        return logistic_config
    elif ode.name == 'LogisticODE_k':
        return logistic_config_k
    elif ode.name == 'SelkovODE' or ode.name == 'SelkovODE_rho' or ode.name == 'SelkovODE_sigma' or ode.name == 'SelkovODE_rho_03' or ode.name == 'SelkovODE_rho_04' or ode.name == 'SelkovODE_rho_06' or ode.name == 'SelkovODE_rho_09':
        return selkov_config
    elif ode.name == 'OscillatingSelkovODE' or ode.name == 'OscillatingSelkovODE_d':
        return oscillating_selkov_config
    elif ode.name == 'OscillatingLogisticODE':
        return oscillating_logistic_config
    elif ode.name == 'HillODE':
        return hill_config
    elif ode.name == 'FracODE':
        return frac_config
    elif ode.name == 'TrigonometricODE':
        return trigonometric_config
    elif ode.name == 'Lorenz' or ode.name == 'Lorenz_sigma' or ode.name == 'Lorenz_rho' or ode.name == 'Lorenz_beta': 
        if x_id == 0:
            return lorenz_config_x0
        elif x_id == 1:
            print('loading config for X1')
            return lorenz_config_x1
        else:
            print('loading config for X2')
            return lorenz_config_x2


def get_interpolation_config(ode, x_id=0):
    if ode.name == 'GompertzODE':
        return gompertz_interp_config
    elif ode.name == 'GompertzODE_par_a':
        return gompertz_interp_config
    elif ode.name == 'GompertzODE_par_b':
        return gompertz_interp_config
    elif ode.name == 'GompertzODE_par_ab':
        return gompertz_interp_config
    elif ode.name == 'Lorenz' or ode.name == 'Lorenz_sigma' or ode.name == 'Lorenz_rho' or ode.name == 'Lorenz_beta':
        if x_id == 0:
            return lorenz_interp_config
        else:
            return lorenz_interp_config2
    elif ode.name == 'LogisticODE' or ode.name == 'LogisticODE_a' or ode.name == 'LogisticODE_k':
        return logistic_interp_config
    elif ode.name == 'OscillatingLogisticODE':
        return oscillating_logistic_interp_config
    elif ode.name == 'SelkovODE' or ode.name == 'SelkovODE_rho' or ode.name == 'SelkovODE_sigma' or ode.name == 'SelkovODE_rho_03' or ode.name == 'SelkovODE_rho_04' or ode.name == 'SelkovODE_rho_06' or ode.name == 'SelkovODE_rho_09' or ode.name == 'OscillatingSelkovODE' or ode.name == 'OscillatingSelkovODE_d':
        return selkov_interp_config
    elif ode.name == 'real':
        return real_interp_config
    elif ode.name == 'HillODE':
        return hill_interp_config
    elif ode.name == 'FracODE':
        return frac_interp_config
    elif ode.name == 'TrigonometricODE':
        return trigonometric_interp_config
    else:
        raise ValueError



gompertz_interp_config = {
    'r': 2,
    'sigma_in_mul': 2.,
    'freq_int': 20,
    'new_sample': 5,
    'n_basis': 50,
    'basis': basis.FourierBasis,
}

real_interp_config = {
    'r': 4,
    'sigma_in': 0.15,
    'freq_int': 365*2,
    'new_sample': 5,
    'n_basis': 50,
    'basis': basis.FourierBasis,
}

logistic_interp_config = {
    'r': 3,
    'sigma_in_mul': 2.,
    'freq_int': 20,
    'new_sample': 5,
    'n_basis': 50,
    'basis': basis.FourierBasis,
}

oscillating_logistic_interp_config = {
    'r': 3,
    'sigma_in_mul': 2.,
    'freq_int': 30, # 20 # 40
    'new_sample': 5,
    'n_basis': 50,
    'basis': basis.FourierBasis,
}

lorenz_interp_config = {
    'r': -1,
    'sigma_in_mul': 2.,
    'freq_int': 100,
    'new_sample': 5,
    'n_basis': 50,
    'basis': basis.FourierBasis,
}

lorenz_interp_config2 = {
    'r': -1,
    'sigma_in': 0.1,
    'freq_int': 100,
    'new_sample': 5,
    'n_basis': 50,
    'basis': basis.FourierBasis,
}

selkov_interp_config = {
    'r': 2,
    'sigma_in': 0.2,
    'freq_int': 30, # 20 # TODO
    'new_sample': 5,
    'n_basis': 50,
    'basis': basis.FourierBasis,
}

hill_interp_config = {
    'r': 3,
    'sigma_in_mul': 2.,
    'freq_int': 20,
    'new_sample': 5,
    'n_basis': 50,
    'basis': basis.FourierBasis,
}

frac_interp_config = {
    'r': 7,
    'sigma_in': 0.01,
    'freq_int': 20,
    'new_sample': 5,
    'n_basis': 50,
    'basis': basis.CubicSplineBasis,
}

trigonometric_interp_config = {
    'r': 2,
    'sigma_in_mul': 2.,
    'freq_int': 20,
    'new_sample': 5,
    'n_basis': 50,
    'basis': basis.FourierBasis,
}



gompertz_config = {'population_size': 15000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   #'function_set': {'neg': 1, 'mul': 3, 'log': 1, 'add': 1}, # original function set
                   'function_set': {'neg': 1, 'mul': 1, 'log': 1, 'add': 1, 'sub': 1, 'div': 1, 'pow': 1, 'sin': 1, 'cos': 1},
                   'const_range': (1, 2),
                   #'generations': 20, # original number of generations
                   'generations': 6,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

gompertz_par_a_config = {'population_size': 30000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   'function_set': {'neg': 1, 'mul': 3, 'log': 1, 'add': 1},
                   'const_range': (1, 2),
                   'generations': 20,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

gompertz_par_b_config = {'population_size': 30000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   'function_set': {'neg': 1, 'mul': 1, 'log': 1, 'add': 1, 'sub': 1, 'div': 1, 'pow': 1, 'sin': 1, 'cos': 1},
                   'const_range': (1, 2),
                   'generations': 6, 
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

gompertz_par_ab_config = {'population_size': 30000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   'function_set': {'neg': 1, 'mul': 1, 'log': 1, 'add': 1, 'sub': 1, 'div': 1, 'pow': 1, 'sin': 1, 'cos': 1},
                   'const_range': (1, 2),
                   'generations': 6,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

real_config = {'population_size': 15000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   'function_set': {'neg': 1, 'mul': 3, 'log': 1, 'add': 1},
                   'const_range': (0, 10),
                   'generations': 20,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

logistic_config = {'population_size': 15000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   #'function_set': {'sub': 1, 'mul': 1, 'pow': 1, 'add': 1},
                   #'function_set': {'sub': 1, 'mul': 1, 'pow': 1, 'add': 1, 'neg': 1, 'div': 1, 'log': 1, 'sin': 1, 'cos': 1}, # corretta
                   'function_set': {'sub': 1, 'mul': 1, 'pow': 1, 'add': 1, 'neg': 1, 'div': 1, 'log': 1}, # dev, per oscillating prima parte
                   'const_range': (1., 2.),
                   #'generations': 20,
                   'generations': 6,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

logistic_config_k = {'population_size': 20000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   'function_set': {'sub': 1, 'mul': 1, 'pow': 1, 'add': 1, 'neg': 1, 'div': 1, 'log': 1, 'sin': 1, 'cos': 1},
                   'const_range': (1., 2.),
                   'generations': 6,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }


selkov_config = {'population_size': 30000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   #'function_set': {'sub': 1, 'mul': 1, 'add': 1}, # original function set
                   'function_set': {'neg': 1, 'mul': 1, 'log': 1, 'add': 1, 'sub': 1, 'div': 1, 'pow': 1, 'sin': 1, 'cos': 1}, 
                   'const_range': (0.01, 1.), # the range of constants to include in the formulas -> we specify a unique range for all constants 
                   #'generations': 20, # original number of generations
                   'generations': 6,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01, # 0.001
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

oscillating_selkov_config = {'population_size': 30000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   #'function_set': {'sub': 1, 'mul': 1, 'add': 1, 'neg': 1, 'sin': 1, 'pow': 1}, 
                   'function_set': {'sub': 1, 'mul': 1, 'add': 1, 'neg': 1, 'sin': 1, 'pow': 1, 'log': 1}, 
                   'const_range': (0.01, 4.), # x oscillating 
                   'generations': 6,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

lorenz_config_x0 = {'population_size': 20000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   'function_set': {'sub': 1, 'mul': 3, 'add': 1},
                   'const_range': (5, 15),
                   'generations': 20,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

lorenz_config_x1 = {'population_size': 20000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   'function_set': {'sub': 1, 'mul': 3, 'add': 1},
                   'const_range': (25, 30),
                   'generations': 20,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

lorenz_config_x2 = {'population_size': 20000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   'function_set': {'sub': 1, 'mul': 3, 'add': 1},
                   'const_range': (1.1, 5),
                   'generations': 20,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

# dev: 
oscillating_logistic_config = {'population_size': 15000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   #'function_set': {'sub': 1, 'mul': 1, 'add': 1, 'neg': 1, 'sin': 1, 'pow': 1, 'log': 1},
                   'function_set': {'sub': 1, 'mul': 1, 'add': 1, 'neg': 1, 'sin': 1, 'pow': 1},
                   #'function_set': {'sub': 1, 'mul': 1, 'add': 1, 'neg': 1, 'sin': 1},  
                   'const_range': (0.1, 7.), # !!!
                   #'generations': 20,
                   'generations': 11,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

gompertz_config_no_coef = {'population_size': 15000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   'function_set': {'neg': 1, 'mul': 3, 'log': 1, 'add': 1},
                   'const_range': None,
                   'generations': 20,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }


hill_config = {'population_size': 15000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   'function_set': {'neg': 1, 'mul': 1, 'pow': 1, 'add': 1, 'div': 1},
                   'const_range': (1, 2),
                   'generations': 20,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 1,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

frac_config = {'population_size': 15000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   'function_set': {'neg': 1, 'mul': 1, 'div': 1, 'add': 1, 'pow': 1},
                   'const_range': (1, 2),
                   'generations': 20,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 0,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }

trigonometric_config = {'population_size': 15000,
                   'p_crossover': 0.6903,
                   'p_subtree_mutation': 0.133,
                   'p_hoist_mutation': 0.0361,
                   'p_point_mutation': 0.0905,
                   #'function_set': {'neg': 1, 'mul': 3, 'log': 1, 'add': 1},
                   'function_set': {'neg': 1, 'mul': 3, 'add': 1, 'sin': 1, 'cos': 1},
                   #'function_set': {'neg': 1, 'mul': 3, 'log': 1, 'add': 1, 'div': 1, 'pow': 1, 'sin': 1, 'cos': 1},
                   'const_range': (1, 2),
                   #'generations': 20,
                   'generations': 6,
                   'stopping_criteria': 0.01,
                   'max_samples': 0.9,
                   'verbose': 0,
                   'parsimony_coefficient': 0.01,
                   'init_depth': (1, 6),
                   'n_jobs': 2,
                   'low_memory': True
                   }
