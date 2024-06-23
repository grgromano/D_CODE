import sympy
import numpy as np

import sys
import os
import re
from genetic import SymbolicODE
import equations
from utils import generator
from config import get_config, get_config_real

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.metrics import root_mean_squared_error


def gp_to_pysym_no_coef(est_gp, ode):
    VarDict = ode.get_var_dict()
    f_star = est_gp._program
    f_star_list, var_list, coef_list = parse_program_to_list(f_star.program)
    f_star_infix = generator.Generator.prefix_to_infix(f_star_list, variables=var_list, coefficients=coef_list)
    f_star_sympy = generator.Generator.infix_to_sympy(f_star_infix, VarDict, "simplify")
    return f_star_sympy


def mask_X(s):
    s = s.replace('X0', 'O')
    s = s.replace('X1', 'P')
    s = s.replace('X2', 'Q')
    s = s.replace('X3', 'R')
    return s

def back_X(s):
    s = s.replace('O', 'X0')
    s = s.replace('P', 'X1')
    s = s.replace('Q', 'X2')
    s = s.replace('R', 'X3')
    return s


def gp_to_pysym_with_coef(est_gp, ode, tol=None, tol2=None, expand=False, promising=False):
    VarDict = ode.get_var_dict()

    # RMK. estensione della funzione originale, per poter prendere in ingresso direttamente il programma genetico, anziché l'intero oggetto SymbolicRegressor
    if promising:
        f_star = est_gp
    else:
        f_star = est_gp._program
    
    f_star_list, var_list, coef_list = parse_program_to_list(f_star.program)
    # print(f_star_list)
    # print(var_list)
    # print(coef_list)

    f_star_infix = generator.Generator.prefix_to_infix(f_star_list, variables=var_list, coefficients=coef_list)
    f_star_infix2 = f_star_infix.replace('{', '').replace('}', '')
    if f_star_infix2 == f_star_infix:
        f_star_sympy = generator.Generator.infix_to_sympy(f_star_infix, VarDict, "simplify")
        return f_star_sympy

    f_star_sympy = generator.Generator.infix_to_sympy(f_star_infix2, VarDict, "simplify")

    if expand:
        f_star_sympy = sympy.expand(f_star_sympy)

    fs = str(f_star_sympy)
    print(fs)

    fs = mask_X(fs)
    if tol is None:
        fs = re.sub(r'([0-9]*\.[0-9]+|[0-9]+)', 'C', fs)
    else:
        consts = re.findall(r'([0-9]*\.[0-9]+|[0-9]+)', fs)
        for const in consts:
            if const in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
                continue
            if (float(const) < 1 + tol) and (float(const) > 1 - tol):
                fs = fs.replace(const, '1')
            elif (tol2 is not None) and (float(const) < tol2) and (float(const) > -1 * tol2):
                fs = fs.replace(const, '0')
            else:
                fs = fs.replace(const, 'C')

    fs = back_X(fs)
    print(fs)
    f_star_sympy = generator.Generator.infix_to_sympy(fs, VarDict, "simplify")
    return f_star_sympy


def check_equal(f1, f2):
    return sympy.simplify(f1 - f2) == 0


def parse_program_to_list(program):
    symbol_list = list()
    var_list = list()
    coef_list = list()

    for i in program:
        if isinstance(i, int):
            symbol_list.append('X' + str(i))
            var_list.append('X' + str(i))
        elif isinstance(i, float):
            symbol_list.append(str(i))
            coef_list.append(str(i))
        else:
            if i.name == 'log':
                symbol_list.append('ln')
            elif i.name == 'neg':
                symbol_list.append('sub')
                symbol_list.append('0')
            else:
                symbol_list.append(i.name)

    var_list = list(set(var_list))
    coef_list = list(set(coef_list))
    return symbol_list, var_list, coef_list


pow2 = make_function(lambda x: x ** 2, 'pow2', 1)
pow3 = make_function(lambda x: x ** 3, 'pow3', 1)


def run_gp(X_train, y_train, ode, x_id=0, seed=0):
    config = get_config(ode, x_id)

    est_gp = SymbolicRegressor(random_state=seed, **config)
    est_gp.fit(X_train, y_train)
    #print(est_gp._program) # identified function in LISP-form, e.g. neg(mul(X0, add(mul(X0, X0), log(X0))))

    if ode.name == 'LogisticODE':
        a = gp_to_pysym_with_coef(est_gp, ode, tol=0.05)
    elif ode.name == 'SelkovODE' or ode.name == 'SelkovODE_rho' or ode.name == 'SelkovODE_sigma' or ode.name == 'SelkovODE_rho_04' or ode.name == 'SelkovODE_rho_06' or ode.name == 'SelkovODE_rho_09':
        a = gp_to_pysym_with_coef(est_gp, ode, tol=0.05, expand=True)
    else:
        a = gp_to_pysym_with_coef(est_gp, ode)
    # try:
    # except Exception:
    #     a = None

    building_blocks = get_building_blocks(est_gp, ode)

    print('building blocks:')
    for i in range(len(building_blocks)):
        print(building_blocks[i])
    print('number of building blocks: ', len(building_blocks))
    print(' ')

    return a, est_gp


def run_gp_real(X_train, y_train, x_id=0, seed=0):
    config = get_config_real(x_id)
    ode = equations.RealODEPlaceHolder()
    est_gp = SymbolicRegressor(random_state=seed, **config)
    est_gp.fit(X_train, y_train)
    print(est_gp._program)
    a = gp_to_pysym_with_coef(est_gp, ode, tol=0.05, expand=True)
    return a, est_gp


# RMK. a differnza di run_gp, questa funzione usa la classe SymbolicODE, che implementa D-CODE
def run_gp_ode(ode_data, X_train, y_train, ode, x_id=0, seed=0):
    if ode.name != 'real':
        config = get_config(ode, x_id)
    else:
        config = get_config_real()

    est_gp = SymbolicODE(random_state=seed, **config)
    est_gp.fit(X_train, y_train, ode_data=ode_data)
    print(est_gp._program)

    # conversione del programma genetico nell'espressione simbolica, ossia la funzione f_hat dell'ODE
    if ode.name == 'SelkovODE' or ode.name == 'SelkovODE_rho' or ode.name == 'SelkovODE_sigma' or ode.name == 'SelkovODE_rho_04' or ode.name == 'SelkovODE_rho_06' or ode.name == 'SelkovODE_rho_09':
        a = gp_to_pysym_with_coef(est_gp, ode, tol=0.05, tol2=0.01, expand=True)
    else:
        a = gp_to_pysym_with_coef(est_gp, ode)

    
    building_blocks = get_building_blocks(est_gp, ode)

    print('building blocks:')
    for i in range(len(building_blocks)):
        print(building_blocks[i])
    print('number of building blocks: ', len(building_blocks))
    print(' ')
    
    return a, est_gp


def get_building_blocks(est_gp, ode):
    subprograms = est_gp.building_blocks

    tol = 1e-2 # tolerance for filtering similar subprograms 
    #testing_range = [0.3, 0.7] # range of x values for testing similarity

    N = 10
    #dim_x = ode.dim_x
    
    if ode.dim_x == 1:
        x_samples = np.random.uniform(ode.init_low, ode.init_high, N).reshape(-1, 1) 
    else:
        x_samples = np.empty((ode.dim_x, N))
        for i in range(ode.dim_x):
            x_samples[i, :] = np.random.uniform(ode.init_low[i], ode.init_high[i], N) 
    print(x_samples)
    
    subprograms_filtered = []
    #x_samples = np.linspace(testing_range[0], testing_range[1], 10).reshape(-1, 1) 
    f_samples = []
    for i in range(len(subprograms)):

        flag = 1
        f_hat = subprograms[i].execute 
        aux = f_hat(x_samples) #print(f_hat(x_samples))2 
        for j in range(len(f_samples)):
            if root_mean_squared_error(aux, f_samples[j]) < tol: # filter similar subprograms
                flag = 0
        if flag:
            f_samples.append(aux)
            subprograms_filtered.append(subprograms[i])

    return subprograms_filtered



# def gp_to_pysym_with_coef_2(est_gp, ode, tol=None, tol2=None, expand=False, promising=False):
#     VarDict = ode.get_var_dict()

#     # RMK. estensione della funzione originale, per poter prendere in ingresso direttamente il programma genetico, anziché l'intero oggetto SymbolicRegressor
#     if promising:
#         f_star = est_gp
#     else:
#         f_star = est_gp._program
    
#     f_star_list, var_list, coef_list = parse_program_to_list(f_star.program)
#     # print(f_star_list)
#     # print(var_list)
#     # print(coef_list)

#     f_star_infix = generator.Generator.prefix_to_infix(f_star_list, variables=var_list, coefficients=coef_list)
#     f_star_infix2 = f_star_infix.replace('{', '').replace('}', '')
#     if f_star_infix2 == f_star_infix:
#         f_star_sympy = generator.Generator.infix_to_sympy(f_star_infix, VarDict, "simplify")
#         return f_star_sympy

#     f_star_sympy = generator.Generator.infix_to_sympy(f_star_infix2, VarDict, "simplify")

#     if expand:
#         f_star_sympy = sympy.expand(f_star_sympy)

#     fs = str(f_star_sympy)
#     print(fs)

#     fs = mask_X(fs)
#     if tol:
#         consts = re.findall(r'([0-9]*\.[0-9]+|[0-9]+)', fs)
#         for const in consts:
#             if const in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
#                 continue
#             if (float(const) < 1 + tol) and (float(const) > 1 - tol):
#                 fs = fs.replace(const, '1')
#             elif (tol2 is not None) and (float(const) < tol2) and (float(const) > -1 * tol2):
#                 fs = fs.replace(const, '0')
#     fs = back_X(fs)
#     print(fs)
#     f_star_sympy = generator.Generator.infix_to_sympy(fs, VarDict, "simplify")
#     return f_star_sympy
