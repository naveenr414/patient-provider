import numpy as np 
import itertools
from patient.utils import solve_linear_program, solve_linear_program_online


def random_policy(parameters):
    """Randomly give a menu of available providers
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, 0-1 vector of which providers to show """
    weights = parameters['weights']
    N,M = weights.shape
    M-=1
    random_matrix = np.random.random((N,M))
    random_provider = np.round(random_matrix)
    return random_provider 

def greedy_policy(parameters):
    """A policy which shows providers greedily
        This shows the top based on the utility
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, which providers to show them """
    weights = parameters['weights']

    N,M = weights.shape
    M-=1

    return np.ones((N,M))


def get_all_menus(N, M):
    """Given N patients and M providers, find all the 0-1 combinations
        of menus
        
    Arguments:
        N: integer, number of patients
        M: integer, number of providers
    
    Returns: List of numpy arrays of all the menus"""
    binary_strings = itertools.product([0, 1], repeat=N * M)
    return [np.array(arr).reshape(N, M) for arr in binary_strings]

def get_fair_optimal_policy(fairness_constraint,seed):
    def policy(parameters):
        return optimal_policy(parameters,fairness_constraint=fairness_constraint,seed=seed)
    return policy 

def optimal_policy(parameters,fairness_constraint=-1,seed=43):
    """The optimal policy; we compute this by iterating through
        all combinations of policies and selecting the best one
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, which providers to show them """
    weights = parameters['weights']
    capacities = parameters['capacities']
    online_arrival = parameters['online_arrival']

    N,M = weights.shape 
    M -= 1 

    if online_arrival:
        lp_solution = solve_linear_program_online(weights,capacities)
    else:
        lp_solution = solve_linear_program(weights,capacities,fairness_constraint=fairness_constraint,seed=seed)
    assortment = np.zeros((N,M))
    
    for (i,j) in lp_solution:
        assortment[i,j] = 1
    
    return np.array(assortment)
