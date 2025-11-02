import random 
import numpy as np 
import itertools
import math
from patient.utils import solve_linear_program


def random_policy(weights,max_per_provider):
    """Randomly give a menu of available providers
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, 0-1 vector of which providers to show """
    N,M = weights.shape
    M-=1
    random_matrix = np.random.random((N,M))
    random_provider = np.round(random_matrix)
    return random_provider 

def greedy_policy(weights,max_per_provider):
    """A policy which shows providers greedily
        This shows the top based on the utility
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, which providers to show them """
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

def optimal_policy(weights,max_per_provider):
    """The optimal policy; we compute this by iterating through
        all combinations of policies and selecting the best one
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, which providers to show them """

    N,M = weights.shape 
    M -= 1 
    lp_solution = solve_linear_program(weights,max_per_provider)
    assortment = np.zeros((N,M))
    
    for (i,j) in lp_solution:
        assortment[i,j] = 1
    
    return np.array(assortment)
