import random 
import numpy as np 
import itertools
import math


def random_policy(simulator,patient,available_providers,memory,per_epoch_function):
    """Randomly give a menu of available providers
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, 0-1 vector of which providers to show """
    random_provider = np.array([1 if random.random() < 0.5 else 0 for i in range(simulator.num_providers)])
    
    return random_provider, memory 

def all_ones_policy(simulator,patient,available_providers,memory,per_epoch_function):
    """A policy which shows all providers
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, which providers to show them """

    return [1 for i in range(simulator.num_providers)], memory 

def greedy_policy(simulator,patient,available_providers,memory,per_epoch_function):
    """A policy which shows providers greedily
        This shows the top based on the utility
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, which providers to show them """

    if simulator.max_menu_size >= simulator.num_providers:
        return [1 for i in range(simulator.num_providers)], memory 

    utilities = patient.provider_rewards
    menu = np.zeros(simulator.num_providers)
    top_menu = np.argsort(utilities)[-simulator.max_menu_size:][::-1]
    menu[top_menu] = 1
    return menu, memory 



def get_all_menus(N, M):
    """Given N patients and M providers, find all the 0-1 combinations
        of menus
        
    Arguments:
        N: integer, number of patients
        M: integer, number of providers
    
    Returns: List of numpy arrays of all the menus"""
    binary_strings = itertools.product([0, 1], repeat=N * M)
    return [np.array(arr).reshape(N, M) for arr in binary_strings]

def optimal_policy(simulator):
    """The optimal policy; we compute this by iterating through
        all combinations of policies and selecting the best one
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, which providers to show them """

    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(simulator.patients)
    M = weights.shape[1]

    all_orders = list(itertools.permutations(list(range(N))))
    all_selections = list(itertools.product([0, 1], repeat=N))
    probabilities = [p**(sum(i))*(1-p)**(N-sum(i)) for i in all_selections]

    scores = []
    all_menus = get_all_menus(N,M)

    for j,menu in enumerate(all_menus):
        total_reward = 0
        for ordering in all_orders:
            for s in range(len(all_selections)):
                selection = all_selections[s] 
                simulated_available_providers = np.ones(M)

                score = 0
                for i in range(N):
                    curr_patient = ordering[i]
                    available_providers = menu[curr_patient]*simulated_available_providers
                    
                    if np.sum(available_providers) == 0 or selection[i] == 0:
                        continue 
                    available_providers *= weights[curr_patient]
                    selected_provider = np.argmax(available_providers)
                    score += available_providers[selected_provider]
                    simulated_available_providers[selected_provider] = 0

                score *= probabilities[s]/(math.factorial(N))
                total_reward += score 
        scores.append(total_reward)
    scores = np.array(scores)
    scores/=N 
    best_menu = np.argmax(scores)
    menu = all_menus[best_menu]
    return menu 
