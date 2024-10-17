from patient.lp_policies import *
import numpy as np 
from patient.utils import solve_linear_program
from patient.lp_policies import compute_swap_scores, add_swap_matches
import gurobipy as gp
from gurobipy import GRB


def lp_more_patients_policy(simulator):
    """Policy that builds upon the LP solution in scenarios with N>M
        Here, it considers which patients should be added to each provider
        in an iterative fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
    
    Returns: List of providers on the menu"""

    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)

    max_per_provider = simulator.provider_max_capacity

    LP_solution = solve_linear_program(weights,max_per_provider)

    matchings = np.zeros((len(simulator.patients),simulator.num_providers))
    pairs = [-1 for i in range(len(simulator.patients))]
    unmatched_providers = set(list(range(simulator.num_providers)))
    unmatched_patients = set(list(range(len(matchings))))

    for (i,j) in LP_solution:
        matchings[i,j] = 1
        pairs[i] = j

        if j in unmatched_providers:
            unmatched_providers.remove(j)
        
        if i in unmatched_patients:
            unmatched_patients.remove(i)
    
    matches_by_provider = np.sum(matchings,axis=0)
    total = np.sum(weights*matchings,axis=0)

    p = simulator.choice_model_settings['top_choice_prob']
    for _ in range(len(matchings)):
        next_best_patient_score = [0 for i in range(simulator.num_providers)]
        next_best_patient = [-1 for i in range(simulator.num_providers)]

        for provider in range(simulator.num_providers):
            for _patient in unmatched_patients:
                score = weights[_patient][provider]
                if score > next_best_patient_score[provider] or next_best_patient[provider] == -1:
                    next_best_patient[provider] = _patient
                    next_best_patient_score[provider] = score

            benefit_new = (total[provider]+next_best_patient_score[provider])/(matches_by_provider[provider]+1) * (1-(1-p)**(matches_by_provider[provider]+1))
            benefit_old = (total[provider])/(matches_by_provider[provider]) * (1-(1-p)**(matches_by_provider[provider]))
            next_best_patient_score[provider] = benefit_new/benefit_old

        next_best_patient_score = zip(list(range(simulator.num_providers)),next_best_patient_score)
        next_best_patient_score = sorted(next_best_patient_score,key=lambda k: k[1],reverse=True)
        if max(next_best_patient) == -1 or next_best_patient_score[0][1] < 1 or len(unmatched_patients) == 0:
            break 
        
        for provider,score in next_best_patient_score:
            if score>1 and next_best_patient[provider] in unmatched_patients:
                matchings[next_best_patient[provider]][provider] = 1
                pairs[next_best_patient[provider]] = provider 
                unmatched_patients.remove(next_best_patient[provider])

    swap_score = compute_swap_scores(simulator,pairs,weights)
    matchings = add_swap_matches(swap_score,matchings,pairs,simulator.max_menu_size)        

    for i in range(len(matchings)):
        unmatched_provider_scores = [(j,weights[i][j]) for j in unmatched_providers]
        unmatched_provider_scores = sorted(unmatched_provider_scores,key=lambda k: k[1],reverse=True)
        unmatched_provider_scores = [j[0] for j in unmatched_provider_scores]

        if np.sum(matchings[i]) < simulator.max_menu_size:
            for j in unmatched_provider_scores[:int(simulator.max_menu_size-np.sum(matchings[i]))]:
                matchings[i][j] = 1
    
    return matchings  



def group_based_policy(simulator,legacy=False):
    """Policy which selects patients through the LP + additional swaps
    
    Arguments:
        simulator: Simulator for patient-provider matching
        legacy: Boolean, should we use the legacy swap score 
    
    Returns: List of providers on the menu"""

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)

    max_per_provider = simulator.provider_max_capacity
    LP_solution = solve_linear_program(weights,max_per_provider)

    matchings = np.zeros((len(simulator.patients),simulator.num_providers))
    pairs = [-1 for i in range(len(simulator.patients))]
    unmatched_providers = set(list(range(simulator.num_providers)))

    for (i,j) in LP_solution:
        matchings[i,j] = 1
        pairs[i] = j

        if j in unmatched_providers:
            unmatched_providers.remove(j)

    swap_score = compute_swap_scores(simulator,pairs,weights)
    if legacy:
        matchings = add_swap_matches_legacy(swap_score,matchings,pairs,simulator.max_menu_size)        
    else:
        matchings = add_swap_matches(swap_score,matchings,pairs,simulator.max_menu_size)        

    for i in range(len(matchings)):
        unmatched_provider_scores = [(j,weights[i][j]) for j in unmatched_providers]
        unmatched_provider_scores = sorted(unmatched_provider_scores,key=lambda k: k[1],reverse=True)
        unmatched_provider_scores = [j[0] for j in unmatched_provider_scores]

        if np.sum(matchings[i]) < simulator.max_menu_size:
            for j in unmatched_provider_scores[:int(simulator.max_menu_size-np.sum(matchings[i]))]:
                matchings[i][j] = 1

    return matchings 

def group_based_legacy_policy(simulator):
    """Older verion of policy which selects patients through the LP + additional swaps
    
    Arguments:
        simulator: Simulator for patient-provider matching
    
    Returns: List of providers on the menu"""

    return group_based_policy(simulator,legacy=True)    

def group_based_unidirectional_policy(simulator):
    """Policy which selects patients through the LP + additional swaps
    It does so with one-way swaps; this doesn't guaranteee match rate
    But could improve match quality
    
    Arguments:
        simulator: Simulator for patient-provider matching
    
    Returns: List of providers on the menu"""

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)

    max_per_provider = simulator.provider_max_capacity
    LP_solution = solve_linear_program(weights,max_per_provider)

    matchings = np.zeros((len(simulator.patients),simulator.num_providers))
    pairs = [-1 for i in range(len(simulator.patients))]
    unmatched_providers = set(list(range(simulator.num_providers)))

    for (i,j) in LP_solution:
        matchings[i,j] = 1
        pairs[i] = j

        if j in unmatched_providers:
            unmatched_providers.remove(j)

    swap_score = compute_swap_scores_unidirection(simulator,pairs,weights)
    for i in range(len(swap_score)):
        for j in range(len(swap_score)):
            if i != j:
                if swap_score[i][j]>0 and swap_score[i][j] + swap_score[j][i] > 0:
                    if pairs[i] != -1:
                        matchings[j][pairs[i]] = 1
                    if pairs[j] != -1:
                        matchings[i][pairs[j]] = 1
                if swap_score[i][j] > 0:
                    if pairs[j] != -1:
                        matchings[i][pairs[j]] = 1
    return matchings 