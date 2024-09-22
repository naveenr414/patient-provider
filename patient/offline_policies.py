import random 
import numpy as np 
from patient.learning import guess_coefficients
from patient.utils import solve_linear_program
from patient.online_policies import compute_swap_scores, add_swap_matches
import itertools
import math 
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import differential_evolution



def offline_solution(simulator):
    """Policy which selects according to the LP, in an offline fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""
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
    print(matchings)
    return matchings  

def offline_solution_loose_constraints(simulator):
    """Policy which selects according to the LP, in an offline fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""

    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(simulator.patients)
    M = weights.shape[1]

    max_per_provider = simulator.provider_max_capacity/(p) * max(N/M,1)
    LP_solution = solve_linear_program(weights,max_per_provider)

    matchings = np.zeros((len(simulator.patients),simulator.num_providers))
    pairs = [-1 for i in range(len(simulator.patients))]
    unmatched_providers = set(list(range(simulator.num_providers)))

    for (i,j) in LP_solution:
        matchings[i,j] = 1
        pairs[i] = j

        if j in unmatched_providers:
            unmatched_providers.remove(j)
    
    return matchings  

def get_all_menus(N, M):
    binary_strings = itertools.product([0, 1], repeat=N * M)
    return [np.array(arr).reshape(N, M) for arr in binary_strings]


def one_shot_policy(simulator,patient,available_providers,memory,per_epoch_function):
    return per_epoch_function[patient.idx], memory 


def optimal_policy_epoch(simulator):
    """A policy which shows all providers
    
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

    for menu in all_menus:
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


def offline_solution_more_patients(simulator):
    """Policy which selects according to the LP, in an offline fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""

    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(simulator.patients)
    M = weights.shape[1]

    max_per_provider = simulator.provider_max_capacity *2

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

def offline_solution_2_more_patients(simulator):
    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(simulator.patients)
    M = weights.shape[1]

    orderings_by_provider = []

    for j in range(M):
        orderings_by_provider.append(np.argsort(weights[:,j])[::-1])

    curr_matchings = [[] for i in range(M)]
    curr_values = [[] for i in range(M)]
    idx = [0 for i in range(M)]
    curr_capacity = [0 for i in range(N)]

    min_matchings_per = 0
    max_matchings_per = 100
    matchings = np.zeros((N,M))

    while np.min(curr_capacity) < max_matchings_per:
        round_values = [np.mean(curr_values[i])*(1-(1-p)**(len(curr_matchings[i]))) for i in range(M)]
        round_values = np.array(round_values)
        round_values[np.isnan(round_values)] = 0

        for i in range(M):
            while idx[i]<len(orderings_by_provider[i]) and curr_capacity[orderings_by_provider[i][idx[i]]] >= max_matchings_per:
                idx[i] += 1
        
        addition_value = np.zeros(M)
        for i in range(M):
            if idx[i]>=N or curr_capacity[orderings_by_provider[i][idx[i]]] >= max_matchings_per:
                continue 
            new_value = np.sum(curr_values[i])+weights[orderings_by_provider[i][idx[i]]][i]
            new_value /= len(curr_matchings[i])+1
            new_value *= (1-(1-p)**(len(curr_matchings[i])+1))
            addition_value[i] = new_value - round_values[i] 
        
        next_add = np.argmax(addition_value)
        if addition_value[next_add] <= 0 and (idx[next_add] >= N or orderings_by_provider[next_add][idx[next_add]] >= min_matchings_per):
            break 
            
        matchings[orderings_by_provider[next_add][idx[next_add]]][next_add] = 1
        curr_capacity[orderings_by_provider[next_add][idx[next_add]]] += 1
        curr_matchings[next_add].append(idx[next_add])
        curr_values[next_add].append(weights[orderings_by_provider[next_add][idx[next_add]]][next_add])
        idx[next_add] += 1
        
    return matchings 

def offline_learning_solution(simulator,patient,available_providers,memory,per_epoch_function):
    """Policy which selects according to the LP, in an offline fashion
        Additionally learns the weights over time
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""

    if memory == None:
        patient_contexts = np.array([p.patient_vector for p in simulator.patients])
        provider_contexts = np.array(simulator.provider_vectors)

        matched_pairs = simulator.matched_pairs
        unmatched_pairs = simulator.unmatched_pairs
        preference_pairs = simulator.preference_pairs

        predicted_coeff = guess_coefficients(matched_pairs,unmatched_pairs,preference_pairs,simulator.context_dim)
                
        weights = np.zeros((simulator.num_patients,simulator.num_providers))
        for i in range(simulator.num_patients):
            for j in range(simulator.num_providers):
                weights[i,j] = (1-np.abs(patient_contexts[i]-provider_contexts[j])).dot(predicted_coeff)/(np.sum(predicted_coeff))

        max_per_provider = simulator.provider_max_capacity

        LP_solution = solve_linear_program(weights,max_per_provider)

        matchings = [0 for i in range(weights.shape[0])]
        for (i,j) in LP_solution:
            matchings[i] = j
        memory = matchings 

    weights = [p.provider_rewards for p in simulator.patients]

    unmatched_providers = set(list(range(len(available_providers))))
    for i in range(len(memory)):
        if memory[i] >= 0 and memory[i] in unmatched_providers:
            unmatched_providers.remove(memory[i])

    default_menu = [0 for i in range(len(available_providers))]

    if memory[patient.idx] >= 0:
        default_menu[memory[patient.idx]] = 1
    
    unmatched_weights = [(i,weights[patient.idx][i]) for i in unmatched_providers]
    unmatched_weights = [(i,j) for (i,j) in unmatched_weights if j > 0]
    unmatched_weights = sorted(unmatched_weights,key=lambda k: k[1],reverse=True)
    for i in range(min(len(unmatched_weights),simulator.max_menu_size-1)):
        default_menu[unmatched_weights[i][0]] = 1
    
    num_added = np.sum(default_menu)
    idx = list(simulator.patient_order).index(patient.idx)-1 
    while num_added < simulator.max_menu_size and idx >= 0:
        curr_patient = simulator.patient_order[idx]
        if memory[curr_patient] >= 0 and (memory[patient.idx] < 0 or weights[patient.idx][memory[curr_patient]] >= weights[patient.idx][memory[patient.idx]]):
            default_menu[memory[simulator.patient_order[idx]]] = 1
            num_added += 1
        idx -=1

    idx = list(simulator.patient_order).index(patient.idx)-1 
    while num_added < simulator.max_menu_size and idx >= 0:
        curr_patient = simulator.patient_order[idx]
        if memory[curr_patient] >= 0:
            default_menu[memory[simulator.patient_order[idx]]] = 1
            num_added += 1
        idx -=1

    return default_menu, memory 


def offline_solution_balance(simulator,patient,available_providers,memory,per_epoch_function):
    """Policy which selects according to the LP, in an offline fashion
        Adds in lamb=1 to account for balance
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""

    if memory == None:
        lamb = 1
        weights = [p.provider_rewards for p in simulator.patients]
        weights = np.array(weights)

        max_per_provider = simulator.provider_max_capacity

        LP_solution = solve_linear_program(weights,max_per_provider,lamb)

        matchings = [0 for i in range(weights.shape[0])]

        for (i,j) in LP_solution:
            matchings[i] = j
        memory = matchings 

    weights = [p.provider_rewards for p in simulator.patients]

    unmatched_providers = set(list(range(len(available_providers))))
    for i in range(len(memory)):
        if memory[i] >= 0 and memory[i] in unmatched_providers:
            unmatched_providers.remove(memory[i])

    default_menu = [0 for i in range(len(available_providers))]

    if memory[patient.idx] >= 0:
        default_menu[memory[patient.idx]] = 1
    
    unmatched_weights = [(i,weights[patient.idx][i]) for i in unmatched_providers]
    unmatched_weights = [(i,j) for (i,j) in unmatched_weights if j > 0]
    unmatched_weights = sorted(unmatched_weights,key=lambda k: k[1],reverse=True)
    for i in range(min(len(unmatched_weights),simulator.max_menu_size-1)):
        default_menu[unmatched_weights[i][0]] = 1
    
    num_added = np.sum(default_menu)
    idx = list(simulator.patient_order).index(patient.idx)-1 
    while num_added < simulator.max_menu_size and idx >= 0:
        curr_patient = simulator.patient_order[idx]
        if memory[curr_patient] >= 0 and (memory[patient.idx] < 0 or weights[patient.idx][memory[curr_patient]] >= weights[patient.idx][memory[patient.idx]]):
            default_menu[memory[simulator.patient_order[idx]]] = 1
            num_added += 1
        idx -=1

    idx = list(simulator.patient_order).index(patient.idx)-1 
    while num_added < simulator.max_menu_size and idx >= 0:
        curr_patient = simulator.patient_order[idx]
        if memory[curr_patient] >= 0:
            default_menu[memory[simulator.patient_order[idx]]] = 1
            num_added += 1
        idx -=1

    return default_menu, memory 
