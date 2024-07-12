import random 
import numpy as np 
import gurobipy as gp
from gurobipy import GRB
from patient.learning import guess_coefficients
from patient.utils import solve_linear_program

def p_approximation(simulator,patient,available_providers,memory,per_epoch_function):
    """Policy which selects according to the LP, in an online fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""
    
    if memory == None:
        weights = [p.provider_rewards for p in simulator.patients]
        weights = np.array(weights)
        max_per_provider = simulator.provider_max_capacity
        max_per_provider *= max(1,len(simulator.patients)/len(available_providers))

        LP_solution = solve_linear_program(weights,max_per_provider)

        matchings = [0 for i in range(weights.shape[0])]
        for (i,j) in LP_solution:
            matchings[i] = j

        memory = matchings 

    default_menu = [0 for i in range(len(available_providers))]
    default_menu[memory[patient.idx]] = 1
    
    return default_menu, memory 

def p_approximation_balance(simulator,patient,available_providers,memory,per_epoch_function):
    """Policy which selects according to the LP, in an online fashion
        Adds in lamb=1 to account for balance
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""

    if memory == None:
        weights = [p.provider_rewards for p in simulator.patients]
        weights = np.array(weights)
        max_per_provider = simulator.provider_max_capacity
        max_per_provider *= max(1,len(simulator.patients)/len(available_providers))

        lamb = 1

        LP_solution = solve_linear_program(weights,max_per_provider,lamb=lamb)

        matchings = [0 for i in range(weights.shape[0])]
        for (i,j) in LP_solution:
            matchings[i] = j

        memory = matchings 

    default_menu = [0 for i in range(len(available_providers))]
    default_menu[memory[patient.idx]] = 1
    
    return default_menu, memory 

def compute_swap_scores(simulator,pairs,weights):
    """Compute the benefit from swapping/adding pairs of provider-patients
    
    Arguments: 
        simulator: Simulator for Patient-Provider pairs
        pairs: LP Matches between patients and providers
        weights: The utilities for each patient-provider pairs
    
    Returns: Numpy array; the benefit of adding on pairs of patients"""

    swap_score = np.zeros((len(simulator.patients),len(simulator.patients)))

    for i in range(len(simulator.patients)):
        for j in range(len(simulator.patients)):
            if pairs[i]>=0 and pairs[j] >= 0 and i != j:
                p = simulator.choice_model_settings['top_choice_prob']

                current_expected_reward = p*weights[i][pairs[i]] + p*weights[j][pairs[j]]

                swap_reward = 0
                for order in [[i,j],[j,i]]:
                    for coin_flips in [[0,0],[0,1],[1,0],[1,1]]:
                        prob = 1/2*np.prod([p**idx*(1-p)**(1-idx) for idx in coin_flips])
                        available_providers = [pairs[i],pairs[j]]
                        for idx in range(2):
                            if coin_flips[idx] == 0:
                                continue 
                            else:
                                utilities = [weights[order[idx]][provider] for provider in available_providers]
                                max_utility = max(utilities)
                                argmax = np.argmax(utilities)
                                swap_reward += prob*max_utility
                                available_providers.pop(argmax)
                score = swap_reward - current_expected_reward

                swap_score[i,j] = score
                swap_score[j,i] = score 
    return swap_score

def add_swap_matches(swap_score,matchings,pairs):
    """Add all pairs and triplets based on the swap score
    Arguments: 
        simulator: Simulator for patient-provider interactions
        swap_score: Numpy array, with the expected value of adding provider
            j to patient i
        matchings: Matching score/utility for patient-provider pairs
        pairs: List of providers matched for each patient by the LP
    
    Returns: New Matchings after adding in pairs and triplets"""

    non_zero = np.nonzero(swap_score>0)
    max_triplets = []

    for i in range(len(non_zero[0])):
        x,y = non_zero[0][i], non_zero[1][i]
        
        for j in range(len(matchings)):
            if j!= x and j!=y and pairs[j] >= 0 and pairs[x] >= 0 and pairs[y] >=0:
                scores = swap_score[x][j] + swap_score[j][y] + swap_score[x][y]
                if scores > 0:
                    max_triplets.append((scores,j,x,y))
    
    max_triplets = sorted(max_triplets,key=lambda k: k[0],reverse=True)
    used_indices = set() 

    for (_,a,b,c) in max_triplets:
        if a in used_indices or b in used_indices or c in used_indices:
            continue 
        matchings[b][pairs[a]] = 1
        matchings[c][pairs[a]] = 1
        matchings[a][pairs[b]] = 1
        matchings[c][pairs[b]] = 1
        matchings[b][pairs[c]] = 1
        matchings[a][pairs[c]] = 1
        used_indices.add(a)
        used_indices.add(b)
        used_indices.add(c)

    max_pairs = []
    for i in range(len(non_zero[0])):
        x,y = non_zero[0][i], non_zero[1][i]
        scores = swap_score[x][y]
        if pairs[x] >= 0 and pairs[y] >=0:
            max_pairs.append(((scores,x,y)))
    max_pairs = sorted(max_pairs,key=lambda k: k[0],reverse=True)
    for (_,a,b) in max_pairs:
        if a in used_indices or b in used_indices:
            continue 
        matchings[b][pairs[a]] = 1
        matchings[a][pairs[b]] = 1
        used_indices.add(a)
        used_indices.add(b)

    return matchings 

def p_approximation_with_additions(simulator,patient,available_providers,memory,per_epoch_function):
    """Policy which selects patients through the LP + additional swaps
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""

    if memory is None:
        weights = [p.provider_rewards for p in simulator.patients]
        weights = np.array(weights)

        max_per_provider = simulator.provider_max_capacity
        max_per_provider *= max(1,len(simulator.patients)/len(available_providers))
        LP_solution = solve_linear_program(weights,max_per_provider)

        matchings = np.zeros((len(simulator.patients),len(available_providers)))
        pairs = [-1 for i in range(len(simulator.patients))]
        unmatched_providers = set(list(range(len(available_providers))))

        for (i,j) in LP_solution:
            matchings[i,j] = 1
            pairs[i] = j

            if j in unmatched_providers:
                unmatched_providers.remove(j)
        
        swap_score = compute_swap_scores(simulator,pairs,weights)
        add_swap_matches(swap_score,matchings,pairs)

        for i in range(len(matchings)):
            unmatched_provider_scores = [(j,weights[i][j]) or j in unmatched_providers]
            unmatched_provider_scores = sorted(unmatched_provider_scores,key=lambda k: k[1],reverse=True)
            unmatched_provider_scores = [j[0] for j in unmatched_provider_scores]

            if np.sum(matchings[i]) < simulator.max_menu_size:
                for j in unmatched_provider_scores[:int(simulator.max_menu_size-np.sum(matchings[i]))]:
                    matchings[i][j] = 1

        memory = matchings 

    default_menu = memory[patient.idx]
    
    return default_menu, memory 

def p_approximation_with_additions_balance(simulator,patient,available_providers,memory,per_epoch_function):
    """Policy which selects patients through the LP + additional swaps
        Adds in balance via lamb=1
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""

    if memory is None:
        weights = [p.provider_rewards for p in simulator.patients]
        weights = np.array(weights)

        lamb = 1
        max_per_provider = simulator.provider_max_capacity
        max_per_provider *= max(1,len(simulator.patients)/len(available_providers))
        LP_solution = solve_linear_program(weights,max_per_provider,lamb=lamb)

        matchings = np.zeros((len(simulator.patients),len(available_providers)))
        pairs = [-1 for i in range(len(simulator.patients))]
        unmatched_providers = set(list(range(len(available_providers))))

        for (i,j) in LP_solution:
            matchings[i,j] = 1
            pairs[i] = j

            if j in unmatched_providers:
                unmatched_providers.remove(j)
        
        swap_score = compute_swap_scores(simulator,pairs,weights)
        add_swap_matches(swap_score,matchings,pairs)

        for i in range(len(matchings)):
            unmatched_provider_scores = [(j,weights[i][j]) or j in unmatched_providers]
            unmatched_provider_scores = sorted(unmatched_provider_scores,key=lambda k: k[1],reverse=True)
            unmatched_provider_scores = [j[0] for j in unmatched_provider_scores]

            if np.sum(matchings[i]) < simulator.max_menu_size:
                for j in unmatched_provider_scores[:int(simulator.max_menu_size-np.sum(matchings[i]))]:
                    matchings[i][j] = 1

        memory = matchings 

    default_menu = memory[patient.idx]
    
    return default_menu, memory 

def p_approximation_with_additions_balance_learning(simulator,patient,available_providers,memory,per_epoch_function):
    """Policy which selects patients through the LP + additional swaps
        Adds in balance via lamb=1, and learns the weights over time
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""

    if memory is None:
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

        lamb = 1
        max_per_provider = simulator.provider_max_capacity
        max_per_provider *= max(1,len(simulator.patients)/len(available_providers))
        LP_solution = solve_linear_program(weights,max_per_provider,lamb=lamb)

        matchings = np.zeros((len(simulator.patients),len(available_providers)))
        pairs = [-1 for i in range(len(simulator.patients))]
        unmatched_providers = set(list(range(len(available_providers))))

        for (i,j) in LP_solution:
            matchings[i,j] = 1
            pairs[i] = j

            if j in unmatched_providers:
                unmatched_providers.remove(j)
        
        swap_score = compute_swap_scores(simulator,pairs,weights)
        add_swap_matches(swap_score,matchings,pairs)

        for i in range(len(matchings)):
            unmatched_provider_scores = [(j,weights[i][j]) or j in unmatched_providers]
            unmatched_provider_scores = sorted(unmatched_provider_scores,key=lambda k: k[1],reverse=True)
            unmatched_provider_scores = [j[0] for j in unmatched_provider_scores]

            if np.sum(matchings[i]) < simulator.max_menu_size:
                for j in unmatched_provider_scores[:int(simulator.max_menu_size-np.sum(matchings[i]))]:
                    matchings[i][j] = 1

        memory = matchings 

    default_menu = memory[patient.idx]
    
    return default_menu, memory 
