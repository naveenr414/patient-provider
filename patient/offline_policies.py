import random 
import numpy as np 
from patient.learning import guess_coefficients
from patient.utils import solve_linear_program

def offline_solution(simulator,patient,available_providers,memory,per_epoch_function):
    """Policy which selects according to the LP, in an offline fashion
    
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
        max_per_provider *= max(1,len(simulator.patients)/len(available_providers))

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
        max_per_provider *= max(1,len(simulator.patients)/len(available_providers))

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
