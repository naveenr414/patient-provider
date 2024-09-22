import random 
import numpy as np 
import gurobipy as gp
from gurobipy import GRB
from patient.learning import guess_coefficients
from patient.utils import solve_linear_program
import itertools

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
            if i == j:
                continue 

            p = simulator.choice_model_settings['top_choice_prob']

            weight_matrix = np.zeros((2,2))
            if pairs[i] >= 0:
                weight_matrix[0,0] = weights[i][pairs[i]]
                weight_matrix[1,0] = weights[j][pairs[i]]
            if pairs[j] >= 0:
                weight_matrix[0,1] = weights[i][pairs[j]]
                weight_matrix[1,1] = weights[j][pairs[j]]
            
            current_expected_reward = p*(weight_matrix[0,0] + weight_matrix[1,1])

            swap_reward = 0
            for order in [[0,1],[1,0]]:
                for coin_flips in [[0,0],[0,1],[1,0],[1,1]]:
                    prob = 1/2*np.prod([p**idx*(1-p)**(1-idx) for idx in coin_flips])
                    available_providers = [1,1]
                    for idx in range(2):
                        if coin_flips[idx] == 0:
                            continue 
                        else:
                            utilities = weight_matrix[order[idx]]*np.array(available_providers)
                            max_utility = max(utilities)
                            argmax = np.argmax(utilities)
                            swap_reward += prob*max_utility
                            available_providers[argmax] = 0
            
            score = swap_reward - current_expected_reward

            swap_score[i,j] = score
            swap_score[j,i] = score 

    return swap_score

def compute_swap_scores_unidirection(simulator,pairs,weights):
    """Compute the benefit from swapping/adding pairs of provider-patients
    
    Arguments: 
        simulator: Simulator for Patient-Provider pairs
        pairs: LP Matches between patients and providers
        weights: The utilities for each patient-provider pairs
    
    Returns: Numpy array; the benefit of adding on pairs of patients"""

    swap_score = np.zeros((len(simulator.patients),len(simulator.patients)))

    for i in range(len(simulator.patients)):
        for j in range(len(simulator.patients)):
            if i == j:
                continue 

            p = simulator.choice_model_settings['top_choice_prob']

            weight_matrix = np.zeros((2,2))
            if pairs[i] >= 0:
                weight_matrix[0,0] = weights[i][pairs[i]]
                weight_matrix[1,0] = weights[j][pairs[i]]
            if pairs[j] >= 0:
                weight_matrix[0,1] = weights[i][pairs[j]]
                weight_matrix[1,1] = weights[j][pairs[j]]
            
            current_expected_reward = p*(weight_matrix[0,0] + weight_matrix[1,1])

            swap_reward = 0
            for order in [[0,1],[1,0]]:
                for coin_flips in [[0,0],[0,1],[1,0],[1,1]]:
                    prob = 1/2*np.prod([p**idx*(1-p)**(1-idx) for idx in coin_flips])
                    available_providers = np.array([[1,1],[0,1]])
                    for idx in range(2):
                        if coin_flips[idx] == 0:
                            continue 
                        else:
                            utilities = weight_matrix[order[idx]]*np.array(available_providers[idx])
                            max_utility = max(utilities)
                            argmax = np.argmax(utilities)
                            swap_reward += prob*max_utility
                            available_providers[idx][argmax] = 0
            
            score = swap_reward - current_expected_reward

            swap_score[i,j] = score

    return swap_score


def add_swap_matches(swap_score,matchings,pairs,max_menu_size):
    """Add all pairs and triplets based on the swap score
    Arguments: 
        simulator: Simulator for patient-provider interactions
        swap_score: Numpy array, with the expected value of adding provider
            j to patient i
        matchings: Matching score/utility for patient-provider pairs
        pairs: List of providers matched for each patient by the LP
    
    Returns: New Matchings after adding in pairs and triplets"""

    used_indices = set() 
    while len(used_indices) < len(matchings):
        all_available_patients = [i for i in range(len(matchings)) if i not in used_indices]
        model = gp.Model("Complete_Graph_Subset")
        num_nodes = len(all_available_patients)
        x = model.addVars(num_nodes,num_nodes, vtype=GRB.BINARY, lb=0, ub=1, name="x") 
        y = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y") 
        model.setParam('OutputFlag', 0)

        # Set objective: maximize sum of weights w_{i,j} * x_{i,j}
        model.setObjective(gp.quicksum(swap_score[i, j] * x[i, j] for i in range(num_nodes) for j in range(num_nodes)), GRB.MAXIMIZE)

        # Add constraints
        for i in range(num_nodes):
            for j in range(num_nodes):
                # x_{i,j} <= y_{i} and x_{i,j} <= y_{j}
                model.addConstr(x[i, j] <= y[i], f"x_{i}_{j}_leq_y_{i}")
                model.addConstr(x[i, j] <= y[j], f"x_{i}_{j}_leq_y_{j}")
                # x_{i,j} >= y_{i} + y_{j} - 1
                model.addConstr(x[i, j] >= y[i] + y[j] - 1, f"x_{i}_{j}_geq_y_{i}_plus_y_{j}_minus_1")
        model.addConstr(gp.quicksum(y[i] for i in range(num_nodes)) <= max_menu_size)

        # Optimize the model
        model.optimize()

        new_nodes = set()
        for i in range(num_nodes):
            for j in range(num_nodes):
                if x[i, j].X > 0.5:
                    new_nodes.add(all_available_patients[i])
                    new_nodes.add(all_available_patients[j])
        
        combo = list(new_nodes)
        score = 0

        for i,j in itertools.combinations(list(combo),2):
            score += swap_score[i][j] 
        if score > 0:
            for i,j in itertools.combinations(list(combo),2):
                if pairs[j] >= 0:
                    matchings[i][pairs[j]] = 1
                if pairs[i] >= 0:
                    matchings[j][pairs[i]] = 1
            for i in list(combo):
                used_indices.add(i)

        else:
            break  
    return matchings

def p_approximation_with_additions(simulator):
    """Policy which selects patients through the LP + additional swaps
    
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

def p_approximation_with_additions_loose_constraints(simulator):
    """Policy which selects patients through the LP + additional swaps
    
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

    swap_score = compute_swap_scores(simulator,pairs,weights)
    matchings = add_swap_matches(swap_score,matchings,pairs,simulator.max_menu_size)        

    for i in range(len(matchings)):
        unmatched_provider_scores = [(j,weights[i][j]) for j in unmatched_providers]
        unmatched_provider_scores = sorted(unmatched_provider_scores,key=lambda k: k[1],reverse=True)
        unmatched_provider_scores = [j[0] for j in unmatched_provider_scores]

        if np.sum(matchings[i]) < simulator.max_menu_size:
            for j in unmatched_provider_scores[:int(simulator.max_menu_size-np.sum(matchings[i]))]:
                matchings[i][j] = 1

    memory = matchings 

    return matchings 


def p_approximation_with_additions_no_match(simulator):
    """Policy which selects patients through the LP + additional swaps
    
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


def p_approximation_with_additions_extra_provider(simulator,patient,available_providers,memory,per_epoch_function):
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
        LP_solution = solve_linear_program(weights,max_per_provider)

        matchings = np.zeros((len(simulator.patients),len(available_providers)),dtype=int)
        pairs = [-1 for i in range(len(simulator.patients))]
        unmatched_providers = set(list(range(len(available_providers))))

        for (i,j) in LP_solution:
            matchings[i,j] = 1
            pairs[i] = j

            if j in unmatched_providers:
                unmatched_providers.remove(j)
    
        swap_score = compute_swap_scores(simulator,pairs,weights)
        add_swap_matches(swap_score,matchings,pairs,simulator.max_menu_size)        

        for i in range(len(matchings)):
            unmatched_provider_scores = [(j,weights[i][j]) for j in unmatched_providers]
            unmatched_provider_scores = sorted(unmatched_provider_scores,key=lambda k: k[1],reverse=True)
            unmatched_provider_scores = [j[0] for j in unmatched_provider_scores]

            if np.sum(matchings[i]) < simulator.max_menu_size:
                for j in unmatched_provider_scores[:int(simulator.max_menu_size-np.sum(matchings[i]))]:
                    matchings[i][j] = 1

        matchings = np.array(matchings)
        p = simulator.choice_model_settings['top_choice_prob']
        for j in range(simulator.num_providers):
            num_matches = np.sum(matchings[:,j])
            avg_match_utility = np.sum(matchings[:,j]*weights[:,j])/num_matches

            unmatched_patients = [(i,weights[i,j]) for i in range(simulator.num_patients) if matchings[i,j] == 0]
            unmatched_patients = sorted(unmatched_patients,key=lambda k: k[1],reverse=True)

            for (i,util) in unmatched_patients:
                p_sum_plus_1 = sum([p*(1-p)**j for j in range(num_matches)])
                p_sums = sum([p*(1-p)**j for j in range(num_matches-1)])

                avg_plus_1 = num_matches/(num_matches+1)*avg_match_utility + 1/(num_matches+1)*util
                
                if p_sum_plus_1*avg_plus_1 > p_sums*avg_match_utility:
                    matchings[i,j] = 1
                    num_matches += 1
                    avg_match_utility = avg_plus_1
                else:
                    break 

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
        add_swap_matches(swap_score,matchings,pairs,simulator.max_menu_size)

        for i in range(len(matchings)):
            unmatched_provider_scores = [(j,weights[i][j]) for j in unmatched_providers]
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
        add_swap_matches(swap_score,matchings,pairs,simulator.max_menu_size)

        for i in range(len(matchings)):
            unmatched_provider_scores = [(j,weights[i][j]) for j in unmatched_providers]
            unmatched_provider_scores = sorted(unmatched_provider_scores,key=lambda k: k[1],reverse=True)
            unmatched_provider_scores = [j[0] for j in unmatched_provider_scores]

            if np.sum(matchings[i]) < simulator.max_menu_size:
                for j in unmatched_provider_scores[:int(simulator.max_menu_size-np.sum(matchings[i]))]:
                    matchings[i][j] = 1

        memory = matchings 

    default_menu = memory[patient.idx]
    
    return default_menu, memory 

def optimal_order_policy(simulator):
    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(weights)

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

    adjacency_edges = {}

    for i in range(N):
        adjacency_edges[i] = []

    for i in range(N):
        for j in range(N):
            if i != j:
                m_i = pairs[i] 
                m_j = pairs[j] 

                if m_i == -1 and m_j != -1:
                    adjacency_edges[i].append(j)
                elif m_i != -1 and m_j != -1 and weights[i][m_i] < weights[i][m_j]:
                    adjacency_edges[i].append(j)

    directed_acyclic_ordering = []
    marked_nodes = [False for i in range(N)]

    def dfs_recursive(start_node):
        if marked_nodes[start_node]:
            return 
        marked_nodes[start_node] = True 
        for j in adjacency_edges[start_node]:
            if not marked_nodes[j]:
                dfs_recursive(j)
        directed_acyclic_ordering.append(start_node)

    for i in range(N):
        dfs_recursive(i)

    directed_acyclic_ordering = directed_acyclic_ordering
    menu = np.zeros(weights.shape)
    for i in range(len(directed_acyclic_ordering)):
        for j in range(i+1):
            if pairs[directed_acyclic_ordering[j]] != -1:
                menu[directed_acyclic_ordering[i]][pairs[directed_acyclic_ordering[j]]] = 1
    return menu 