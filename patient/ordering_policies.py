from patient.utils import solve_linear_program
import numpy as np
from copy import deepcopy
import random
from patient.provider_policies import provider_focused_less_interference_policy

def compute_optimal_divisions(directed_acyclic_ordering,adjacency_edges,num_divisions):
    """Compute the location where we should split up the DAG
        We aim to minimize the number internal edges
    Arguments:
        directed_acyclic_ordering: List of nodes, the order of the DAG
        adjacency_edges: Dictionary, with the output nodes for each input node
        num_divisions: The number of divisions we can put up
        
    Returns: List of divisions, integers, of size (num_divisions+1), with l[0] = 0"""
    internal_edges = np.zeros((len(directed_acyclic_ordering),len(directed_acyclic_ordering)))
    for start in range(len(directed_acyclic_ordering)):
        for end in range(start+1,len(directed_acyclic_ordering)):
            curr_nodes = directed_acyclic_ordering[start:end]
            internal_edges[start,end] = internal_edges[start,end-1] + len(set(curr_nodes).intersection(set(adjacency_edges[directed_acyclic_ordering[end]])))

    dp_array = [[0 for i in range(len(directed_acyclic_ordering)+1)] for i in range(num_divisions+1)]
    decision_array = [[-1 for i in range(len(directed_acyclic_ordering)+1)] for i in range(num_divisions+1)]

    for n in range(num_divisions+1):
        for i in range(len(directed_acyclic_ordering)-1,-1,-1):
            if n <= 1:
                dp_array[n][i] = internal_edges[i,len(directed_acyclic_ordering)-1]
                decision_array[n][i] = len(directed_acyclic_ordering)-1
            elif i == len(directed_acyclic_ordering)-1:
                dp_array[n][i] = 0
                decision_array[n][i] = len(directed_acyclic_ordering)-1
            elif i == len(directed_acyclic_ordering):
                dp_array[n][i] = 0
            else:
                next_places = [dp_array[n-1][j+1] + internal_edges[i,j] for j in range(i,len(directed_acyclic_ordering))]
                dp_array[n][i] = min(next_places)
                decision_array[n][i] = np.argmin(next_places) + i
  
    best_order = [-1]
    i = 0
    n = num_divisions
    while i<len(directed_acyclic_ordering)-1:
        best_order.append(decision_array[n][i])
        i = decision_array[n][i]
        n -= 1
                
    return sorted(list(set(best_order)))

def optimal_order_policy(simulator,num_divisions=2):
    """Compute the policy, assumign that we follow the optimal order
    
    Arguments:
        simulator: Simulator for patient-provider matching
    
    Returns: List of providers on the menu"""

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
    num_divisions = min(simulator.batch_size,len(directed_acyclic_ordering))
    divisions = compute_optimal_divisions(directed_acyclic_ordering,adjacency_edges,num_divisions)
    num_divisions = len(divisions)-1
    division_by_num = []
    curr_division = 0
    for i in range(len(directed_acyclic_ordering)):
        if i == divisions[curr_division]+1:
            curr_division += 1
        division_by_num.append(curr_division)

    simulator.custom_patient_order = []
    for _ in range(simulator.num_trials):
        new_ordering = deepcopy(directed_acyclic_ordering)
        for i in range(num_divisions):
            new_ordering[divisions[i]+1:divisions[i+1]+1] = random.sample(new_ordering[divisions[i]+1:divisions[i+1]+1],divisions[i+1]-divisions[i])
        simulator.custom_patient_order.append(new_ordering)

    menu = np.ones(weights.shape)
    for i in range(len(directed_acyclic_ordering)):
        for j in range(0,i+1):
            if i == j or division_by_num[i] > division_by_num[j]:
                menu[directed_acyclic_ordering[i]][pairs[directed_acyclic_ordering[j]]] = 1
    return menu 


def compute_optimal_order(simulator):
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
    
    return directed_acyclic_ordering
