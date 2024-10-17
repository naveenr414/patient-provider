from patient.utils import solve_linear_program
import numpy as np

def optimal_order_policy(simulator):
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
    
    menu = np.zeros(weights.shape)
    for i in range(len(directed_acyclic_ordering)):
        for j in range(i+1):
            if i == j or directed_acyclic_ordering[j] in adjacency_edges[directed_acyclic_ordering[i]]:
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

