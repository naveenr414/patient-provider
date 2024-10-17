import numpy as np 
from patient.utils import solve_linear_program
import gurobipy as gp
from gurobipy import GRB
import itertools 

def lp_policy(simulator):
    """Policy which selects according to the LP, in an offline fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
    
    Returns: List of providers on the menu"""
    weights = np.array([p.provider_rewards for p in simulator.patients])

    max_per_provider = simulator.provider_max_capacity
    LP_solution = solve_linear_program(weights,max_per_provider)

    matchings = np.zeros((len(simulator.patients),simulator.num_providers))

    for (i,j) in LP_solution:
        matchings[i,j] = 1
    return matchings  

def lp_multiple_match_policy(simulator):
    """Policy which selects according to the LP
        Allows for multiple patients to be matched with a provider
    
    Arguments:
        simulator: Simulator for patient-provider matching
    
    Returns: List of providers on the menu"""
    weights = np.array([p.provider_rewards for p in simulator.patients])
    N, M = weights.shape

    max_per_provider = simulator.provider_max_capacity*max(N//M,1)
    LP_solution = solve_linear_program(weights,max_per_provider)

    matchings = np.zeros((len(simulator.patients),simulator.num_providers))

    for (i,j) in LP_solution:
        matchings[i,j] = 1
    return matchings  


def lp_fairness_policy(simulator):
    """Policy which selects according to the LP, in an offline fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
    
    Returns: List of providers on the menu"""
    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)

    max_per_provider = simulator.provider_max_capacity
    N,P = weights.shape 

    m = gp.Model("bipartite_matching")
    m.setParam('OutputFlag', 0)
    x = m.addVars(N, P, vtype=GRB.BINARY, name="x")

    l = m.addVars(1,name="l")

    m.setObjective(l[0], GRB.MAXIMIZE)

    for j in range(P):
        m.addConstr(gp.quicksum(x[i, j] for i in range(N)) <= max_per_provider, name=f"match_{j}_limit")

    for i in range(N):
        m.addConstr(gp.quicksum(x[i, j] for j in range(P)) <= 1, name=f"match_{j}")
        m.addConstr(gp.quicksum(x[i, j]*weights[i,j] for j in range(P)) >= l[0], name=f"match_{j}")

    m.optimize()

    # Extract the solution
    solution = []
    for i in range(N):
        for j in range(P):
            if x[i, j].X > 0.5:
                solution.append((i, j))
    LP_solution = solution 

    matchings = np.zeros((len(simulator.patients),simulator.num_providers))

    for (i,j) in LP_solution:
        matchings[i,j] = 1
    return matchings  

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


def add_swap_matches_legacy(swap_score,matchings,pairs,max_menu_size):
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
