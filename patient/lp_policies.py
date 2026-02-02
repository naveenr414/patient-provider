import numpy as np 
from patient.utils import solve_linear_program
import gurobipy as gp
from gurobipy import GRB
import itertools 
import torch
import torch.nn as nn
import torch.nn.functional as F
import json 
import random 


def lp_policy(parameters,fairness_constraint=-1):
    """Policy which selects according to the LP, in an offline fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
    
    Returns: List of providers on the menu"""
    weights = parameters['weights']
    capacities = parameters['capacities']
    N,M = weights.shape 
    M -= 1 
    lp_solution = solve_linear_program(weights,capacities)
    assortment = np.zeros((N,M))
    
    for (i,j) in lp_solution:
        assortment[i,j] = 1
    
    return np.array(assortment)

def simulate_assortment(X_sol, weights_list, capacities, permutations=[]):
    N, M = X_sol.shape
    M_plus1 = M + 1
    K = len(weights_list)
    
    simulated_rewards = np.zeros(K)
    z = np.zeros((N,K))
    
    for k, weights in enumerate(weights_list):
        rewards_per_perm = []
        perm = permutations[k]
        remaining_capacity = capacities.copy()
        remaining_capacity.append(N)  # exit option, effectively infinite
        reward = 0.0
        
        for idx,i in enumerate(perm):
            # Determine available options (offered & capacity > 0)
            available = np.zeros(M_plus1)
            for j in range(M):
                if X_sol[i,j] == 1 and remaining_capacity[j] > 0:
                    available[j] = 1
            available[M] = 1  # exit always available
            

            # Choose the option with max weight
            choices = np.where(available)[0]
            selected = choices[np.argmax(weights[i, choices])]
            # if k == 1 and idx<=2:
            #     print("Avaialble",available)
            #     print("Choice {}".format(selected))

            reward += weights[i, selected]
            z[i,k] =weights[i, selected] 
            
            # Update capacity if not exit
            if selected < M:
                remaining_capacity[selected] -= 1
        
        rewards_per_perm.append(reward)
        
        simulated_rewards[k] = np.mean(rewards_per_perm)
    # print(z[permutations[1],1])
    return simulated_rewards

def create_random_weights(weights,epsilon):
    noisy = weights + np.random.uniform(-epsilon, epsilon, weights.shape)
    noisy = np.clip(noisy, 0, 1)
    margin = 1e-6   # much larger than bonus
    noisy = margin + (1 - 2*margin) * noisy
    bonus_eps = 1e-12
    J = weights.shape[1]
    bonus = bonus_eps * np.arange(J)[None, :]
    noisy = noisy + bonus
    return noisy 

def full_milp_policy(parameters,fairness_threshold=None,cluster_by_patient=None):
    """
    Find optimal assortment policy by optimizing over multiple random orderings.
    
    Arguments:
        weights: N x M matrix where weights[i,j] is reward for person i choosing option j
                 Last column (M-1) is the outside option with infinite capacity
        max_per_provider: Capacity for each of the first M-1 options
        threshold: Threshold parameter (unused for now)
        ordering: Type of ordering ("uniform" for random permutations)
        num_permutations: Number of random orderings to consider
    
    Returns:
        assortment: N x (M-1) binary matrix indicating which options to show each person
        objective_value: Expected reward across all orderings
    """
    weights = parameters['weights']
    capacities = parameters['capacities']
    max_shown = parameters['max_shown']
    epsilon = parameters.get('noise', 0.0)

    N, M_plus1 = weights.shape
    M = M_plus1 - 1

    # For simplicity: single scenario
    K = 10  
    weights_list = []
    orderings = []
    for i in range(K):
        np.random.seed(i)
        weights_list.append(create_random_weights(weights,epsilon))
        orderings.append(np.random.permutation(N))
    
    model = gp.Model("assortment_sequential")
    model.Params.LogToConsole = 0

    # --- Variables ---
    X = model.addVars(N, M, vtype=GRB.BINARY, name="X")            # shared assortment
    y = model.addVars(N, M_plus1, K, vtype=GRB.BINARY, name="y")   # selection per scenario
    z = model.addVars(N, K, vtype=GRB.CONTINUOUS, name="z")        # utility per scenario
    c = model.addVars(N, M, K, vtype=GRB.CONTINUOUS, name="c")     # remaining capacity per timestep

    # --- Constraints ---
    for k in range(K):
        weights_k = weights_list[k]
        ordering = orderings[k]

        # 1. One selection per patient
        for i in range(N):
            model.addConstr(gp.quicksum(y[i,j,k] for j in range(M_plus1)) == 1)

        # 2. Max_shown per patient
        for i in range(N):
            model.addConstr(gp.quicksum(X[i,j] for j in range(M)) <= max_shown)

        # 3. Initialize capacities at timestep 0
        for j in range(M):
            model.addConstr(c[0,j,k] == capacities[j])

        # 5. Cannot pick unavailable providers (offered + available capacity)
        for t in range(N):
            patient = ordering[t]
            for j in range(M):
                if t == 0:
                    model.addConstr(c[t,j,k] == capacities[j])
                else:
                    prev_patient = ordering[t-1]
                    model.addConstr(c[t,j,k] == c[t-1,j,k] - y[prev_patient,j,k])

                # Availability constraint
                model.addConstr(y[patient,j,k] <= X[patient,j])
                model.addConstr(y[patient,j,k] <= c[t,j,k])

        # 6. Link utility
        for i in range(N):
            model.addConstr(z[i,k] == gp.quicksum(weights_k[i,j]*y[i,j,k] for j in range(M_plus1)))

        # 7. Rationality constraints (best among available)
        bigM = 1
        for t in range(N):
            patient = ordering[t]
            for j in range(M_plus1):
                for l in range(M_plus1):
                    if j == l:
                        continue
                    if l < M:
                        # only enforce if l is offered
                        model.addConstr(
                                                        z[patient,k] >= weights_k[patient,l]
                                                                    - bigM*(1 - y[patient,j,k])
                                                                    - bigM*(1 - X[patient,l])
                                                                    - bigM*(1 - c[t,l,k])   # effectively disables if capacity = 0
                                                    )                    
                    else:
                        # exit option always available
                        model.addConstr(z[patient,k] >= weights_k[patient,l] - bigM*(1 - y[patient,j,k]))

    # --- Objective ---
    model.setObjective(gp.quicksum(z[i,k] for i in range(N) for k in range(K)), GRB.MAXIMIZE)

    # --- Solve ---
    model.optimize()

    # --- Extract solution ---
    X_sol = np.zeros((N,M), dtype=int)
    z_sol = np.zeros((N,K))
    y_sol = np.zeros((N,M,K))
    c_sol = np.zeros((N,M,K))
    objective_value = None
    if model.status == GRB.OPTIMAL:
        for i in range(N):
            for j in range(M):
                X_sol[i,j] = int(X[i,j].X)
                for k in range(K):
                    y_sol[i,j,k] = float(y[i,j,k].X)
                    c_sol[i,j,k] = float(c[i,j,k].X)
            for j in range(K):
                z_sol[i,j] = float(z[i,j].X)
        objective_value = model.ObjVal
    return X_sol

def full_lp_policy(parameters,fairness_threshold=None,cluster_by_patient=None):
    """
    Find optimal assortment policy by optimizing over multiple random orderings.
    
    Arguments:
        weights: N x M matrix where weights[i,j] is reward for person i choosing option j
                 Last column (M-1) is the outside option with infinite capacity
        max_per_provider: Capacity for each of the first M-1 options
        threshold: Threshold parameter (unused for now)
        ordering: Type of ordering ("uniform" for random permutations)
        num_permutations: Number of random orderings to consider
    
    Returns:
        assortment: N x (M-1) binary matrix indicating which options to show each person
        objective_value: Expected reward across all orderings
    """
    weights = parameters['weights']
    capacities = parameters['capacities']
    max_shown = parameters['max_shown']
    epsilon = parameters.get('noise', 0.0)

    N, M_plus1 = weights.shape
    M = M_plus1 - 1

    # For simplicity: single scenario
    K = 10  
    weights_list = []
    orderings = []
    for i in range(K):
        np.random.seed(i)
        weights_list.append(create_random_weights(weights,epsilon))
        orderings.append(np.random.permutation(N))
    
    model = gp.Model("assortment_sequential")
    model.Params.LogToConsole = 0

    # --- Variables ---
    X = model.addVars(N, M, vtype=GRB.CONTINUOUS, name="X")            # shared assortment
    y = model.addVars(N, M_plus1, K, vtype=GRB.CONTINUOUS, name="y")   # selection per scenario
    z = model.addVars(N, K, vtype=GRB.CONTINUOUS, name="z")        # utility per scenario
    c = model.addVars(N, M, K, vtype=GRB.CONTINUOUS, name="c")     # remaining capacity per timestep

    # --- Constraints ---
    for k in range(K):
        weights_k = weights_list[k]
        ordering = orderings[k]

        # 1. One selection per patient
        for i in range(N):
            model.addConstr(gp.quicksum(y[i,j,k] for j in range(M_plus1)) == 1)

        # 2. Max_shown per patient
        for i in range(N):
            model.addConstr(gp.quicksum(X[i,j] for j in range(M)) <= max_shown)

        # 3. Initialize capacities at timestep 0
        for j in range(M):
            model.addConstr(c[0,j,k] == capacities[j])

        # 5. Cannot pick unavailable providers (offered + available capacity)
        for t in range(N):
            patient = ordering[t]
            for j in range(M):
                if t == 0:
                    model.addConstr(c[t,j,k] == capacities[j])
                else:
                    prev_patient = ordering[t-1]
                    model.addConstr(c[t,j,k] == c[t-1,j,k] - y[prev_patient,j,k])

                # Availability constraint
                model.addConstr(y[patient,j,k] <= X[patient,j])
                model.addConstr(y[patient,j,k] <= c[t,j,k])

        # 6. Link utility
        for i in range(N):
            model.addConstr(z[i,k] == gp.quicksum(weights_k[i,j]*y[i,j,k] for j in range(M_plus1)))

        # 7. Rationality constraints (best among available)
        bigM = 1
        for t in range(N):
            patient = ordering[t]
            for j in range(M_plus1):
                for l in range(M_plus1):
                    if j == l:
                        continue
                    if l < M:
                        # only enforce if l is offered
                        model.addConstr(
                                                        z[patient,k] >= weights_k[patient,l]
                                                                    - bigM*(1 - y[patient,j,k])
                                                                    - bigM*(1 - X[patient,l])
                                                                    - bigM*(1 - c[t,l,k])   # effectively disables if capacity = 0
                                                    )                    
                    else:
                        # exit option always available
                        model.addConstr(z[patient,k] >= weights_k[patient,l] - bigM*(1 - y[patient,j,k]))

    # --- Objective ---
    model.setObjective(gp.quicksum(z[i,k] for i in range(N) for k in range(K)), GRB.MAXIMIZE)

    # --- Solve ---
    model.optimize()

    # --- Extract solution ---
    X_sol = np.zeros((N,M), dtype=int)
    z_sol = np.zeros((N,K))
    y_sol = np.zeros((N,M,K))
    c_sol = np.zeros((N,M,K))
    objective_value = None
    if model.status == GRB.OPTIMAL:
        for i in range(N):
            for j in range(M):
                X_sol[i,j] = int(X[i,j].X)
                for k in range(K):
                    y_sol[i,j,k] = float(y[i,j,k].X)
                    c_sol[i,j,k] = float(c[i,j,k].X)
            for j in range(K):
                z_sol[i,j] = float(z[i,j].X)
        objective_value = model.ObjVal
    print(objective_value/(K*N)) 
    res = simulate_assortment(X_sol, weights_list, capacities=[1]*M, permutations=orderings)
    print(np.mean(res)/20)    
    
    return X_sol



def greedy_justified(parameters,K=10):
    """
    Refined greedy algorithm for sequential assortment.
    Improves on _greedy_hybrid by:
    - Sequential simulation per scenario
    - Marginal utility scaled by remaining capacity
    - Soft selection using log-odds advantage over outside option
    - Weighted accumulation over scenarios
    """
    weights = parameters['weights']
    capacities = parameters['capacities'].copy()
    max_shown = parameters['max_shown']
    epsilon = parameters['noise']
    
    N, M_plus1 = weights.shape
    M = M_plus1 - 1
    
    N, M_plus1 = weights.shape
    M = M_plus1 - 1
    
    # Scenario generation
    scenarios = [(create_random_weights(weights, epsilon), np.random.permutation(N)) for _ in range(K)]
    
    # Accumulate marginal gains
    marginal_matrix = np.zeros((N, M))
    
    for weights_k, ordering in scenarios:
        remaining_capacity = capacities.copy()
        for t in range(N):
            patient = ordering[t]
            # Compute expected marginal gain per provider
            gains = np.full(M, -np.inf)
            for j in range(M):
                if remaining_capacity[j] <= 0:
                    continue
                # Marginal gain: advantage over outside option
                gains[j] = weights_k[patient, j] - weights_k[patient, M]
            # Pick top max_shown by marginal gain
            top_idx = np.argsort(-gains)[:max_shown]
            for j in top_idx:
                if gains[j] > -np.inf:
                    marginal_matrix[patient, j] += gains[j]
            # Decrement capacity for tentative top choice
            if np.any(gains > -np.inf):
                chosen = top_idx[0]
                remaining_capacity[chosen] -= 1
    
    # Final assignment: pick top max_shown per patient
    X_sol = np.zeros((N, M), dtype=int)
    for i in range(N):
        top_idx = np.argsort(-marginal_matrix[i])[:max_shown]
        for j in top_idx:
            if marginal_matrix[i, j] > 0:
                X_sol[i, j] = 1
    return X_sol
def dual_adjusted_menu_fixed_point(
    parameters,
    S=50,             # number of scenarios
    T=5,              # outer iterations
    seed=None
):
    """
    Dual-adjusted menu construction via fixed-point iteration.

    Implements:
      - menu-induced consumption z_{i,j}
      - exposure-weighted marginal scores m_{i,j}
      - shadow prices lambda_j via marginal system value
      - iterative menu updates

    Returns:
      X : binary menu matrix, shape (N, M)
    """
    weights = parameters['weights']
    capacities = parameters['capacities'].copy()
    K = parameters['max_shown']


    rng = np.random.default_rng(seed)
    N, MPlus_One = weights.shape
    M = MPlus_One-1

    # ---------- helper: simulate system value under fixed menus ----------
    def simulate_value(X, caps, ordering):
        remaining = caps.copy()
        total = 0.0

        for i in ordering:
            best_gain = 0.0
            best_j = None
            for j in range(M):
                if X[i, j] == 0:
                    continue
                if remaining[j] <= 0:
                    continue
                gain = weights[i, j]
                if gain > best_gain:
                    best_gain = gain
                    best_j = j

            if best_j is not None:
                remaining[best_j] -= 1
                total += best_gain

        return total

    # ---------- initialization: naive menus ----------
    X = np.zeros((N, M), dtype=int)
    for i in range(N):
        idx = np.argsort(-weights[i])[:K]
        for j in idx:
            if weights[i, j] > 0:
                X[i, j] = 1

    # ---------- fixed-point iteration ----------
    for it in range(T):

        # ---- Step 1: simulate exposure m and consumption z ----
        m = np.zeros((N, M))
        z = np.zeros((N, M))

        scenarios = [rng.permutation(N) for _ in range(S)]

        for ordering in scenarios:
            remaining = capacities.copy()

            for i in ordering:
                # available menu items
                candidates = [
                    j for j in range(M)
                    if X[i, j] == 1 and remaining[j] > 0 and weights[i, j] > 0
                ]
                if not candidates:
                    continue

                # exposure contribution: top-K available
                top = sorted(
                    candidates,
                    key=lambda j: weights[i, j],
                    reverse=True
                )[:K]
                for j in top:
                    m[i, j] += weights[i, j]

                # actual consumption
                j_star = max(candidates, key=lambda j: weights[i, j])
                remaining[j_star] -= 1
                z[i, j_star] += 1

        m /= S
        z /= S

        # ---- Step 2: estimate shadow prices lambda_j ----
        lambda_j = np.zeros(M)

        for j in range(M):
            delta = 0.0
            for ordering in scenarios:
                V_base = simulate_value(X, capacities, ordering)
                caps_plus = capacities.copy()
                caps_plus[j] += 1
                V_plus = simulate_value(X, caps_plus, ordering)
                delta += (V_plus - V_base)
            lambda_j[j] = delta / S

        # ---- Step 3: capacity-adjusted scores ----
        m_net = m - z * lambda_j[None, :]

        # ---- Step 4: update menus ----
        X_new = np.zeros_like(X)
        for i in range(N):
            idx = np.argsort(-m_net[i])[:K]
            for j in idx:
                if m_net[i, j] > 0:
                    X_new[i, j] = 1

        # ---- convergence check ----
        if np.array_equal(X_new, X):
            break
        X = X_new

    return X


def optimal_dual_assignment_with_exit_simple(parameters, K=10):
    """
    Exact dual-adjusted assignment using Gurobi.
    
    capacities: array of length M+1 (last entry is outside option capacity)
    weights: N x (M+1)
    """
    weights = parameters['weights']
    capacities = parameters['capacities'].copy()
    max_shown = parameters['max_shown']
    epsilon = parameters['noise']
    
    N, M_plus1 = weights.shape  # include outside option
    M = M_plus1-1
    # Accumulate duals across scenarios
    duals_accum = np.zeros(M_plus1)
    scenarios = [(create_random_weights(weights, epsilon), np.random.permutation(N)) for _ in range(K)]
    z = np.zeros((N,M))

    for weights_k, ordering in scenarios:
        remaining_capacity = capacities.copy()
        for t in range(N):
            patient = ordering[t]
            # Compute expected marginal gain per provider
            gains = np.full(M, -np.inf)
            for j in range(M):
                if remaining_capacity[j] <= 0:
                    continue
                # Marginal gain: advantage over outside option
                gains[j] = weights_k[patient, j] - weights_k[patient, M]
            # Pick top max_shown by marginal gain

            if gains.max() == -np.inf:
                continue
            top_idx = np.argmax(gains)
            if top_idx < M:
                remaining_capacity[top_idx] -= 1
                # z[patient, top_idx] += 1/len(scenarios)


        # Build MILP model
        model = gp.Model()
        model.Params.OutputFlag = 0  # suppress solver output
        
        # Variables: x[i,j] binary
        x = model.addVars(N, M_plus1, vtype=GRB.CONTINUOUS,lb=0,ub=1)
        
        # Objective: maximize total weight
        model.setObjective(
            gp.quicksum(weights_k[i,j]*x[i,j] for i in range(N) for j in range(M_plus1)),
            GRB.MAXIMIZE
        )
        
        # Constraints
        # 1. Each patient assigned to exactly one option
        for i in range(N):
            model.addConstr(gp.quicksum(x[i,j] for j in range(M_plus1)) == 1)
        
        # 2. Provider capacities (including outside option)
        capacity_constraints = []
        for j in range(M_plus1):
            constr = model.addConstr(gp.quicksum(x[i,j] for i in range(N)) <= capacities[j])
            capacity_constraints.append(constr)
        
        # Solve MILP
        model.optimize()

        solution = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                solution[i,j] = x[i, j].X
                z[i, j] += x[i, j].X/len(scenarios)

                
        # Extract duals for capacity constraints
        # Note: for MILP, Gurobi reports duals of the LP relaxation
        lambdas = np.array([constr.Pi for constr in capacity_constraints])
        duals_accum += lambdas/len(scenarios)

    # Adjust original weights by averaged duals
    adjusted_weights = np.zeros(weights[:,:-1].shape)

    print([i for i in range(len(duals_accum)) if duals_accum[i] == 0])
    for i in range(len(adjusted_weights)):
        for j in range(len(adjusted_weights[i])):
            factor = (duals_accum[j]-duals_accum[-1])
            average_delta = np.mean([max(w[0][i,j]-w[0][i,-1],0) for w in scenarios])
            adjusted_weights[i,j] = (average_delta - factor)
    # Final assignment: pick top max_shown per patient
    X_sol = np.zeros((N, M), dtype=int)
    for i in range(N):
        top_idx = np.argsort(-adjusted_weights[i])[:max_shown]
        for j in top_idx:
            if adjusted_weights[i, j] > 0:
                X_sol[i, j] = 1
    return X_sol


def optimal_dual_assignment_with_exit(parameters, K=10):
    """
    Exact dual-adjusted assignment using Gurobi.
    
    capacities: array of length M+1 (last entry is outside option capacity)
    weights: N x (M+1)
    """
    weights = parameters['weights']
    capacities = parameters['capacities'].copy()
    max_shown = parameters['max_shown']
    epsilon = parameters['noise']
    
    N, M_plus1 = weights.shape  # include outside option
    M = M_plus1-1
    # Accumulate duals across scenarios
    duals_accum = np.zeros(M_plus1)
    scenarios = [(create_random_weights(weights, epsilon), np.random.permutation(N)) for _ in range(K)]
    z = np.zeros((N,M))

    for weights_k, ordering in scenarios:
        # Build MILP model
        model = gp.Model()
        model.Params.OutputFlag = 0  # suppress solver output
        
        # Variables: x[i,j] binary
        x = model.addVars(N, M_plus1, vtype=GRB.CONTINUOUS,lb=0,ub=1)
        
        # Objective: maximize total weight
        model.setObjective(
            gp.quicksum(weights_k[i,j]*x[i,j] for i in range(N) for j in range(M_plus1)),
            GRB.MAXIMIZE
        )
        
        # Constraints
        # 1. Each patient assigned to exactly one option
        for i in range(N):
            model.addConstr(gp.quicksum(x[i,j] for j in range(M_plus1)) == 1)
        
        # 2. Provider capacities (including outside option)
        capacity_constraints = []
        for j in range(M_plus1):
            constr = model.addConstr(gp.quicksum(x[i,j] for i in range(N)) <= capacities[j])
            capacity_constraints.append(constr)
        
        # Solve MILP
        model.optimize()

        solution = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                solution[i,j] = x[i, j].X
                z[i, j] += x[i, j].X/len(scenarios)

                
        # Extract duals for capacity constraints
        # Note: for MILP, Gurobi reports duals of the LP relaxation
        lambdas = np.array([constr.Pi for constr in capacity_constraints])
        duals_accum += lambdas/len(scenarios)

    # Adjust original weights by averaged duals
    adjusted_weights = np.zeros(weights[:,:-1].shape)

    print([i for i in range(len(duals_accum)) if duals_accum[i] == 0])
    for i in range(len(adjusted_weights)):
        for j in range(len(adjusted_weights[i])):
            factor = (duals_accum[j]-duals_accum[-1])*z[i,j]
            average_delta = np.mean([max(w[0][i,j]-w[0][i,-1],0) for w in scenarios])
            adjusted_weights[i,j] = (average_delta - factor)
    # Final assignment: pick top max_shown per patient
    X_sol = np.zeros((N, M), dtype=int)
    for i in range(N):
        top_idx = np.argsort(-adjusted_weights[i])[:max_shown]
        for j in top_idx:
            if adjusted_weights[i, j] > 0:
                X_sol[i, j] = 1
    return X_sol

def greedy_justified_fair(parameters, eta, K=10, seed=0):
    """
    Greedy algorithm with fairness across clusters.
    
    Fairness rule:
        • Only the worst-off cluster receives a boost.
        • Other clusters get priority = 1.
        • Priorities affect permutation order.
    
    clusters: array of length N with cluster indices
    """
    np.random.seed(seed)

    # ------ Parameters ------
    weights = parameters['weights']
    capacities_init = parameters['capacities']
    max_shown = parameters['max_shown']
    epsilon = parameters['noise']

    N, M_plus1 = weights.shape
    M = M_plus1 - 1

    patient_data = json.load(open("../../data/patient_data_{}_{}_{}_comorbidity.json".format(seed,N,M)))
    zipcode_clusters = json.load(open("../../data/ct_zipcode_cluster.json"))
    clusters = list(zipcode_clusters.values())
    for i in patient_data:
        if i['location'] not in zipcode_clusters:
            zipcode_clusters[i['location']] = random.choice(clusters)
    clusters = np.array([zipcode_clusters[i['location']] for i in patient_data])



    clusters = np.asarray(clusters)
    n_clusters = clusters.max() + 1
    cluster_sizes = np.bincount(clusters, minlength=n_clusters)

    # Track moving average of cluster utilities
    cluster_utils = np.array([1,1,0,1,1])

    # Accumulate marginal values across scenarios
    marginal_matrix = np.zeros((N, M))

    # ------------------------------------------------
    #           MAIN LOOP ACROSS SCENARIOS
    # ------------------------------------------------
    for it in range(K):
        u_min = cluster_utils.min()
        d = cluster_utils - u_min         # distance above worst-off cluster
        p_cluster = np.exp(-eta * d)      # guarantees worst cluster = 1
        p_cluster /= p_cluster.max()      # normalize to [0,1]

        # Average priority by cluster for debugging

        priority = p_cluster[clusters]    # each patient gets cluster-based priority
        keys = -np.log(np.random.rand(N)) / priority
        ordering = np.argsort(keys)

        # Add noise for robustness
        weights_k = create_random_weights(weights, epsilon)

        # Reset capacity for this scenario
        remaining_capacity = capacities_init.copy()

        # Track per-cluster utility for this scenario
        cluster_util_scenario = np.zeros(n_clusters)

        # ----- 3. Sequential greedy with capacity -----
        for t in range(N):
            i = ordering[t]
            c = clusters[i]

            gains = np.full(M, -np.inf)
            for j in range(M):
                if remaining_capacity[j] > 0:
                    gains[j] = weights_k[i, j] - weights_k[i, M]


            # probabilistic marginal assignment
            if np.random.rand() < p_cluster[c]:
                # Record marginal gains
                top_idx = np.argsort(-gains)[:max_shown]
                for j in top_idx:
                    if gains[j] > -np.inf:
                        marginal_matrix[i, j] += gains[j]

                # Choose best available option
                if np.any(gains > -np.inf):
                    chosen = top_idx[0]
                    remaining_capacity[chosen] -= 1
                    cluster_util_scenario[c] += weights_k[i, chosen]

        # ----- 4. Update cluster utilities (average per cluster) -----
        avg_cluster_util = cluster_util_scenario / cluster_sizes
        cluster_utils = (cluster_utils * it + avg_cluster_util) / (it + 1)

    # ------------------------------------------------
    #               FINAL SELECTION
    # ------------------------------------------------
    X_sol = np.zeros((N, M), dtype=int)
    for i in range(N):
        top_idx = np.argsort(-marginal_matrix[i])[:max_shown]
        for j in top_idx:
            if marginal_matrix[i, j] > 0:
                X_sol[i, j] = 1

    return X_sol
