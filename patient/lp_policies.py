import numpy as np 
from patient.utils import solve_linear_program, solve_linear_program_dynamic
import gurobipy as gp
from gurobipy import GRB
import itertools 

def lp_policy(weights,max_per_provider):
    """Policy which selects according to the LP, in an offline fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
    
    Returns: List of providers on the menu"""
    N,M = weights.shape 
    M -= 1 
    lp_solution = solve_linear_program(weights,max_per_provider)
    assortment = np.zeros((N,M))
    
    for (i,j) in lp_solution:
        assortment[i,j] = 1
    
    return np.array(assortment)

def get_gradient_policy(threshold,ordering):
    def policy(w,m):
        return gradient_policy(w,m,threshold=threshold,ordering=ordering)
    
    return policy 

def gradient_policy(weights, max_per_provider, threshold=0.1, ordering="uniform", num_permutations=10):
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
    N, M = weights.shape
    M_real = M - 1  # Number of capacity-constrained options
    
    # Generate random permutations
    orderings = []
    for _ in range(num_permutations):
        orderings.append(np.random.permutation(N))
    
    # Create Gurobi model
    m = gp.Model("assortment_optimization")
    m.setParam('OutputFlag', 0)
    
    # Decision variables: assortment[i, j] = 1 if we show option j to person i
    assortment = m.addVars(N, M_real, vtype=GRB.BINARY, name="assortment")
    
    # For each ordering, we need to track:
    # - Which option each person chooses
    # - Remaining capacity after each person
    
    # Variables for each ordering
    total_reward = 0
    
    for perm_idx, perm in enumerate(orderings):
        choice = m.addVars(N, M, vtype=GRB.BINARY, name=f"choice_perm{perm_idx}")
        available = m.addVars(N, M_real, vtype=GRB.BINARY, name=f"available_perm{perm_idx}")        
        cumulative = m.addVars(N + 1, M_real, vtype=GRB.INTEGER, name=f"cumulative_perm{perm_idx}")
        for j in range(M_real):
            m.addConstr(cumulative[0, j] == 0, name=f"init_perm{perm_idx}_opt{j}")
        for pos in range(N):
            person_id = perm[pos]
            for j in range(M_real):
                m.addConstr(
                    cumulative[pos, j] <= max_per_provider - 1 + (1 - available[pos, j]) * N,
                    name=f"avail_lower_perm{perm_idx}_pos{pos}_opt{j}"
                )
                m.addConstr(
                    cumulative[pos, j] >= max_per_provider - available[pos, j] * N,
                    name=f"avail_upper_perm{perm_idx}_pos{pos}_opt{j}"
                )
            
            for j in range(M_real):
                m.addConstr(
                    choice[person_id, j] <= assortment[person_id, j],
                    name=f"assort_constraint_perm{perm_idx}_person{person_id}_opt{j}"
                )
                m.addConstr(
                    choice[person_id, j] <= available[pos, j],
                    name=f"avail_constraint_perm{perm_idx}_person{person_id}_opt{j}"
                )
            
            m.addConstr(
                gp.quicksum(choice[person_id, j] for j in range(M)) == 1,
                name=f"choose_one_perm{perm_idx}_person{person_id}"
            )
            
            for j in range(M_real):
                m.addConstr(
                    cumulative[pos + 1, j] == cumulative[pos, j] + choice[person_id, j],
                    name=f"update_cumulative_perm{perm_idx}_pos{pos}_opt{j}"
                )
        
        ordering_reward = gp.quicksum(
            weights[i, j] * choice[i, j] 
            for i in range(N) 
            for j in range(M)
        )
        total_reward += ordering_reward
    
    m.setObjective(total_reward / num_permutations, GRB.MAXIMIZE)
    m.optimize()
    
    assortment_solution = np.zeros((N, M_real))
    for i in range(N):
        for j in range(M_real):
            if assortment[i, j].X > 0.5:
                assortment_solution[i, j] = 1
        
    return assortment_solution

def simulate_ordering(assortment, weights, max_per_provider, ordering):
    """
    Simulate one ordering with given assortment.
    
    Returns total reward for this ordering.
    """
    N, M = weights.shape
    M_real = M - 1
    
    capacities = np.array([max_per_provider] * M_real + [N])  # Last option has infinite capacity
    total_reward = 0
    
    for person_id in ordering:
        # Find available options in assortment
        available_mask = np.concatenate([assortment[person_id], [1]])  # Outside option always available
        available_mask[:M_real] = available_mask[:M_real] * (capacities[:M_real] > 0)
        
        # Choose best available option
        masked_weights = weights[person_id] * available_mask
        masked_weights[available_mask == 0] = -np.inf
        chosen = np.argmax(masked_weights)
        
        # Update capacity and reward
        capacities[chosen] -= 1
        total_reward += weights[person_id, chosen]
    
    return total_reward


def gradient_policy_fast(weights, max_per_provider, threshold=0.1, ordering="uniform", 
                         num_permutations=10, learning_rate=0.1, num_iterations=10):
    """
    Fast gradient-based approximation using continuous relaxation.
    
    Uses a differentiable softmax relaxation and gradient ascent.
    """
    N, M = weights.shape
    M_real = M - 1
    
    # Initialize assortment probabilities (logits)
    assortment_logits = np.zeros((N, M_real))
    
    # Generate random permutations once
    orderings = [np.random.permutation(N) for _ in range(num_permutations)]
    
    best_assortment = None
    best_value = -np.inf
    
    for iteration in range(num_iterations):
        # Convert logits to probabilities using sigmoid
        assortment_probs = 1 / (1 + np.exp(-assortment_logits))
        
        # Compute gradient using finite differences (simple but effective)
        gradient = np.zeros((N, M_real))
        epsilon = 0.01
        
        for i in range(N):
            for j in range(M_real):
                # Current value
                current_assortment = (assortment_probs > 0.5).astype(int)
                current_value = np.mean([
                    simulate_ordering(current_assortment, weights, max_per_provider, perm)
                    for perm in orderings
                ])
                
                # Perturbed value
                assortment_logits[i, j] += epsilon
                perturbed_probs = 1 / (1 + np.exp(-assortment_logits))
                perturbed_assortment = (perturbed_probs > 0.5).astype(int)
                perturbed_value = np.mean([
                    simulate_ordering(perturbed_assortment, weights, max_per_provider, perm)
                    for perm in orderings
                ])
                assortment_logits[i, j] -= epsilon
                
                # Gradient
                gradient[i, j] = (perturbed_value - current_value) / epsilon
        
        # Update logits
        assortment_logits += learning_rate * gradient
        
        # Track best solution
        current_assortment = (assortment_probs > 0.5).astype(int)
        current_value = np.mean([
            simulate_ordering(current_assortment, weights, max_per_provider, perm)
            for perm in orderings
        ])
        
        if current_value > best_value:
            best_value = current_value
            best_assortment = current_assortment.copy()
    
    return best_assortment
