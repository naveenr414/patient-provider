import random 
import numpy as np 
import gurobipy as gp
from gurobipy import GRB
from patient.learning import guess_coefficients

def solve_linear_program(weights,max_per_provider,lamb=0):
    N,P = weights.shape 

    m = gp.Model("bipartite_matching")
    m.setParam('OutputFlag', 0)
    x = m.addVars(N, P, vtype=GRB.BINARY, name="x")

    v = m.addVars(P, name="v")
    beta_bar = m.addVars(1,name="bar")


    m.setObjective(gp.quicksum(weights[i, j] * x[i, j] for i in range(N) for j in range(P)) - lamb/P*gp.quicksum(v[j] for j in range(P)), GRB.MAXIMIZE)
    m.addConstr(beta_bar[0] == 1/P * gp.quicksum(x[i, j] for i in range(N) for j in range(P)))

    for j in range(P):
        m.addConstr(gp.quicksum(x[i, j] for i in range(N)) <= max_per_provider, name=f"match_{j}_limit")
        m.addConstr(-v[j] <= gp.quicksum(x[i,j] for i in range(N))-beta_bar[0], name=f"match_{j}_limit2")
        m.addConstr(v[j] >= gp.quicksum(x[i,j] for i in range(N))-beta_bar[0], name=f"match_{j}_limit3")

    for i in range(N):
        m.addConstr(gp.quicksum(x[i, j] for j in range(P)) <= 1, name=f"match_{j}")

    m.optimize()

    # Extract the solution
    solution = []
    for i in range(N):
        for j in range(P):
            if x[i, j].X > 0.5:
                solution.append((i, j))
    return solution 

def offline_solution(simulator,patient,available_providers,memory,per_epoch_function):
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
    
    for order in simulator.patient_order:
        default_menu[memory[order]] = 1

        if order == patient.idx:
            break 
    
    return default_menu, memory 

def offline_learning_solution(simulator,patient,available_providers,memory,per_epoch_function):
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

    default_menu = [0 for i in range(len(available_providers))]
    
    for order in simulator.patient_order:
        default_menu[memory[order]] = 1

        if order == patient.idx:
            break 
    
    return default_menu, memory 


def offline_solution_balance(simulator,patient,available_providers,memory,per_epoch_function):
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

    default_menu = [0 for i in range(len(available_providers))]
    default_menu[memory[patient.idx]] = 1

    providers_left = np.zeros(len(available_providers))

    for i in range(patient.idx+1,len(simulator.patient_order)):
        providers_left[memory[simulator.patient_order[i]]] += 1
    
    for i in range(len(providers_left)):
        if providers_left[i] == 0:
            default_menu[i] = 1

    return default_menu, memory 
