import random 
import numpy as np 
import gurobipy as gp
from gurobipy import GRB

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

def p_approximation(simulator,patient,available_providers,memory,per_epoch_function):
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
    
    for i in range(len(simulator.patient_order)):
        if simulator.patient_order[i] == patient.idx:
            break 
        if simulator.patient_order[i] in simulator.unmatched_patients:
            default_menu[memory[simulator.patient_order[i]]] = 1

    return default_menu, memory 

def p_approximation_balance(simulator,patient,available_providers,memory,per_epoch_function):
    if memory == None:
        available_providers  = [1 if i > 0 else 0 for i in simulator.provider_capacities]
        weights = [p.provider_rewards for p in simulator.patients]
        weights = np.array(weights)
        weights = np.array([i*available_providers for i in weights])

        lamb = 0.1

        max_per_provider = simulator.provider_max_capacity
        LP_solution = solve_linear_program(weights,max_per_provider,lamb=lamb)

        matchings = [0 for i in range(weights.shape[0])]
        for (i,j) in LP_solution:
            matchings[i] = j
        memory = matchings 

    default_menu = [0 for i in range(len(available_providers))]
    default_menu[memory[patient.idx]] = 1
    
    for i in range(len(simulator.patient_order)):
        if simulator.patient_order[i] == patient.idx:
            break 
        if simulator.patient_order[i] in simulator.unmatched_patients:
            default_menu[memory[simulator.patient_order[i]]] = 1
    
    return default_menu, memory 
