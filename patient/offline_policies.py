import random 
import numpy as np 
import gurobipy as gp
from gurobipy import GRB

def solve_linear_program(weights,max_per_provider):
    N,P = weights.shape 

    m = gp.Model("bipartite_matching")
    m.setParam('OutputFlag', 0)
    x = m.addVars(N, P, vtype=GRB.BINARY, name="x")
    m.setObjective(gp.quicksum(weights[i, j] * x[i, j] for i in range(N) for j in range(P)), GRB.MAXIMIZE)
    for j in range(P):
        m.addConstr(gp.quicksum(x[i, j] for i in range(N)) <= max_per_provider, name=f"match_{j}_limit")

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
