import random 
import numpy as np 
import gurobipy as gp
from gurobipy import GRB


def random_policy(patient,provider_capacities,provider_max_capacities):
    """Randomly give a menu of available providers
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, which providers to show them """

    available_providers = [i for i in range(len(provider_capacities)) if random.random() < 0.5]
    return available_providers

def greedy_policy(patient,provider_capacities,provider_max_capacities):
    """A policy which shows all providers
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, which providers to show them """

    return list(range(len(provider_capacities)))

def max_patient_utility(patient,provider_capacities,provider_max_capacities):
    """Leverage the Linear Program from the Davis paper to greedily optimize
        Combining this with a discount function 
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, which providers to show them """

    utilities = patient.provider_rewards
    utilities = [utilities[i]*int(provider_capacities[i]>0) for i in range(len(provider_capacities))]
    def discount_function(x,max_capacity):
        return np.exp(-(1-x/max_capacity))      

    utilities = [utilities[i]*(discount_function(provider_capacities[i],provider_max_capacities[i])) for i in range(len(provider_max_capacities))]  

    exit_option = patient.exit_option
    N = len(provider_capacities)

    model = gp.Model("LP")
    model.setParam('OutputFlag', 0)
    w = model.addVars(N+1, name="w")
    objective_expr = gp.LinExpr()
    for j in range(1, N+1):
        objective_expr += utilities[j-1] * w[j]
    model.setObjective(objective_expr, GRB.MAXIMIZE)

    sum_expr = gp.LinExpr()
    for j in range(1, N+1):
        sum_expr += w[j]
    model.addConstr(sum_expr + w[0] == 1)

    for j in range(1, N+1):
        if utilities[j-1]>0:
            model.addConstr(w[j] / utilities[j-1] <= w[0]/exit_option)
        else:
            model.addConstr(w[j]  <= 0)
    model.optimize()

    w_vals = [round((w[j].x*exit_option)/(utilities[j-1]*w[0].x)) if utilities[j-1]*w[0].x > 0 else 0 for j in range(1,N+1)]

    menu = [i for i in range(len(w_vals)) if w_vals[i] == 1]

    return menu 

def max_match_prob(patient,provider_capacities,provider_max_capacities):
    """Leverage the Linear Program from the Davis paper to greedily optimize
        Maximize for the number of matches
        Combining this with a discount function 
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, which providers to show them """

    utilities = patient.provider_rewards
    utilities = [utilities[i]*int(provider_capacities[i]>0) for i in range(len(provider_capacities))]
    def discount_function(x,max_capacity):
        return np.exp(-(1-x/max_capacity))      

    utilities = [utilities[i]*(discount_function(provider_capacities[i],provider_max_capacities[i])) for i in range(len(provider_max_capacities))]  

    exit_option = patient.exit_option
    N = len(provider_capacities)

    model = gp.Model("LP")
    model.setParam('OutputFlag', 0)
    w = model.addVars(N+1, name="w")
    objective_expr = gp.LinExpr()
    for j in range(1, N+1):
        objective_expr += w[j]
    model.setObjective(objective_expr, GRB.MAXIMIZE)

    sum_expr = gp.LinExpr()
    for j in range(1, N+1):
        sum_expr += w[j]
    model.addConstr(sum_expr + w[0] == 1)

    for j in range(1, N+1):
        if utilities[j-1]>0:
            model.addConstr(w[j] / utilities[j-1] <= w[0]/exit_option)
        else:
            model.addConstr(w[j]  <= 0)
    model.optimize()

    w_vals = [round((w[j].x*exit_option)/(utilities[j-1]*w[0].x)) if utilities[j-1]*w[0].x > 0 else 0 for j in range(1,N+1)]

    menu = [i for i in range(len(w_vals)) if w_vals[i] == 1]
    optimal_value = model.objVal
    
    return menu 