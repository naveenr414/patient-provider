import random 
import numpy as np 
import gurobipy as gp
from gurobipy import GRB

def guess_coefficients(matched_pairs,unmatched_pairs,preference_pairs,context_dim):
    value_pos = []
    value_neg = []

    if len(matched_pairs) == 0:
        return np.array([1 for i in range(context_dim)])

    for patient,provider in matched_pairs:
        context_diff = 1-np.abs(provider-patient)
        value_pos.append(context_diff)

    for patient,provider in unmatched_pairs:
        context_diff = 1-np.abs(provider-patient)
        value_neg.append(context_diff)

    value_pair_pos = []
    value_pair_neg = []

    for patient,provider_1,provider_2 in preference_pairs:
        # Provider_1 > Provider_2
        context_diff = 1-np.abs(provider_1-patient)
        value_pair_pos.append(context_diff)

        context_diff = 1-np.abs(provider_2-patient)
        value_pair_neg.append(context_diff)

    n = len(value_pos[0])

    # Create a new model
    m = gp.Model("lp")
    m.setParam('OutputFlag', 0)

    # Create variables
    y = m.addMVar(shape=n, name="y")

    # Set objective: maximize sum(y.dot(value_pos[i])) - sum(y.dot(value_neg[i]))
    obj = gp.quicksum(y[j]*(value_pos[i][j]-value_neg[i][j]) for i in range(len(value_pos)) for j in range(len(value_pos[i])))
    m.setObjective(obj, GRB.MAXIMIZE)

    # Add constraints: y.dot(value_pair_pos[i]) > y.dot(value_pair_neg[i]) for all i
    for i in range(len(value_pair_pos)):
        m.addConstr(gp.quicksum(y[j]*value_pair_pos[i][j] for j in range(len(value_pair_pos[i]))) >= gp.quicksum(y[j]*value_pair_neg[i][j] for j in range(len(value_pair_neg[i]))), name=f"c{i}")

    for i in range(n):
        m.addConstr(y[i] >= 0)
        m.addConstr(y[i] <= 1)

    # Optimize model
    m.optimize()

    y = np.array([float(y[i].X) for i in range(n)])
    return y