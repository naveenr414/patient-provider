import numpy as np
import gurobipy as gp
from gurobipy import GRB

def provider_focused_policy(simulator,min_matchings_per=[],max_matchings_per=[]):
    """Policy that optimizes menus for each provider, while
    ignoring inter-provider interference effects
    It does so by maximiznig (1-p)^{x} * \sum \theta/x
    
    Arguments:
        simulator: Simulator for patient-provider matching
    
    Returns: List of providers on the menu"""

    
    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(simulator.patients)
    M = weights.shape[1]

    if min_matchings_per == []:
        min_matchings_per = [0 for i in range(N)]
    
    if max_matchings_per == []:
        max_matchings_per = [min(simulator.max_menu_size,M) for i in range(N)]

    def get_solution(B):
        m = gp.Model("bipartite_matching")
        m.setParam('OutputFlag', 0)
        x = m.addVars(N, M, vtype=GRB.BINARY, name="x")

        m.setObjective(gp.quicksum(weights[i, j] * x[i, j] for i in range(N) for j in range(M)), GRB.MAXIMIZE)

        for j in range(M):
            m.addConstr(gp.quicksum(x[i,j] for i in range(N)) <= B)

        for i in range(N):
            m.addConstr(gp.quicksum(x[i,j] for j in range(M)) <= max_matchings_per[i], name=f"match_{j}")
            m.addConstr(gp.quicksum(x[i,j] for j in range(M)) >= min_matchings_per[i], name=f"match_{j}")

        m.optimize()
        if m.status == GRB.INFEASIBLE:
            return -1, np.zeros((N,M))
        obj_value = m.getObjective().getValue()
        real_value = (1-(1-p)**B)/B*obj_value

        solution = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                if x[i, j].X > 0.5:
                    solution[i,j] = 1
    
        return real_value, solution 

    values = [get_solution(b)[0] for b in range(1,N+1)]
    max_b = np.argmax(values)+1

    sol = get_solution(max_b)[1] 
    return sol

def provider_focused_less_interference_policy(simulator):
    """Policy that optimizes menus for each provider, while
    ignoring inter-provider interference effects
    It does so by maximiznig (1-p)^{x} * \sum \theta/x
    It accounts for inter-provider interference by 
    Restricting the total number of matches for each provider 
    with max matchings and min matchings
    
    Arguments:
        simulator: Simulator for patient-provider matching
    
    Returns: List of providers on the menu"""

    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(simulator.patients)
    M = weights.shape[1]

    max_matchings_per = [round(-1/(1+((1-p)/p)**3)*(min(simulator.max_menu_size,M)-1) + min(simulator.max_menu_size,M)) for i in range(N)]
    min_matchings_per = [i/2 for i in max_matchings_per]

    return provider_focused_policy(simulator,min_matchings_per=min_matchings_per,max_matchings_per=max_matchings_per)