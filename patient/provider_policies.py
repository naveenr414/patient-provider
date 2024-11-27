import numpy as np
import gurobipy as gp
from gurobipy import GRB

def provider_focused_policy(simulator,min_matchings_per=[],max_matchings_per=[],lamb=0,use_log=False):
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

        if not use_log:
            m.setObjective(gp.quicksum((weights[i, j]-lamb*p) * x[i, j] for i in range(N) for j in range(M)), GRB.MAXIMIZE)
        else:
            breakpoints = [0,1, 8,32]  # Example range
            values = [0]+[np.log(x) for x in breakpoints[1:]]  # Precompute log values

            sum_x = m.addVars(N, lb=0, name="sum_x")  # Sum of x_{i,j}
            z = m.addVars(N, name="z")  # Log values

            for i in range(N):
                m.addConstr(sum_x[i] == gp.quicksum(x[i, j] for j in range(M)))
            for i in range(N):
                m.addGenConstrPWL(sum_x[i], z[i], breakpoints, values, "PWL_{}".format(i))

            m.setObjective(gp.quicksum(weights[i, j] * x[i, j] for i in range(N) for j in range(M))-gp.quicksum(lamb*p*z[i] for i in range(N)), GRB.MAXIMIZE)


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

    values = []
    solutions = []
    for b in range(1,N+1):
        value, sol = get_solution(b)
        if len(values) > 0 and value < values[-1]:
            break 
        values.append(value)
        solutions.append(sol)

    max_b = np.argmax(values)

    sol = solutions[max_b]
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

def provider_focused_linear_regularization_policy(lamb):
    """Policy that optimizes menus for each provider, while
    adding in a penalty for the total number of things offered
    This can help reduce provider-side interference
    in a linear manner
    
    Arguments:
        lamb: Lamb value to restrict the intereference between menus
    
    Returns: List of providers on the menu"""

    def policy(simulator):
        return provider_focused_policy(simulator,min_matchings_per=[],max_matchings_per=[],lamb=lamb)
    return policy 

def provider_focused_log_regularization_policy(lamb):
    """Policy that optimizes menus for each provider, while
    adding in a penalty for the total number of things offered
    This can help reduce provider-side interference
    in a linear manner
    
    Arguments:
        lamb: Lamb value to restrict the intereference between menus
    
    Returns: List of providers on the menu"""

    def policy(simulator):
        return provider_focused_policy(simulator,min_matchings_per=[],max_matchings_per=[],lamb=lamb,use_log=True)
    return policy 
