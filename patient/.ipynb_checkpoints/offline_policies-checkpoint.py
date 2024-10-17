import random 
import numpy as np 
from patient.learning import guess_coefficients
from patient.utils import solve_linear_program
from patient.online_policies import compute_swap_scores, add_swap_matches
import itertools
import math 
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import differential_evolution



def offline_solution(simulator):
    """Policy which selects according to the LP, in an offline fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""
    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)

    max_per_provider = simulator.provider_max_capacity
    LP_solution = solve_linear_program(weights,max_per_provider)

    matchings = np.zeros((len(simulator.patients),simulator.num_providers))
    pairs = [-1 for i in range(len(simulator.patients))]
    unmatched_providers = set(list(range(simulator.num_providers)))

    for (i,j) in LP_solution:
        matchings[i,j] = 1
        pairs[i] = j

        if j in unmatched_providers:
            unmatched_providers.remove(j)
    return matchings  

def offline_solution_fairness(simulator):
    """Policy which selects according to the LP, in an offline fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""
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
    pairs = [-1 for i in range(len(simulator.patients))]
    unmatched_providers = set(list(range(simulator.num_providers)))

    for (i,j) in LP_solution:
        matchings[i,j] = 1
        pairs[i] = j

        if j in unmatched_providers:
            unmatched_providers.remove(j)
    return matchings  


def offline_solution_loose_constraints(simulator):
    """Policy which selects according to the LP, in an offline fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""

    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(simulator.patients)
    M = weights.shape[1]

    max_per_provider = simulator.provider_max_capacity/(p) * max(N/M,1)
    LP_solution = solve_linear_program(weights,max_per_provider)

    matchings = np.zeros((len(simulator.patients),simulator.num_providers))
    pairs = [-1 for i in range(len(simulator.patients))]
    unmatched_providers = set(list(range(simulator.num_providers)))

    for (i,j) in LP_solution:
        matchings[i,j] = 1
        pairs[i] = j

        if j in unmatched_providers:
            unmatched_providers.remove(j)
    
    return matchings  

def get_all_menus(N, M):
    binary_strings = itertools.product([0, 1], repeat=N * M)
    return [np.array(arr).reshape(N, M) for arr in binary_strings]


def one_shot_policy(simulator,patient,available_providers,memory,per_epoch_function):
    return per_epoch_function[patient.idx], memory 


def optimal_policy_epoch(simulator):
    """A policy which shows all providers
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, which providers to show them """

    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(simulator.patients)
    M = weights.shape[1]

    all_orders = list(itertools.permutations(list(range(N))))
    all_selections = list(itertools.product([0, 1], repeat=N))
    probabilities = [p**(sum(i))*(1-p)**(N-sum(i)) for i in all_selections]

    scores = []
    all_menus = get_all_menus(N,M)

    for menu in all_menus:
        total_reward = 0
        for ordering in all_orders:
            for s in range(len(all_selections)):
                selection = all_selections[s] 
                simulated_available_providers = np.ones(M)

                score = 0
                for i in range(N):
                    curr_patient = ordering[i]
                    available_providers = menu[curr_patient]*simulated_available_providers
                    
                    if np.sum(available_providers) == 0 or selection[i] == 0:
                        continue 
                    available_providers *= weights[curr_patient]
                    selected_provider = np.argmax(available_providers)
                    score += available_providers[selected_provider]
                    simulated_available_providers[selected_provider] = 0

                score *= probabilities[s]/(math.factorial(N))
                total_reward += score 
        scores.append(total_reward)
    scores = np.array(scores)
    scores/=N 
    best_menu = np.argmax(scores)
    menu = all_menus[best_menu]
    return menu 


def offline_solution_more_patients(simulator):
    """Policy which selects according to the LP, in an offline fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Unused 
    
    Returns: List of providers on the menu, along with the memory"""

    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(simulator.patients)
    M = weights.shape[1]

    max_per_provider = simulator.provider_max_capacity

    LP_solution = solve_linear_program(weights,max_per_provider)

    matchings = np.zeros((len(simulator.patients),simulator.num_providers))
    pairs = [-1 for i in range(len(simulator.patients))]
    unmatched_providers = set(list(range(simulator.num_providers)))
    unmatched_patients = set(list(range(len(matchings))))

    for (i,j) in LP_solution:
        matchings[i,j] = 1
        pairs[i] = j

        if j in unmatched_providers:
            unmatched_providers.remove(j)
        
        if i in unmatched_patients:
            unmatched_patients.remove(i)
    
    matches_by_provider = np.sum(matchings,axis=0)
    total = np.sum(weights*matchings,axis=0)

    p = simulator.choice_model_settings['top_choice_prob']
    for _ in range(len(matchings)):
        next_best_patient_score = [0 for i in range(simulator.num_providers)]
        next_best_patient = [-1 for i in range(simulator.num_providers)]

        for provider in range(simulator.num_providers):
            for _patient in unmatched_patients:
                score = weights[_patient][provider]
                if score > next_best_patient_score[provider] or next_best_patient[provider] == -1:
                    next_best_patient[provider] = _patient
                    next_best_patient_score[provider] = score

            benefit_new = (total[provider]+next_best_patient_score[provider])/(matches_by_provider[provider]+1) * (1-(1-p)**(matches_by_provider[provider]+1))
            benefit_old = (total[provider])/(matches_by_provider[provider]) * (1-(1-p)**(matches_by_provider[provider]))
            next_best_patient_score[provider] = benefit_new/benefit_old

        next_best_patient_score = zip(list(range(simulator.num_providers)),next_best_patient_score)
        next_best_patient_score = sorted(next_best_patient_score,key=lambda k: k[1],reverse=True)
        if max(next_best_patient) == -1 or next_best_patient_score[0][1] < 1 or len(unmatched_patients) == 0:
            break 
        
        for provider,score in next_best_patient_score:
            if score>1 and next_best_patient[provider] in unmatched_patients:
                matchings[next_best_patient[provider]][provider] = 1
                pairs[next_best_patient[provider]] = provider 
                unmatched_patients.remove(next_best_patient[provider])

    swap_score = compute_swap_scores(simulator,pairs,weights)
    matchings = add_swap_matches(swap_score,matchings,pairs,simulator.max_menu_size)        

    for i in range(len(matchings)):
        unmatched_provider_scores = [(j,weights[i][j]) for j in unmatched_providers]
        unmatched_provider_scores = sorted(unmatched_provider_scores,key=lambda k: k[1],reverse=True)
        unmatched_provider_scores = [j[0] for j in unmatched_provider_scores]

        if np.sum(matchings[i]) < simulator.max_menu_size:
            for j in unmatched_provider_scores[:int(simulator.max_menu_size-np.sum(matchings[i]))]:
                matchings[i][j] = 1
    
    return matchings  

def get_offline_solution_min_matchings(min_matchings_per):
    def f(simulator):
        return offline_solution_2_more_patients(simulator,min_matchings_per=min_matchings_per)
    return f

def offline_solution_2_more_patients(simulator,min_matchings_per=0):
    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(simulator.patients)
    M = weights.shape[1]

    orderings_by_provider = []

    for j in range(M):
        orderings_by_provider.append(np.argsort(weights[:,j])[::-1])

    curr_matchings = [[] for i in range(M)]
    curr_values = [[] for i in range(M)]
    idx = [0 for i in range(M)]
    curr_capacity = [0 for i in range(N)]

    max_matchings_per = min(round(1/p),simulator.max_menu_size)
    matchings = np.zeros((N,M))

    while np.min(curr_capacity) < max_matchings_per:
        round_values = [np.mean(curr_values[i])*(1-(1-p)**(len(curr_matchings[i]))) for i in range(M)]
        round_values = np.array(round_values)
        round_values[np.isnan(round_values)] = 0

        for i in range(M):
            while idx[i]<len(orderings_by_provider[i]) and curr_capacity[orderings_by_provider[i][idx[i]]] >= max_matchings_per:
                idx[i] += 1
        
        addition_value = np.zeros(M)
        for i in range(M):
            if idx[i]>=N or curr_capacity[orderings_by_provider[i][idx[i]]] >= max_matchings_per:
                continue 
            new_value = np.sum(curr_values[i])+weights[orderings_by_provider[i][idx[i]]][i]
            new_value /= len(curr_matchings[i])+1
            new_value *= (1-(1-p)**(len(curr_matchings[i])+1))
            addition_value[i] = new_value - round_values[i] 

        for i in np.argsort(addition_value)[::-1]:
            if idx[i] < N:
                if addition_value[i]>0 or curr_capacity[orderings_by_provider[i][idx[i]]] < min_matchings_per:
                    next_add = i 
                    break 
        else: 
            break 
            
        matchings[orderings_by_provider[next_add][idx[next_add]]][next_add] = 1
        curr_capacity[orderings_by_provider[next_add][idx[next_add]]] += 1
        curr_matchings[next_add].append(idx[next_add])
        curr_values[next_add].append(weights[orderings_by_provider[next_add][idx[next_add]]][next_add])
        idx[next_add] += 1
    return matchings 


def offline_solution_3_more_patients(simulator,min_matchings_per=0):
    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(simulator.patients)
    M = weights.shape[1]

    orderings_by_provider = []

    for j in range(M):
        orderings_by_provider.append(np.argsort(weights[:,j])[::-1])

    rewards = np.zeros((N,M))
    for j in range(M):
        for num_patients_taken in range(1,N+1):
            avg_reward = np.sum(weights[orderings_by_provider[j][:num_patients_taken]])/(num_patients_taken)
            rewards[num_patients_taken-1,j] = avg_reward*(1-(1-p)**(num_patients_taken))

    # Suppose we consider patient i, and take the first x patients for provider j
    contains_patient = np.zeros((N,N,M))
    for j in range(M):
        for num_patients_taken in range(1,N+1):
            for i in range(N):
                contains_patient[i,num_patients_taken-1,j] = int(i in orderings_by_provider[j][:num_patients_taken])

    max_matchings_per = M
    min_matchings_per = 0

    m = gp.Model("bipartite_matching")
    m.setParam('OutputFlag', 0)
    x = m.addVars(N, M, vtype=GRB.BINARY, name="x")

    m.setObjective(gp.quicksum(rewards[i, j] * x[i, j] for i in range(N) for j in range(M)), GRB.MAXIMIZE)

    for i in range(N):
        m.addConstr(gp.quicksum(contains_patient[i, x_bar,j]*x[x_bar,j] for j in range(M) for x_bar in range(N)) <= max_matchings_per, name=f"match_{j}")
        m.addConstr(gp.quicksum(contains_patient[i, x_bar,j]*x[x_bar,j] for j in range(M) for x_bar in range(N)) >= min_matchings_per, name=f"match_{j}")

    for j in range(M):
        m.addConstr(gp.quicksum(x[i,j] for i in range(N)) <= 1)

    m.optimize()

    # Extract the solution
    solution = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            if x[i, j].X > 0.5:
                solution[i,j] = 1
    print("Solution {}".format(solution))
    return solution 

# +
def offline_solution_4_more_patients(simulator,min_matchings_per=0):
    p = simulator.choice_model_settings['top_choice_prob']

    weights = [p.provider_rewards for p in simulator.patients]
    weights = np.array(weights)
    N = len(simulator.patients)
    M = weights.shape[1]

    max_matchings_per = [min(round(1/p**(1.85)),simulator.max_menu_size) for i in range(N)]
    
#     max_matchings_per = []
#     for i in range(N):
#         preferred_providers = sorted(weights[i,:],reverse=True)
#         vals = [round(np.mean(preferred_providers[:j])*(1-(1-p)**j),4) for j in range(1,len(preferred_providers)+1)]
#         max_matchings_per.append(np.argmax(vals)+1)
            
    min_matchings_per = min(round(1/p**(0.5)),simulator.max_menu_size)

    def get_solution(B):
        m = gp.Model("bipartite_matching")
        m.setParam('OutputFlag', 0)
        x = m.addVars(N, M, vtype=GRB.BINARY, name="x")

        m.setObjective(gp.quicksum(weights[i, j] * x[i, j] for i in range(N) for j in range(M)), GRB.MAXIMIZE)

        for j in range(M):
            m.addConstr(gp.quicksum(x[i,j] for i in range(N)) <= B)

        for i in range(N):
            m.addConstr(gp.quicksum(x[i,j] for j in range(M)) <= max_matchings_per[i], name=f"match_{j}")
            m.addConstr(gp.quicksum(x[i,j] for j in range(M)) >= min_matchings_per, name=f"match_{j}")

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
    print(sol)
    return sol
