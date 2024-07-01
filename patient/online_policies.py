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

def p_approximation(simulator,patient,available_providers,memory,per_epoch_function):
    if memory == None:
        weights = [p.provider_rewards for p in simulator.patients]
        weights = np.array(weights)

        # TODO: Justify this
        max_per_provider = simulator.provider_max_capacity
        max_per_provider *= max(1,len(simulator.patients)/len(available_providers))

        LP_solution = solve_linear_program(weights,max_per_provider)

        matchings = [0 for i in range(weights.shape[0])]
        for (i,j) in LP_solution:
            matchings[i] = j

        memory = matchings 

    default_menu = [0 for i in range(len(available_providers))]
    default_menu[memory[patient.idx]] = 1
    
    return default_menu, memory 

def p_approximation_balance(simulator,patient,available_providers,memory,per_epoch_function):
    if memory == None:
        available_providers  = [1 if i > 0 else 0 for i in simulator.provider_capacities]
        weights = [p.provider_rewards for p in simulator.patients]
        weights = np.array(weights)
        weights = np.array([i*available_providers for i in weights])

        lamb = 1

        max_per_provider = simulator.provider_max_capacity
        max_per_provider *= max(1,len(simulator.patients)/len(available_providers))

        LP_solution = solve_linear_program(weights,max_per_provider,lamb=lamb)

        matchings = [0 for i in range(weights.shape[0])]
        for (i,j) in LP_solution:
            matchings[i] = j
        memory = matchings 

    default_menu = [0 for i in range(len(available_providers))]
    default_menu[memory[patient.idx]] = 1
    
    
    return default_menu, memory 

def p_approximation_with_additions(simulator,patient,available_providers,memory,per_epoch_function):
    if memory is None:
        weights = [p.provider_rewards for p in simulator.patients]
        weights = np.array(weights)

        max_per_provider = simulator.provider_max_capacity
        # max_per_provider *= max(1,len(simulator.patients)/len(available_providers))
        LP_solution = solve_linear_program(weights,max_per_provider)

        matchings = np.zeros((len(simulator.patients),len(available_providers)))
        pairs = [-1 for i in range(len(simulator.patients))]
        unmatched_providers = set(list(range(len(available_providers))))

        for (i,j) in LP_solution:
            matchings[i,j] = 1
            pairs[i] = j

            if j in unmatched_providers:
                unmatched_providers.remove(j)

        swap_score = np.zeros((len(simulator.patients),len(simulator.patients)))

        for i in range(len(simulator.patients)):
            for j in range(len(simulator.patients)):
                if pairs[i]>=0 and pairs[j] >= 0:
                    p = simulator.choice_model_settings['top_choice_prob']

                    if weights[i][pairs[j]] > weights[i][pairs[i]]:
                        score = 1/2*(p*weights[i][pairs[j]] + p*(1-p)*weights[j][pairs[j]] + p**2*weights[j][pairs[i]])
                        score += 1/2*(p*weights[j][pairs[j]] + p*(1-p)*weights[i][pairs[j]] + p**2*weights[i][pairs[i]]) 
                    elif weights[j][pairs[i]] > weights[j][pairs[j]]:
                        score = 1/2*(p*weights[j][pairs[i]] + p*(1-p)*weights[i][pairs[i]] + p**2*weights[i][pairs[j]])
                        score += 1/2*(p*weights[i][pairs[i]] + p*(1-p)*weights[j][pairs[i]] + p**2*weights[j][pairs[j]]) 
                    else:
                        score = p*(weights[i][pairs[i]] + weights[j][pairs[j]])
                    
                    score -= p*(weights[i][pairs[i]] + weights[j][pairs[j]])

                    swap_score[i,j] = score
                    swap_score[j,i] = score 
                elif pairs[i] >= 0:
                    p = simulator.choice_model_settings['top_choice_prob']
                    score = 1/2*(p*weights[i][pairs[i]] + (1-p)*p*weights[j][pairs[i]])
                    score += 1/2*(p*weights[j][pairs[i]] + (1-p)*p*weights[i][pairs[i]])
                    score -= p*weights[i][pairs[i]]
                    swap_score[i,j] = swap_score[j,i] = score

        if np.min(swap_score) == 0:
            for i in range(len(simulator.patients)):
                for j in range(len(simulator.patients)):
                    if pairs[j] >= 0:
                        matchings[i][pairs[j]] = 1
        else:
            non_zero = np.nonzero(swap_score)
            max_triplets = []

            for i in range(len(non_zero[0])):
                x,y = non_zero[0][i], non_zero[1][i]
                
                for j in range(len(simulator.patients)):
                    if j!= x and j!=y:
                        scores = swap_score[x][j] + swap_score[j][y] + swap_score[x][y]
                        if scores > 0:
                            max_triplets.append((scores,j,x,y))
            
            max_triplets = sorted(max_triplets,key=lambda k: k[0])
            used_indices = set() 

            for (_,a,b,c) in max_triplets:
                if a in used_indices or b in used_indices or c in used_indices:
                    continue 

                if pairs[a] >= 0:
                    matchings[b][pairs[a]] = 1
                    matchings[c][pairs[a]] = 1
                if pairs[b] >= 0:
                    matchings[a][pairs[b]] = 1
                    matchings[c][pairs[b]] = 1
                if pairs[c] >= 0:
                    matchings[b][pairs[c]] = 1
                    matchings[a][pairs[c]] = 1
                used_indices.add(a)
                used_indices.add(b)
                used_indices.add(c)

            max_pairs = []
            for i in range(len(non_zero[0])):
                x,y = non_zero[0][i], non_zero[1][i]
                scores = swap_score[x][y]
                max_pairs.append(((scores,x,y)))
            max_pairs = sorted(max_pairs,key=lambda k: k[0])
            for (_,a,b) in max_pairs:
                if a in used_indices or b in used_indices:
                    continue 
                
                if pairs[a] >= 0:
                    matchings[b][pairs[a]] = 1
                if pairs[b] >= 0:
                    matchings[a][pairs[b]] = 1
                used_indices.add(a)
                used_indices.add(b)

        for i in range(len(simulator.patients)):
            for j in unmatched_providers:
                matchings[i][j] = 1

        memory = matchings 

    default_menu = memory[patient.idx]
    
    return default_menu, memory 

def p_approximation_with_additions_balance(simulator,patient,available_providers,memory,per_epoch_function):
    if memory is None:
        lamb = 1
        weights = [p.provider_rewards for p in simulator.patients]
        weights = np.array(weights)

        max_per_provider = simulator.provider_max_capacity

        LP_solution = solve_linear_program(weights,max_per_provider,lamb=lamb)

        matchings = np.zeros((len(simulator.patients),len(available_providers)))
        pairs = [-1 for i in range(len(simulator.patients))]

        unmatched_providers = set(list(range(len(available_providers))))
        for (i,j) in LP_solution:
            matchings[i,j] = 1
            pairs[i] = j

            if j in unmatched_providers:
                unmatched_providers.remove(j)

        swap_score = np.zeros((len(simulator.patients),len(simulator.patients)))

        for i in range(len(simulator.patients)):
            for j in range(len(simulator.patients)):
                if pairs[i]>=0 and pairs[j] >= 0:
                    p = simulator.choice_model_settings['top_choice_prob']

                    if weights[i][pairs[j]] > weights[i][pairs[i]]:
                        score = 1/2*(p*weights[i][pairs[j]] + p*(1-p)*weights[j][pairs[j]] + p**2*weights[j][pairs[i]])
                        score += 1/2*(p*weights[j][pairs[j]] + p*(1-p)*weights[i][pairs[j]] + p**2*weights[i][pairs[i]]) 
                    elif weights[j][pairs[i]] > weights[j][pairs[j]]:
                        score = 1/2*(p*weights[j][pairs[i]] + p*(1-p)*weights[i][pairs[i]] + p**2*weights[i][pairs[j]])
                        score += 1/2*(p*weights[i][pairs[i]] + p*(1-p)*weights[j][pairs[i]] + p**2*weights[j][pairs[j]]) 
                    else:
                        score = p*(weights[i][pairs[i]] + weights[j][pairs[j]])
                    
                    score -= p*(weights[i][pairs[i]] + weights[j][pairs[j]])

                    swap_score[i,j] = score
                    swap_score[j,i] = score 
                elif pairs[i] >= 0:
                    p = simulator.choice_model_settings['top_choice_prob']
                    score = 1/2*(p*weights[i][pairs[i]] + (1-p)*p*weights[j][pairs[i]])
                    score += 1/2*(p*weights[j][pairs[i]] + (1-p)*p*weights[i][pairs[i]])
                    score -= p*weights[i][pairs[i]]
                    swap_score[i,j] = swap_score[j,i] = score

        if np.min(swap_score) == 0:
            for i in range(len(simulator.patients)):
                for j in range(len(simulator.patients)):
                    if pairs[j] >= 0:
                        matchings[i][pairs[j]] = 1
        else:
            non_zero = np.nonzero(swap_score)
            max_triplets = []

            for i in range(len(non_zero[0])):
                x,y = non_zero[0][i], non_zero[1][i]
                
                for j in range(len(simulator.patients)):
                    if j!= x and j!=y:
                        scores = swap_score[x][j] + swap_score[j][y] + swap_score[x][y]
                        if scores > 0:
                            max_triplets.append((scores,j,x,y))
            
            max_triplets = sorted(max_triplets,key=lambda k: k[0])
            used_indices = set() 

            for (_,a,b,c) in max_triplets:
                if a in used_indices or b in used_indices or c in used_indices:
                    continue 

                if pairs[a] >= 0:
                    matchings[b][pairs[a]] = 1
                    matchings[c][pairs[a]] = 1
                if pairs[b] >= 0:
                    matchings[a][pairs[b]] = 1
                    matchings[c][pairs[b]] = 1
                if pairs[c] >= 0:
                    matchings[b][pairs[c]] = 1
                    matchings[a][pairs[c]] = 1
                used_indices.add(a)
                used_indices.add(b)
                used_indices.add(c)

            max_pairs = []
            for i in range(len(non_zero[0])):
                x,y = non_zero[0][i], non_zero[1][i]
                scores = swap_score[x][y]
                max_pairs.append(((scores,x,y)))
            max_pairs = sorted(max_pairs,key=lambda k: k[0])
            for (_,a,b) in max_pairs:
                if a in used_indices or b in used_indices:
                    continue 
                
                if pairs[a] >= 0:
                    matchings[b][pairs[a]] = 1
                if pairs[b] >= 0:
                    matchings[a][pairs[b]] = 1
                used_indices.add(a)
                used_indices.add(b)

        for i in range(len(simulator.patients)):
            for j in unmatched_providers:
                matchings[i][j] = 1


        memory = matchings 

    default_menu = memory[patient.idx]
    
    return default_menu, memory 

def p_approximation_with_additions_balance_learning(simulator,patient,available_providers,memory,per_epoch_function):
    if memory is None:
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

        lamb = 1
        weights = np.array(weights)

        max_per_provider = simulator.provider_max_capacity

        LP_solution = solve_linear_program(weights,max_per_provider,lamb=lamb)

        matchings = np.zeros((len(simulator.patients),len(available_providers)))
        pairs = [-1 for i in range(len(simulator.patients))]

        unmatched_providers = set(list(range(len(available_providers))))
        for (i,j) in LP_solution:
            matchings[i,j] = 1
            pairs[i] = j

            if j in unmatched_providers:
                unmatched_providers.remove(j)

        swap_score = np.zeros((len(simulator.patients),len(simulator.patients)))

        for i in range(len(simulator.patients)):
            for j in range(len(simulator.patients)):
                if pairs[i]>=0 and pairs[j] >= 0:
                    p = simulator.choice_model_settings['top_choice_prob']

                    if weights[i][pairs[j]] > weights[i][pairs[i]]:
                        score = 1/2*(p*weights[i][pairs[j]] + p*(1-p)*weights[j][pairs[j]] + p**2*weights[j][pairs[i]])
                        score += 1/2*(p*weights[j][pairs[j]] + p*(1-p)*weights[i][pairs[j]] + p**2*weights[i][pairs[i]]) 
                    elif weights[j][pairs[i]] > weights[j][pairs[j]]:
                        score = 1/2*(p*weights[j][pairs[i]] + p*(1-p)*weights[i][pairs[i]] + p**2*weights[i][pairs[j]])
                        score += 1/2*(p*weights[i][pairs[i]] + p*(1-p)*weights[j][pairs[i]] + p**2*weights[j][pairs[j]]) 
                    else:
                        score = p*(weights[i][pairs[i]] + weights[j][pairs[j]])
                    
                    score -= p*(weights[i][pairs[i]] + weights[j][pairs[j]])

                    swap_score[i,j] = score
                    swap_score[j,i] = score 
                elif pairs[i] >= 0:
                    p = simulator.choice_model_settings['top_choice_prob']
                    score = 1/2*(p*weights[i][pairs[i]] + (1-p)*p*weights[j][pairs[i]])
                    score += 1/2*(p*weights[j][pairs[i]] + (1-p)*p*weights[i][pairs[i]])
                    score -= p*weights[i][pairs[i]]
                    swap_score[i,j] = swap_score[j,i] = score

        if np.min(swap_score) == 0:
            for i in range(len(simulator.patients)):
                for j in range(len(simulator.patients)):
                    if pairs[j] >= 0:
                        matchings[i][pairs[j]] = 1
        else:
            non_zero = np.nonzero(swap_score)
            max_triplets = []

            for i in range(len(non_zero[0])):
                x,y = non_zero[0][i], non_zero[1][i]
                
                for j in range(len(simulator.patients)):
                    if j!= x and j!=y:
                        scores = swap_score[x][j] + swap_score[j][y] + swap_score[x][y]
                        if scores > 0:
                            max_triplets.append((scores,j,x,y))
            
            max_triplets = sorted(max_triplets,key=lambda k: k[0])
            used_indices = set() 

            for (_,a,b,c) in max_triplets:
                if a in used_indices or b in used_indices or c in used_indices:
                    continue 

                if pairs[a] >= 0:
                    matchings[b][pairs[a]] = 1
                    matchings[c][pairs[a]] = 1
                if pairs[b] >= 0:
                    matchings[a][pairs[b]] = 1
                    matchings[c][pairs[b]] = 1
                if pairs[c] >= 0:
                    matchings[b][pairs[c]] = 1
                    matchings[a][pairs[c]] = 1
                used_indices.add(a)
                used_indices.add(b)
                used_indices.add(c)

            max_pairs = []
            for i in range(len(non_zero[0])):
                x,y = non_zero[0][i], non_zero[1][i]
                scores = swap_score[x][y]
                max_pairs.append(((scores,x,y)))
            max_pairs = sorted(max_pairs,key=lambda k: k[0])
            for (_,a,b) in max_pairs:
                if a in used_indices or b in used_indices:
                    continue 
                
                if pairs[a] >= 0:
                    matchings[b][pairs[a]] = 1
                if pairs[b] >= 0:
                    matchings[a][pairs[b]] = 1
                used_indices.add(a)
                used_indices.add(b)

        for i in range(len(simulator.patients)):
            for j in unmatched_providers:
                matchings[i][j] = 1

        memory = matchings 

    default_menu = memory[patient.idx]

    return default_menu, memory 