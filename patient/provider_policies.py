import numpy as np
import gurobipy as gp
from gurobipy import GRB

import torch
import torch.nn as nn
import torch.optim as optim

from patient.lp_policies import lp_policy
from patient.baseline_policies import greedy_policy


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
            breakpoints = [0,1,4,8]  # Example range
            values = [0]+[0,100,100]# [np.log(x) for x in breakpoints[1:]]  # Precompute log values

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

        solution = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                if x[i, j].X > 0.5:
                    solution[i,j] = 1
        real_value = 0
        for j in range(M):
            lower = np.sum(solution[:,j])
            upper = (1-(1-p)**lower)
            prod = np.sum(solution[:,j]*weights[:,j])
            real_value += upper/lower*prod 
        
        # for i in range(N):
        #     log_term = np.log(np.sum(solution[i,:]))*lamb*p
        #     real_value -= log_term 

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


def objective(x, theta, p):
    # Compute the sum of x over rows for each column
    sum_x = torch.sum(x, dim=0)  # Shape: (columns,)

    # Compute numerator and denominator for the first term
    term1_num = 1 - (1 - p) ** sum_x  # Numerator
    term1_den = sum_x + 1e-2  # Add epsilon to avoid division by zero

    # Smoothly adjust the term using sum_x as a weighting factor
    smooth_factor = sum_x / (sum_x + 1e-2)  # Smoothly scales from 0 to 1
    term1 = smooth_factor * (term1_num / term1_den) * torch.sum(x * theta, dim=0)

    # Sum over columns and normalize by M
    term1 = torch.sum(term1) / theta.shape[1]
    return term1

def entropy_penalty(x, theta, p):
    row_sums = torch.sum(x, dim=1)  # Sum over columns for each row
    
    # Calculate the maximum value in each row
    row_max = torch.max(x, dim=1).values  # Maximum value in each row
    
    # Penalty is proportional to (row_sum - row_max)
    penalty = row_sums - row_max
    
    # Ensure penalty is non-negative (torch.relu ensures differentiability)
    penalty = torch.relu(penalty)+1
    
    # Sum penalty across rows
    total_penalty = torch.sum(penalty)/theta.shape[0]
    
    return total_penalty

def l1_regularization(x,theta,p):
    term3 = -torch.sum(x*torch.log(x+1e-8))
    return term3 

def objective_slow(z, theta, p, sorted_theta,rows_with_top_i,top_num_by_patient_provider,lamb=1, smooth_reg='entropy', epsilon=1e-5):
    x = torch.sigmoid(z)
        
    rows_with_top_i = torch.zeros((theta.shape[1], theta.shape[1]), device=theta.device)
    argsorted = torch.argsort(theta, dim=1, descending=True)

    # Mask elements in `argsorted` where x is 0
    mask = x.gather(1, argsorted) != 0

    # # Filter out -1 indices
    filtered_argsorted = [
        torch.masked_select(argsorted[i], mask[i]) for i in range(len(argsorted))
    ]

    # Populate `rows_with_top_i`
    for provider in range(theta.shape[1]):
        for patient in range(len(filtered_argsorted)):
            if x[patient, provider] == 0:
                continue 
            rows_with_top_i[provider,top_num_by_patient_provider[patient][provider]:] += 1
    rows_with_top_i /= (theta.shape[0] - 1)

    # Step 3: Compute `is_top_k`
    is_top_k = torch.zeros((theta.shape[0], theta.shape[1], theta.shape[1]), device=theta.device)
    
    if theta.shape[0] == 1:
        const = 1
    else:
        const = 1 / (theta.shape[0] - 1)
    for patient in range(theta.shape[0]):
        for provider in range(theta.shape[1]):
            is_top_k[patient, provider, top_num_by_patient_provider[patient][provider]:] = const 

    # Step 4: Normalize `x`
    normalized_x = torch.zeros_like(x)
    S = theta.shape[1]

    prod = torch.ones_like(is_top_k[:, :, 0], dtype=x.dtype)  # Shape: (x.shape[0], x.shape[1])
    tot = torch.zeros_like(prod)

    for top_num in range(S):
        tot += (1 / S) * prod  # Accumulate the total

        if top_num < S - 1:
            # Compute delta in a vectorized way
            delta = rows_with_top_i[:, top_num].unsqueeze(0) - is_top_k[:, :, top_num]
            prod *= (1 - p * delta)
    normalized_x = x * tot
    sorted_normalized_x = normalized_x.gather(1, sorted_theta)

    # Compute cumulative products (1 - normalized_x) along rows
    one_minus_sorted = 1 - sorted_normalized_x
    cumprods = torch.cumprod(one_minus_sorted, dim=1)

    # Shift the cumulative products to use for the original scaling (prepending 1 for first index)
    shifted_cumprods = torch.cat([torch.ones(cumprods.size(0), 1, device=cumprods.device), cumprods[:, :-1]], dim=1)

    # Apply the cumulative product scaling to the original indices
    scaled_normalized_x = sorted_normalized_x * shifted_cumprods

    # Scatter back to the original positions
    normalized_x = torch.zeros_like(normalized_x)
    normalized_x.scatter_(1, sorted_theta, scaled_normalized_x)
    # Normalize row-wise

    # Compute numerator for the first term (using normalized x)
    term = p * torch.sum(normalized_x * theta, dim=0)
        
    term = torch.sum(term) / theta.shape[1]  # Normalize by number of columns

    reg_term = 0
    # Add smooth regularization term
    if smooth_reg == 'logit' and lamb > 0:
        reg_term = torch.sum(torch.logit(x, eps=epsilon) ** 2)  # Logit-based penalty
    elif smooth_reg == 'entropy' and lamb > 0:
        reg_term = -torch.sum(x * torch.log(x + epsilon) + (1 - x) * torch.log(1 - x + epsilon))  # Entropy-based penalty
    loss = term - lamb * reg_term

    return loss


def objective_fast(z, theta, p, sorted_theta,rows_with_top_i,top_num_by_patient_provider,lamb=1, smooth_reg='entropy', epsilon=1e-5):
    # Reparameterize x using sigmoid
    x = torch.sigmoid(z)  # x is now bounded in [0, 1]
    
    # Compute the sum of x across all columns for each row
    row_sums = torch.sum(x, dim=0, keepdim=True)  # Shape: (rows, 1)
    
    # Normalize x by row sums
    normalized_x = x / (p*torch.maximum(row_sums, torch.tensor(1.0, device=x.device)))*(1-(1-p)**(torch.maximum(row_sums, torch.tensor(1.0, device=x.device)))) 

    sorted_normalized_x = normalized_x.gather(1, sorted_theta)

    # Compute cumulative products (1 - normalized_x) along rows
    one_minus_sorted = 1 - sorted_normalized_x
    cumprods = torch.cumprod(one_minus_sorted, dim=1)

    # Shift the cumulative products to use for the original scaling (prepending 1 for first index)
    shifted_cumprods = torch.cat([torch.ones(cumprods.size(0), 1, device=cumprods.device), cumprods[:, :-1]], dim=1)

    # Apply the cumulative product scaling to the original indices
    scaled_normalized_x = sorted_normalized_x * shifted_cumprods

    # Scatter back to the original positions
    normalized_x = torch.zeros_like(normalized_x)
    normalized_x.scatter_(1, sorted_theta, scaled_normalized_x)
    # Normalize row-wise

    prod = p*torch.sum(normalized_x,dim=0)

    # Compute numerator for the first term (using normalized x)
    term1_num = prod * torch.sum(normalized_x * theta, dim=0)

    term1_den = torch.sum(normalized_x, dim=0) + 1e-8  # Avoid division by zero
    term1_den = torch.maximum(term1_den,torch.tensor(1.0, device=x.device))

    # Compute the main term
    term1 = (term1_num / term1_den)
        
    term1 = torch.sum(term1) / theta.shape[1]  # Normalize by number of columns

    reg_term = 0
    # Add smooth regularization term
    if smooth_reg == 'logit' and lamb > 0:
        reg_term = torch.sum(torch.logit(x, eps=epsilon) ** 2)  # Logit-based penalty
    elif smooth_reg == 'entropy' and lamb > 0:
        reg_term = -torch.sum(x * torch.log(x + epsilon) + (1 - x) * torch.log(1 - x + epsilon))  # Entropy-based penalty
    loss = term1 - lamb * reg_term

    return loss

def gradient_descent_policy(simulator):
    import time 
    start = time.time() 
    p = simulator.choice_model_settings['top_choice_prob']

    theta = [p.provider_rewards for p in simulator.patients]
    theta = torch.Tensor(theta)

    sorted_theta = torch.argsort(theta, dim=1,descending=True)  

    rows_with_top_i = np.zeros((theta.shape[1],theta.shape[1]))

    argsorted = [np.argsort(i).numpy()[::-1] for i in theta]

    for provider in range(len(rows_with_top_i)):
        curr = 0
        for i in range(len(rows_with_top_i[provider])):
            for j in range(len(argsorted)):
                if argsorted[j][i] == provider:
                    curr += 1
            rows_with_top_i[provider][i] = curr
    rows_with_top_i /= theta.shape[0]

    top_num_by_patient_provider = [[theta.shape[0] for i in range(theta.shape[1])] for i in range(theta.shape[0])]

    for i in range(theta.shape[0]):
        for top_num in range(theta.shape[1]):
            top_num_by_patient_provider[i][sorted_theta[i][top_num]] = top_num


    N = len(simulator.patients)
    M = theta.shape[1]

    best_x = None
    best_loss = 1000

    start = time.time() 
    for _ in range(1): 
        if _ == 0:
            x = torch.Tensor(lp_policy(simulator))*10-10/2  
        else:
            x = torch.rand(N, M, requires_grad=True)  
        x.requires_grad = True
        scale = 10

        # Optimizer
        if scale == 10:
            optimizer = optim.Adam([x], lr=0.25)
        elif scale == 100:
            optimizer = optim.Adam([x], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

        values_by_loss = []

        # Training loop
        for epoch in range(10*scale):  # Adjust number of iterations as needed
            optimizer.zero_grad()

            
            if epoch > 8*scale:
                lamb = 1
            elif epoch > 7*scale:
                lamb = 0.25
            else:
                lamb = 0
            # Compute the objective
            loss = -objective_slow(x, theta, p,sorted_theta,rows_with_top_i,top_num_by_patient_provider,lamb=lamb)

            true_loss = -objective_slow(torch.round(torch.sigmoid(x))*1000-500, theta, p,sorted_theta,rows_with_top_i,top_num_by_patient_provider,lamb=0)
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_([x], max_norm=10)

            # Gradient step
            optimizer.step()
            scheduler.step()

            if epoch%10 == 0:
                values_by_loss.append((true_loss.detach(),torch.sigmoid(x).detach()))
        
        min_loc = np.argmin([i[0] for i in values_by_loss])
        if values_by_loss[min_loc][0] < best_loss:
            best_loss = values_by_loss[min_loc][0]
            best_x = values_by_loss[min_loc][1]
    return np.round(((best_x).detach().numpy()))


def gradient_descent_policy_fast(simulator):
    import time 
    start = time.time() 
    p = simulator.choice_model_settings['top_choice_prob']

    theta = [p.provider_rewards for p in simulator.patients]
    theta = torch.Tensor(theta)

    sorted_theta = torch.argsort(theta, dim=1,descending=True)  

    rows_with_top_i = np.zeros((theta.shape[1],theta.shape[1]))

    argsorted = [np.argsort(i).numpy()[::-1] for i in theta]

    for provider in range(len(rows_with_top_i)):
        curr = 0
        for i in range(len(rows_with_top_i[provider])):
            for j in range(len(argsorted)):
                if argsorted[j][i] == provider:
                    curr += 1
            rows_with_top_i[provider][i] = curr
    rows_with_top_i /= theta.shape[0]

    top_num_by_patient_provider = [[theta.shape[0] for i in range(theta.shape[1])] for i in range(theta.shape[0])]

    for i in range(theta.shape[0]):
        for top_num in range(theta.shape[1]):
            top_num_by_patient_provider[i][sorted_theta[i][top_num]] = top_num


    N = len(simulator.patients)
    M = theta.shape[1]

    best_x = None
    best_loss = 1000

    start = time.time() 
    for _ in range(1): 
        if _ == 0:
            x = torch.Tensor(lp_policy(simulator))*10-10/2  
        else:
            x = torch.rand(N, M, requires_grad=True)  
        x.requires_grad = True
        scale = 10

        # Optimizer
        if scale == 10:
            optimizer = optim.Adam([x], lr=0.25)
        elif scale == 100:
            optimizer = optim.Adam([x], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

        values_by_loss = []

        # Training loop
        for epoch in range(10*scale):  # Adjust number of iterations as needed
            optimizer.zero_grad()

            
            if epoch > 8*scale:
                lamb = 1
            elif epoch > 7*scale:
                lamb = 0.25
            else:
                lamb = 0
            # Compute the objective
            loss = -objective_fast(x, theta, p,sorted_theta,rows_with_top_i,top_num_by_patient_provider,lamb=lamb)

            true_loss = -objective_fast(torch.round(torch.sigmoid(x))*1000-500, theta, p,sorted_theta,rows_with_top_i,top_num_by_patient_provider,lamb=0)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_([x], max_norm=10)

            # Gradient step
            optimizer.step()
            scheduler.step()

            if epoch%10 == 0:
                values_by_loss.append((true_loss.detach(),torch.sigmoid(x).detach()))
        
        min_loc = np.argmin([i[0] for i in values_by_loss])
        if values_by_loss[min_loc][0] < best_loss:
            best_loss = values_by_loss[min_loc][0]
            best_x = values_by_loss[min_loc][1]
    return np.round(((best_x).detach().numpy()))