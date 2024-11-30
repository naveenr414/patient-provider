import numpy as np
import gurobipy as gp
from gurobipy import GRB

import torch
import torch.nn as nn
import torch.optim as optim


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
    print("B {}, Per provider {}, Per Patient {}".format(max_b+1,np.sum(sol,axis=0),np.sum(sol,axis=1)))
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

def gradient_descent_policy(simulator):
    p = simulator.choice_model_settings['top_choice_prob']

    theta = [p.provider_rewards for p in simulator.patients]
    theta = torch.Tensor(theta)
    N = len(simulator.patients)
    M = theta.shape[1]

    lamb = 0
    lamb2 = 0

    best_loss = float('inf')
    best_x = None
    for _ in range(10):  # Run 5 independent optimizations
        x = torch.rand(N, M, requires_grad=True)  # Variables to optimize (not constrained to [0, 1])

        # Optimizer
        optimizer = optim.Adam([x], lr=0.01)

        values_by_loss = []

        # Training loop
        for epoch in range(250):  # Adjust number of iterations as needed
            optimizer.zero_grad()
            
            # Compute the objective
            loss = -objective(x, theta, p)
            loss += lamb*entropy_penalty(x,theta,p)
            loss += lamb2*l1_regularization(x,theta,p)  # Minimize negative of objective (maximize objective)
            
            # Backpropagation
            loss.backward()

            # Gradient step
            optimizer.step()
            
            # Clip x to enforce constraints
            with torch.no_grad():
                x.clamp_(0, 1)  # Ensure x_{i,j} stays in [0, 1]
            values_by_loss.append((loss.detach(),x.detach()))
        loss_values = [i[0] for i in values_by_loss]
        final_loss = np.min(loss_values)
        if final_loss < best_loss:
            best_loss = final_loss
            best_x = values_by_loss[np.argmin(loss_values)][1]
    return (best_x > 0.1).detach().int().numpy()
    
def objective(z, theta, p, lamb=1, smooth_reg='entropy', epsilon=1e-5):
    # Reparameterize x using sigmoid
    x = torch.sigmoid(z)  # x is now bounded in [0, 1]
    
    # Compute the sum of x over rows for each column
    sum_x = torch.sum(x, dim=0)  # Shape: (columns,)
    
    # Compute the sum of x across all columns for each row
    row_sums = torch.sum(x, dim=1, keepdim=True)  # Shape: (rows, 1)
    
    # Normalize x by row sums
    normalized_x = x / torch.maximum(row_sums, torch.tensor(1.0, device=sum_x.device))  # Avoid division by zero
    
    # Compute numerator for the first term (using normalized x)
    term1_num = (1 - (1 - p) ** sum_x) * torch.sum(normalized_x * theta, dim=0)
    
    # Compute denominator for the first term (using normalized x)
    term1_den = torch.sum(x, dim=0) + 1e-8  # Avoid division by zero
    term1_den = torch.maximum(term1_den,torch.tensor(1.0, device=sum_x.device))

    # Compute the main term
    term1 = (term1_num / term1_den)
        
    
    term1 = torch.sum(term1) / theta.shape[1]  # Normalize by number of columns


    # Add smooth regularization term
    if smooth_reg == 'logit':
        reg_term = torch.sum(torch.logit(x, eps=epsilon) ** 2)  # Logit-based penalty
    elif smooth_reg == 'entropy':
        reg_term = -torch.sum(x * torch.log(x + epsilon) + (1 - x) * torch.log(1 - x + epsilon))  # Entropy-based penalty
    else:
        raise ValueError("Unsupported regularization: choose 'logit' or 'entropy'")
        # Final loss with regularization
    loss = term1 - lamb * reg_term
    
    return loss


def gradient_descent_policy_2(simulator):
    p = simulator.choice_model_settings['top_choice_prob']

    theta = [p.provider_rewards for p in simulator.patients]
    theta = torch.Tensor(theta)
    N = len(simulator.patients)
    M = theta.shape[1]

    lamb = 0
    lamb2 = 0

    best_loss = float('inf')
    best_x = None
    for _ in range(1):  # Run 5 independent optimizations
        x = torch.rand(N, M, requires_grad=True)  # Variables to optimize (not constrained to [0, 1])

        # Optimizer
        optimizer = optim.Adam([x], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

        values_by_loss = []

        # Training loop
        for epoch in range(1000):  # Adjust number of iterations as needed
            optimizer.zero_grad()
            
            if epoch > 800:
                lamb = 1
            elif epoch > 400:
                lamb = 0.25
            else:
                lamb = 0
            # Compute the objective
            loss = -objective(x, theta, p,lamb=lamb)
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_([x], max_norm=10)

            # Gradient step
            optimizer.step()
            scheduler.step()
            
            # Clip x to enforce constraints
            # with torch.no_grad():
            #     x.clamp_(0, 1)  # Ensure x_{i,j} stays in [0, 1]
            values_by_loss.append((loss.detach(),torch.sigmoid(x).detach()))
        
        best_x = values_by_loss[-1][1]
    return np.round(((best_x).detach().numpy()))
