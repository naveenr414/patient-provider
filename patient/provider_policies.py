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

def objective(z, theta, p, sorted_theta,rows_with_top_i,lamb=1, smooth_reg='entropy', epsilon=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.sigmoid(z)
        
    rows_with_top_i = torch.zeros((theta.shape[1], theta.shape[0]), device=theta.device)
    argsorted = torch.argsort(theta, dim=1, descending=True)

    # Mask elements in `argsorted` where x is 0
    mask = x.gather(1, argsorted) != 0
    filtered_argsorted = torch.where(mask, argsorted, -1)  # Replace invalid indices with -1
    is_top_k = torch.zeros((theta.shape[0], theta.shape[1], theta.shape[0]), device=theta.device)

    # We need to update rows_with_top_i for each provider from their rank onwards
    for patient in range(len(filtered_argsorted)):
        # For each provider at the current rank
        for rank in range(len(filtered_argsorted[patient])):
            provider = filtered_argsorted[patient][rank]
            # Update rows_with_top_i for this provider starting from the current rank onwards
            rows_with_top_i[provider, rank:] += 1
            is_top_k[patient,provider,rank:] +=  1 / (theta.shape[0] - 1)


    # Normalize rows_with_top_i
    rows_with_top_i /= (theta.shape[0] - 1)

    # Step 4: Normalize `x`
    normalized_x = torch.zeros_like(x).to(device)
    delta = rows_with_top_i[None, :, :] - is_top_k
    delta = torch.roll(delta, shifts=1, dims=2)  # Shift columns to the right (dims=2 corresponds to columns)
    delta[:, :, 0] = 0  # Set the leftmost column to 1


    delta = 1-p*delta
    delta_raised = torch.cumprod(delta,dim=2)
    # Raise delta to successive powers along the third dimension
    delta_swapped = delta_raised.permute(0, 2, 1)

    normalized_x = x[:,None,:] * delta_swapped

    sorted_normalized_x = normalized_x.gather(dim=2, index=sorted_theta.unsqueeze(1).expand(-1, normalized_x.size(1), -1))

    # Compute cumulative products (1 - normalized_x) along rows
    one_minus_sorted = 1 - sorted_normalized_x
    cumprods = torch.cumprod(one_minus_sorted, dim=2)

    # # Shift the cumulative products to use for the original scaling (prepending 1 for first index)
    shifted_cumprods = torch.cat([torch.ones(cumprods.size(0), cumprods.size(1) ,1,device=cumprods.device), cumprods[:,:,:-1]], dim=2).to(device)


    # Apply the cumulative product scaling to the original indices
    scaled_normalized_x = sorted_normalized_x * shifted_cumprods
    scaled_normalized_x = torch.mean(scaled_normalized_x,dim=1)

    # Scatter back to the original positions
    normalized_x = torch.zeros_like(scaled_normalized_x)
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

def gradient_descent_policy_2(simulator):
    global torch
    p = simulator.choice_model_settings['top_choice_prob']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    theta = [p.provider_rewards for p in simulator.patients]
    theta = torch.Tensor(theta).to(device)

    rows_with_top_i = np.zeros((theta.shape[1],theta.shape[0]))

    argsorted = torch.argsort(theta, dim=1, descending=True)
    provider_indices = torch.arange(theta.shape[1], device=device)
    rows_with_top_i = torch.zeros((theta.shape[1], theta.shape[0]), device=device)
    for i in range(theta.shape[0]):
        provider_ranks = argsorted[i]
        rows_with_top_i[provider_ranks, provider_indices] += 1
    rows_with_top_i = torch.cumsum(rows_with_top_i, dim=1) / theta.shape[0]

    N = len(simulator.patients)
    M = theta.shape[1]

    best_x = None
    best_loss = 1000
    for _ in range(2): 
        if _ == 0:
            x = torch.tensor(lp_policy(simulator), dtype=torch.float32, device=device) * 10 - 5
        else:
            x = torch.rand(N, M, requires_grad=True, device=device)

        x.requires_grad = True

        print("Device {}".format(x.device))

        # Optimizer
        optimizer = optim.Adam([x], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

        values_by_loss = []

        # Training loop
        for epoch in range(1000):  # Adjust number of iterations as needed
            optimizer.zero_grad()
            
            if epoch > 800:
                lamb = 1
            elif epoch > 700:
                lamb = 0.25
            else:
                lamb = 0
            # Compute the objective
            import torch.profiler

            with torch.profiler.profile(with_stack=True) as prof:
                loss = -objective(x, theta, p,argsorted,rows_with_top_i,lamb=lamb)
                loss.backward()


            if lamb != 0:
                true_loss = -objective(torch.round(torch.sigmoid(x))*1000-500, theta, p,argsorted,rows_with_top_i,lamb=0)
            else:
                true_loss = loss

            # Backpropagation
            torch.nn.utils.clip_grad_norm_([x], max_norm=10)

            # Gradient step
            optimizer.step()
            scheduler.step()

            values_by_loss.append((true_loss.detach(), torch.sigmoid(x).detach()))
        
        min_loc = torch.argmin(torch.tensor([i[0] for i in values_by_loss], device=device))
        if values_by_loss[min_loc][0] < best_loss:
            best_loss = values_by_loss[min_loc][0]
            best_x = values_by_loss[min_loc][1]
    return torch.round(best_x).cpu().numpy()
