import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math

import torch
import torch.nn as nn
import torch.optim as optim

from patient.lp_policies import lp_policy
from patient.baseline_policies import greedy_policy

def objective_fast(z, theta, p, sorted_theta,max_per_provider,lamb=1, smooth_reg='entropy', epsilon=1e-5):
    """Our Lower-Bound policy, which computes the lower bound given a proposed assortment
    
    Arguments:
        z: Assortment, of size patients x providers; this is passed in as logits
        theta: Match quality matrix, of size patients x providers
        p: Match probability, chance of each match
        sorted_theta: Sorted indices per-patient, determining which providers
            each patient prefers
        lamb: Penalty for the assortment not being {0,1}
        smooth_reg: Which penalty to use; either entropy or logit
        epsilon: Float, small number so that values <= epsilon are not penalized in the 
            regularization
    
    Returns: Float, the objective value for this assortment"""

    # Reparameterize x using sigmoid
    x = torch.sigmoid(z)  # x is now bounded in [0, 1]
    
    # Compute the sum of x across all columns for each row
    row_sums = torch.sum(x, dim=0, keepdim=True)  # Shape: (rows, 1)
    
    # Normalize x by row sums
    normalized_x = x / (p*torch.maximum(row_sums, torch.tensor(1.0, device=x.device)))

    if max_per_provider > 1:
        row_sums = torch.maximum(row_sums, torch.tensor(1.0, device=x.device))
        kp = row_sums * p
        condition = max_per_provider > kp  # Boolean mask
        case1 = torch.pow(1 - p, row_sums)

        stddev = torch.sqrt(kp * (1 - p))
        z = (kp - max_per_provider) / stddev
        denom = z**2 + 1
        normal_factor = 1 / math.sqrt(2 * math.pi)
        case2 = (z / denom) * normal_factor * torch.exp(-0.5 * z**2)
        result = torch.where(condition, case1, case2)
        result = 1-result
    else:
        result = (1-(1-p)**(torch.maximum(row_sums, torch.tensor(1.0, device=x.device))))

    normalized_x = normalized_x *result 

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
        
    term1 = torch.sum(term1) / theta.shape[0]  # Normalize by number of columns

    reg_term = 0
    # Add smooth regularization term
    if smooth_reg == 'logit' and lamb > 0:
        reg_term = torch.sum(torch.logit(x, eps=epsilon) ** 2)  # Logit-based penalty
    elif smooth_reg == 'entropy' and lamb > 0:
        reg_term = -torch.sum(x * torch.log(x + epsilon) + (1 - x) * torch.log(1 - x + epsilon))  # Entropy-based penalty
    loss = term1 - lamb * reg_term

    return loss

def gradient_descent_policy_fast(simulator):
    """Run gradient descent to find the optimal assortment
        According to the objective_fast function
        
    Arguments:
        simulator: Simulator object
    
    Returns: Assortment, 0-1 matrix of size patients x providers"""

    p = simulator.choice_model_settings['top_choice_prob']
    max_per_provider = simulator.provider_max_capacity

    theta = [p.provider_rewards for p in simulator.patients]
    theta = torch.Tensor(theta)

    sorted_theta = torch.argsort(theta, dim=1,descending=True)  
    top_num_by_patient_provider = [[theta.shape[0] for i in range(theta.shape[1])] for i in range(theta.shape[0])]

    for i in range(theta.shape[0]):
        for top_num in range(theta.shape[1]):
            top_num_by_patient_provider[i][sorted_theta[i][top_num]] = top_num


    N = len(simulator.patients)
    M = theta.shape[1]

    best_x = None
    best_loss = 1000
    num_trials = 3
    max_gradients = 0
    for _ in range(num_trials): 
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
        # TODO: Change this back to 10*scale
        for epoch in range(100*scale):  # Adjust number of iterations as needed
            optimizer.zero_grad()
            if epoch > 8*scale:
                lamb = 1
            elif epoch > 7*scale:
                lamb = 0.25
            else:
                lamb = 0
            # Compute the objective
            loss = -objective_fast(x, theta, p,sorted_theta,max_per_provider,lamb=lamb)

            true_loss = -objective_fast(torch.round(torch.sigmoid(x))*1000-500, theta, p,sorted_theta,max_per_provider,lamb=0)

            # Backpropagation
            loss.backward()
            max_gradients = max(max_gradients,torch.max(x.grad).numpy())

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