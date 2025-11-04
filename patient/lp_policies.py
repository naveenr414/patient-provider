import numpy as np 
from patient.utils import solve_linear_program
import gurobipy as gp
from gurobipy import GRB
import itertools 
import torch
import torch.nn as nn
import torch.nn.functional as F
import json 
import random 


def lp_policy(parameters):
    """Policy which selects according to the LP, in an offline fashion
    
    Arguments:
        simulator: Simulator for patient-provider matching
    
    Returns: List of providers on the menu"""
    weights = parameters['weights']
    capacities = parameters['capacities']
    N,M = weights.shape 
    M -= 1 
    lp_solution = solve_linear_program(weights,capacities)
    assortment = np.zeros((N,M))
    
    for (i,j) in lp_solution:
        assortment[i,j] = 1
    
    return np.array(assortment)

def get_gradient_policy(N,M,fairness_constraint,seed):
    patient_data = json.load(open("../../data/patient_data_{}_{}_{}_comorbidity.json".format(seed,N,M)))
    zipcode_clusters = json.load(open("../../data/ct_zipcode_cluster.json"))
    clusters = list(zipcode_clusters.values())
    for i in patient_data:
        if i['location'] not in zipcode_clusters:
            zipcode_clusters[i['location']] = random.choice(clusters)
    cluster_by_patient = [zipcode_clusters[i['location']] for i in patient_data]

    def policy(parameters):
        if fairness_constraint == -1:
            return gradient_policy(parameters,fairness_threshold=None)
        else:
            return gradient_policy(parameters,fairness_threshold=fairness_constraint,cluster_by_patient=cluster_by_patient)
    
    return policy 

class AssortmentOptimizer:
    """
    Robust optimizer with fairness constraints across patient clusters.
    Ensures average utility across clusters doesn't differ by more than fairness_threshold.
    """
    
    def __init__(self, weights, capacities, cluster_assignments=None, 
                 n_permutations=10, temperature=1.0, epsilon=0.1, 
                 fairness_threshold=None, fairness_penalty=0.5,
                 norm_type='linf', device='cpu'):
        """
        Args:
            weights: (N, M) array where weights[i,j] is patient i's preference for provider j
                     Last column is the exit option (always available)
            capacities: (M,) array of capacities (last one ignored for exit option)
            cluster_assignments: (N,) array mapping each patient to a cluster ID
                                If None, no fairness constraints applied
            n_permutations: Number of random permutations to sample
            temperature: Temperature for softmax (lower = more discrete)
            epsilon: Maximum perturbation budget for adversarial weights
            fairness_threshold: Maximum allowed difference in average utility between clusters
                               If None, no fairness constraints
            fairness_penalty: Weight for fairness violation penalty in loss
            norm_type: 'linf' for L-infinity norm, 'l2' for L2 norm
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.weights_original = torch.tensor(weights, dtype=torch.float32, device=device)
        self.capacities = torch.tensor(capacities, dtype=torch.float32, device=device)
        self.n_patients, self.n_options = self.weights_original.shape
        self.n_providers = self.n_options - 1
        self.n_permutations = n_permutations
        self.temperature = temperature
        self.epsilon = epsilon
        self.norm_type = norm_type
        self.fairness_threshold = fairness_threshold
        self.fairness_penalty = fairness_penalty
        self.cached_adv_directions = None

        # Setup cluster information
        if cluster_assignments is not None:
            self.cluster_assignments = torch.tensor(cluster_assignments, dtype=torch.long, device=device)
            self.clusters = torch.unique(self.cluster_assignments)
            self.n_clusters = len(self.clusters)
            
            # Pre-compute cluster membership masks for efficiency
            self.cluster_masks = {}
            self.cluster_sizes = {}
            for c in self.clusters.tolist():
                mask = (self.cluster_assignments == c)
                self.cluster_masks[c] = mask
                self.cluster_sizes[c] = mask.sum().item()
        else:
            self.cluster_assignments = None
            self.clusters = None
            self.n_clusters = 0

    def compute_fairness_violation(self, cluster_avg_utilities):
        """
        Compute fairness constraint violation.
        
        Args:
            cluster_avg_utilities: Dict mapping cluster_id -> average utility
            
        Returns:
            Total fairness violation (sum of pairwise differences exceeding threshold)
        """
        if self.fairness_threshold is None or self.cluster_assignments is None:
            return torch.tensor(0.0, device=self.device)
        
        violation = 0.0
        cluster_list = list(cluster_avg_utilities.keys())
        
        # Check all pairs of clusters
        u = torch.stack(list(cluster_avg_utilities.values()))  # (K,)
        pairwise_diff = torch.abs(u.unsqueeze(0) - u.unsqueeze(1))
        violation = F.relu(pairwise_diff - self.fairness_threshold).sum() / 2  # avoid double-counting
        
        return violation

    def compute_objective(self, X, weights, permutations=None, return_cluster_utilities=False):
        """
        Differentiable objective using soft assignments with batched permutation processing.
        
        Args:
            X: (N, M-1) tensor, soft assortment probabilities
            weights: (N, M) tensor, (possibly perturbed) patient weights
            permutations: Optional list of permutations
            return_cluster_utilities: If True, also return per-cluster average utilities
            
        Returns:
            Average utility across all permutations (and optionally cluster utilities)
        """
        if permutations is None:
            permutations = [torch.randperm(self.n_patients, device=self.device) 
                        for _ in range(self.n_permutations)]
        
        # Batch process all permutations at once
        patient_utilities = self._compute_batch_permutations_differentiable(X, weights, permutations)
        
        # Average across permutations: (N,)
        avg_patient_utilities = patient_utilities.mean(dim=0)
        avg_utility = avg_patient_utilities.sum()

        if return_cluster_utilities and self.cluster_assignments is not None:
            cluster_avg_utilities = {}
            for c in self.clusters.tolist():
                mask = self.cluster_masks[c]
                cluster_avg_utilities[c] = avg_patient_utilities[mask].mean()
            return avg_utility, cluster_avg_utilities
        
        return avg_utility


    def _compute_batch_permutations_differentiable(self, X, weights, permutations):
        """
        Vectorized differentiable version processing ALL permutations in parallel.
        
        Args:
            X: (N, M-1) tensor
            weights: (N, M) tensor
            permutations: list of B permutation tensors, each (N,)
        
        Returns:
            (B, N) tensor of utilities, where [b, i] is utility for patient i in permutation b
        """
        B = len(permutations)  # batch size (number of permutations)
        N, M_minus_1 = X.shape
        M = self.n_options
        device = self.device
        
        # Stack all permutation indices: (B, N)
        perm_indices = torch.stack(permutations)
        
        # Batch index into weights and X: (B, N, M) and (B, N, M-1)
        weights_perm = weights[perm_indices]  # (B, N, M)
        X_perm = X[perm_indices]              # (B, N, M-1)
        
        # Initialize remaining capacities for each permutation: (B, M-1)
        remaining_capacity = self.capacities[:-1].unsqueeze(0).expand(B, -1).clone()
        
        # Compute soft availability based on capacity
        capacity_scale = 10.0
        cap_avail = torch.sigmoid(remaining_capacity * capacity_scale)  # (B, M-1)
        
        # Combine provider availability + exit option: (B, N, M)
        # X_perm: (B, N, M-1), cap_avail: (B, M-1) -> need to broadcast
        available_providers = X_perm * cap_avail.unsqueeze(1)  # (B, N, M-1)
        available_full = torch.cat([
            available_providers,
            torch.ones(B, N, 1, device=device)  # exit option always available
        ], dim=2)  # (B, N, M)
        
        # Compute logits and selection probabilities: (B, N, M)
        logits = weights_perm * available_full / self.temperature
        selection_probs = F.softmax(logits, dim=2)  # softmax over M options
        
        # Expected utilities for each patient in each permutation: (B, N)
        patient_utilities = (selection_probs * weights_perm).sum(dim=2)
        
        # --- Soft capacity update approximation ---
        # Provider load per permutation: sum over patients, normalized
        # selection_probs[:, :, :M-1]: (B, N, M-1)
        provider_load = selection_probs[:, :, :self.n_providers].sum(dim=1) / N  # (B, M-1)
        remaining_capacity = remaining_capacity - provider_load
        remaining_capacity = torch.clamp(remaining_capacity, min=0.0)
        
        # Restore original patient order for each permutation
        # Create inverse permutations
        batch_range = torch.arange(B, device=device).unsqueeze(1)  # (B, 1)
        inverse_perms = torch.argsort(perm_indices, dim=1)  # (B, N)
        
        # Gather back to original order: (B, N)
        patient_utilities = patient_utilities[batch_range, inverse_perms]
        
        return patient_utilities  # (B, N)


    def find_worst_case_perturbation_fgsm(
        self,
        X_soft,
        permutations,
        n_adv_steps=None,   # kept for compatibility
        adv_lr=None,        # kept for compatibility
        epsilon=None
    ):
        """
        Fast FGSM-style approximation to find worst-case perturbation of weights.
        Now uses batched permutation processing for speed.

        Args:
            X_soft (torch.Tensor): [N, M-1] soft assortment matrix
            permutations (list[torch.Tensor]): list of patient permutations
            n_adv_steps (int): unused (kept for compatibility)
            adv_lr (float): unused (kept for compatibility)
            epsilon (float): override perturbation magnitude (default: self.epsilon)

        Returns:
            torch.Tensor: worst-case perturbed weights, same shape as self.weights_original
        """
        if epsilon is None:
            epsilon = self.epsilon

        # If epsilon == 0, skip adversarial step
        if epsilon == 0:
            return self.weights_original

        # Clone and set requires_grad
        weights_adv = self.weights_original.clone().detach().requires_grad_(True)

        # Compute the differentiable objective across permutations (now batched!)
        obj = self.compute_objective(X_soft, weights_adv, permutations)

        # We *minimize* worst-case utility, so take gradient of negative objective
        loss = -obj
        grad = torch.autograd.grad(loss, weights_adv, retain_graph=False)[0]

        # FGSM perturbation: move weights in direction that *minimizes* utility
        if self.norm_type == 'linf':
            perturbation = epsilon * grad.sign()
        elif self.norm_type == 'l2':
            grad_norm = grad.norm(p=2, dim=1, keepdim=True) + 1e-8
            perturbation = epsilon * grad / grad_norm
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

        worst_weights = weights_adv + perturbation
        worst_weights = torch.clamp(worst_weights, min=0.0)
        return worst_weights.detach()

    
    def optimize(self, n_iterations=100, lr=0.1, n_adv_steps=5, adv_lr=0.05,
                 temperature_schedule=None, verbose=True, eval_frequency=10):
        """
        Fair robust optimization with cluster fairness constraints.
        
        Args:
            n_iterations: Number of outer loop iterations
            lr: Learning rate for X optimization
            n_adv_steps: Number of inner loop steps for adversarial perturbation
            adv_lr: Learning rate for adversarial optimization
            temperature_schedule: Optional function for temperature annealing
            verbose: Print progress
            eval_frequency: How often to evaluate and print
            
        Returns:
            X_optimal: (N, M-1) binary assortment matrix
            history: Dict with objective, worst_case, fairness violation
        """
        # Initialize X with logits
        X_logits = nn.Parameter(torch.randn(
            self.n_patients, self.n_providers, 
            device=self.device, requires_grad=True
        ) * 0.01)
        
        optimizer = torch.optim.Adam([X_logits], lr=lr)
        history = {
            'objective': [], 
            'worst_case_objective': [], 
            'fairness_violation': [],
            'cluster_utilities': [],
            'iterations': []
        }
        
        # Sample fixed permutations once
        fixed_perms = [torch.randperm(self.n_patients, device=self.device) 
                      for _ in range(self.n_permutations)]
        
        for iteration in range(n_iterations):
            if temperature_schedule is not None:
                self.temperature = temperature_schedule(iteration)
            
            X_soft = torch.sigmoid(X_logits)
            if iteration % 10 == 0 or self.cached_adv_directions is None:
                worst_weights = self.find_worst_case_perturbation_fgsm(
                    X_soft.detach(), fixed_perms, n_adv_steps, adv_lr
                )                
                self.cached_adv_directions = (worst_weights - self.weights_original)
            else:
                worst_weights = self.weights_original + self.cached_adv_directions

            # Find worst-case perturbation

            
            # Compute objective with fairness penalty
            optimizer.zero_grad()
            if self.fairness_threshold is not None:
                obj_worst, cluster_utils = self.compute_objective(
                    X_soft, worst_weights, fixed_perms, return_cluster_utilities=True
                )
                fairness_viol = self.compute_fairness_violation(cluster_utils)
                
                # Combined loss: maximize utility while minimizing fairness violation
                loss = -obj_worst + self.fairness_penalty * len(worst_weights)*fairness_viol
            else:
                obj_worst = self.compute_objective(X_soft, worst_weights, fixed_perms)
                fairness_viol = torch.tensor(0.0)
                loss = -obj_worst
            
            loss.backward()
            optimizer.step()
            
            # Periodic evaluation
            if iteration % eval_frequency == 0 or iteration == n_iterations - 1:
                with torch.no_grad():
                    if self.fairness_threshold is not None:
                        obj_original, cluster_utils_orig = self.compute_objective(
                            X_soft, self.weights_original, fixed_perms, 
                            return_cluster_utilities=True
                        )
                        fairness_viol_orig = self.compute_fairness_violation(cluster_utils_orig)
                    else:
                        obj_original = self.compute_objective(X_soft, self.weights_original, fixed_perms)
                        fairness_viol_orig = torch.tensor(0.0)
                        cluster_utils_orig = {}
                
                history['objective'].append(obj_original.item())
                history['worst_case_objective'].append(obj_worst.item())
                history['fairness_violation'].append(fairness_viol_orig.item())
                history['cluster_utilities'].append({k: v.item() for k, v in cluster_utils_orig.items()})
                history['iterations'].append(iteration)
                
        # Convert to hard assignment
        X_optimal = (torch.sigmoid(X_logits) > 0.5).float()
        
        return X_optimal.detach().cpu().numpy(), history


def gradient_policy(parameters,fairness_threshold=None,cluster_by_patient=None):
    """
    Find optimal assortment policy by optimizing over multiple random orderings.
    
    Arguments:
        weights: N x M matrix where weights[i,j] is reward for person i choosing option j
                 Last column (M-1) is the outside option with infinite capacity
        max_per_provider: Capacity for each of the first M-1 options
        threshold: Threshold parameter (unused for now)
        ordering: Type of ordering ("uniform" for random permutations)
        num_permutations: Number of random orderings to consider
    
    Returns:
        assortment: N x (M-1) binary matrix indicating which options to show each person
        objective_value: Expected reward across all orderings
    """
    weights = parameters['weights']
    capacities = parameters['capacities']

    optimizer = AssortmentOptimizer(weights, capacities, n_permutations=10,fairness_threshold=fairness_threshold,cluster_assignments=cluster_by_patient)
    X_optimal, _ = optimizer.optimize(n_iterations=100, lr=0.1)
    return np.array(X_optimal)
    