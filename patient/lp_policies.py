import numpy as np 
from patient.utils import solve_linear_program, solve_linear_program_dynamic
import gurobipy as gp
from gurobipy import GRB
import itertools 
import torch
import torch.nn as nn
import torch.nn.functional as F


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

def get_gradient_policy(threshold,ordering):
    def policy(w,m):
        return gradient_policy(w,m,threshold=threshold,ordering=ordering)
    
    return policy 

class AssortmentOptimizer:
    """
    Optimized robust optimizer for capacity-constrained assignment.
    Key speedups: vectorization, compiled functions, reduced adversarial steps.
    """
    
    def __init__(self, weights, capacities, n_permutations=10, temperature=1.0, 
                 epsilon=0.1, norm_type='linf', device='cpu', compile_model=False):
        """
        Args:
            weights: (N, M) array where weights[i,j] is patient i's preference for provider j
                     Last column is the exit option (always available)
            capacities: (M,) array of capacities (last one ignored for exit option)
            n_permutations: Number of random permutations to sample
            temperature: Temperature for softmax (lower = more discrete)
            epsilon: Maximum perturbation budget for adversarial weights
            norm_type: 'linf' for L-infinity norm, 'l2' for L2 norm
            device: 'cpu' or 'cuda'
            compile_model: Use torch.compile for JIT compilation (PyTorch 2.0+)
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
        
        # Try to compile for speed (PyTorch 2.0+)
        if compile_model:
            try:
                self._compute_single_permutation_differentiable = torch.compile(
                    self._compute_single_permutation_differentiable
                )
            except:
                pass
        
    def compute_objective(self, X, weights, permutations=None):
        """
        Differentiable objective using soft assignments.
        
        Args:
            X: (N, M-1) tensor, soft assortment probabilities
            weights: (N, M) tensor, (possibly perturbed) patient weights
            permutations: Optional list of permutations
            
        Returns:
            Average utility across all permutations
        """
        if permutations is None:
            permutations = [torch.randperm(self.n_patients, device=self.device) 
                          for _ in range(self.n_permutations)]
        
        total_utility = 0.0
        
        for perm in permutations:
            utility = self._compute_single_permutation_differentiable(X, weights, perm)
            total_utility += utility
            
        return total_utility / self.n_permutations
    
    def _compute_single_permutation_differentiable(self, X, weights, permutation):
        """
        Optimized differentiable version using soft selection.
        """
        remaining_capacity = self.capacities[:-1].clone()  # Exclude exit option capacity
        total_utility = 0.0
        
        # Pre-compute sigmoid once if capacity doesn't change much
        sigmoid_factor = 10.0
        
        for patient_idx in permutation:
            patient_weights = weights[patient_idx]  # (M,)
            patient_assortment = X[patient_idx]  # (M-1,)
            
            # Soft availability for providers
            capacity_available = torch.sigmoid(remaining_capacity * sigmoid_factor)
            available_providers = patient_assortment * capacity_available  # (M-1,)
            
            # Exit option always fully available
            available_full = torch.cat([
                available_providers, 
                torch.ones(1, device=self.device)
            ])  # (M,)
            
            # Compute selection probabilities
            logits = patient_weights * available_full / self.temperature
            selection_probs = F.softmax(logits, dim=0)  # (M,)
            
            # Expected utility (vectorized)
            selected_utility = torch.dot(selection_probs, patient_weights)
            
            # Soft capacity update (only for providers)
            provider_selection_probs = selection_probs[:self.n_providers]
            remaining_capacity = remaining_capacity - provider_selection_probs
            
            total_utility += selected_utility
            
        return total_utility
    
    def find_worst_case_perturbation(self, X, permutations, n_adv_steps=5, adv_lr=0.05):
        """
        Optimized adversarial perturbation search.
        Uses fewer steps and higher learning rate for speed.
        
        Args:
            X: Current assortment (N, M-1)
            permutations: Fixed permutations for evaluation
            n_adv_steps: Number of gradient steps (reduced default: 5)
            adv_lr: Learning rate (increased default: 0.05)
            
        Returns:
            Perturbed weights that minimize the objective
        """
        if self.epsilon == 0:
            return self.weights_original
        
        # Initialize perturbation with requires_grad
        delta = torch.zeros_like(self.weights_original, requires_grad=True)
        
        # Use SGD instead of tracking gradients manually (faster)
        adv_optimizer = torch.optim.SGD([delta], lr=adv_lr)
        
        for _ in range(n_adv_steps):
            adv_optimizer.zero_grad()
            
            perturbed_weights = self.weights_original + delta
            obj = self.compute_objective(X, perturbed_weights, permutations)
            
            # Minimize objective w.r.t. delta
            obj.backward()
            adv_optimizer.step()
            
            # Project onto epsilon ball
            with torch.no_grad():
                if self.norm_type == 'linf':
                    delta.clamp_(-self.epsilon, self.epsilon)
                elif self.norm_type == 'l2':
                    # Vectorized L2 projection
                    delta_norm = delta.norm(p=2)
                    if delta_norm > self.epsilon:
                        delta.mul_(self.epsilon / delta_norm)
        
        return (self.weights_original + delta).detach()
    
    def optimize(self, n_iterations=100, lr=0.1, n_adv_steps=5, adv_lr=0.05,
                 temperature_schedule=None, verbose=True, eval_frequency=10):
        """
        Fast robust optimization with reduced evaluations.
        
        Args:
            n_iterations: Number of outer loop iterations
            lr: Learning rate for X optimization
            n_adv_steps: Number of inner loop steps (default reduced to 5)
            adv_lr: Learning rate for adversarial optimization (default increased to 0.05)
            temperature_schedule: Optional function for temperature annealing
            verbose: Print progress
            eval_frequency: How often to evaluate and print (default: every 10 iters)
            
        Returns:
            X_optimal: (N, M-1) binary assortment matrix
            history: Dict with 'objective' and 'worst_case_objective' lists
        """
        # Initialize X with logits
        X_logits = nn.Parameter(torch.randn(  # Random init can be faster than zeros
            self.n_patients, self.n_providers, 
            device=self.device, requires_grad=True
        ) * 0.01)
        
        optimizer = torch.optim.Adam([X_logits], lr=lr)
        history = {'objective': [], 'worst_case_objective': [], 'iterations': []}
        
        # Sample fixed permutations once
        fixed_perms = [torch.randperm(self.n_patients, device=self.device) 
                      for _ in range(self.n_permutations)]
        
        for iteration in range(n_iterations):
            # Update temperature if schedule provided
            if temperature_schedule is not None:
                self.temperature = temperature_schedule(iteration)
            
            # Convert logits to probabilities
            X_soft = torch.sigmoid(X_logits)
            
            # Find worst-case perturbation (inner minimization)
            worst_weights = self.find_worst_case_perturbation(
                X_soft.detach(), fixed_perms, n_adv_steps, adv_lr
            )
            
            # Optimize X against worst-case weights (outer maximization)
            optimizer.zero_grad()
            
            obj_worst = self.compute_objective(X_soft, worst_weights, fixed_perms)
            loss = -obj_worst  # Negative for maximization
            
            loss.backward()
            optimizer.step()
            
            # Only evaluate periodically to save time
            if iteration % eval_frequency == 0 or iteration == n_iterations - 1:
                with torch.no_grad():
                    obj_original = self.compute_objective(X_soft, self.weights_original, fixed_perms)
                
                history['objective'].append(obj_original.item())
                history['worst_case_objective'].append(obj_worst.item())
                history['iterations'].append(iteration)
                
                if verbose:
                    temp_str = f", Temp = {self.temperature:.3f}" if temperature_schedule else ""
                    eps_str = f", ε = {self.epsilon:.3f}" if self.epsilon > 0 else ""
                    print(f"Iter {iteration}: Obj = {obj_original.item():.4f}, "
                          f"Worst = {obj_worst.item():.4f}{temp_str}{eps_str}")
        
        # Convert to hard assignment
        X_optimal = (torch.sigmoid(X_logits) > 0.5).float()
        
        return X_optimal.detach().cpu().numpy(), history


class UltraFastAssortmentOptimizer(AssortmentOptimizer):
    """
    Ultra-fast version with aggressive approximations:
    - Fewer permutations
    - Adaptive adversarial steps (fewer later in training)
    - Early stopping
    """
    
    def optimize(self, n_iterations=100, lr=0.1, initial_adv_steps=10, 
                 min_adv_steps=3, adv_lr=0.05, temperature_schedule=None, 
                 verbose=True, eval_frequency=10, early_stop_threshold=0.01):
        """
        Ultra-fast optimization with adaptive adversarial budget.
        """
        X_logits = nn.Parameter(torch.randn(
            self.n_patients, self.n_providers, 
            device=self.device, requires_grad=True
        ) * 0.01)
        
        optimizer = torch.optim.Adam([X_logits], lr=lr)
        history = {'objective': [], 'worst_case_objective': [], 'iterations': []}
        
        fixed_perms = [torch.randperm(self.n_patients, device=self.device) 
                      for _ in range(self.n_permutations)]
        
        prev_obj = float('-inf')
        
        for iteration in range(n_iterations):
            if temperature_schedule is not None:
                self.temperature = temperature_schedule(iteration)
            
            # Adaptive adversarial steps: more at start, fewer later
            current_adv_steps = max(
                min_adv_steps,
                int(initial_adv_steps * (1 - iteration / n_iterations))
            )
            
            X_soft = torch.sigmoid(X_logits)
            
            worst_weights = self.find_worst_case_perturbation(
                X_soft.detach(), fixed_perms, current_adv_steps, adv_lr
            )
            
            optimizer.zero_grad()
            obj_worst = self.compute_objective(X_soft, worst_weights, fixed_perms)
            loss = -obj_worst
            loss.backward()
            optimizer.step()
            
            if iteration % eval_frequency == 0 or iteration == n_iterations - 1:
                with torch.no_grad():
                    obj_original = self.compute_objective(X_soft, self.weights_original, fixed_perms)
                
                history['objective'].append(obj_original.item())
                history['worst_case_objective'].append(obj_worst.item())
                history['iterations'].append(iteration)
                
                # Early stopping
                improvement = obj_original.item() - prev_obj
                if iteration > 20 and improvement < early_stop_threshold:
                    if verbose:
                        print(f"Early stopping at iteration {iteration} (improvement: {improvement:.4f})")
                    break
                
                prev_obj = obj_original.item()
                
                if verbose:
                    print(f"Iter {iteration}: Obj = {obj_original.item():.4f}, "
                          f"Worst = {obj_worst.item():.4f}, AdvSteps = {current_adv_steps}")
        
        X_optimal = (torch.sigmoid(X_logits) > 0.5).float()
        return X_optimal.detach().cpu().numpy(), history


def gradient_policy(parameters,error_threshold=0.1):
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

    optimizer = UltraFastAssortmentOptimizer(weights, capacities, n_permutations=10)
    X_optimal, history = optimizer.optimize(n_iterations=100, lr=0.1)
    return np.array(X_optimal)
    
def simulate_ordering(assortment, weights, max_per_provider, ordering):
    """
    Simulate one ordering with given assortment.
    
    Returns total reward for this ordering.
    """
    N, M = weights.shape
    M_real = M - 1
    
    capacities = np.array([max_per_provider] * M_real + [N])  # Last option has infinite capacity
    total_reward = 0
    
    for person_id in ordering:
        # Find available options in assortment
        available_mask = np.concatenate([assortment[person_id], [1]])  # Outside option always available
        available_mask[:M_real] = available_mask[:M_real] * (capacities[:M_real] > 0)
        
        # Choose best available option
        masked_weights = weights[person_id] * available_mask
        masked_weights[available_mask == 0] = -np.inf
        chosen = np.argmax(masked_weights)
        
        # Update capacity and reward
        capacities[chosen] -= 1
        total_reward += weights[person_id, chosen]
    
    return total_reward


def gradient_policy_fast(parameters, threshold=0.1, ordering="uniform", 
                         num_permutations=10, learning_rate=0.1, num_iterations=10):
    """
    Fast gradient-based approximation using continuous relaxation.
    
    Uses a differentiable softmax relaxation and gradient ascent.
    """
    weights = parameters['weights']
    capacities = parameters['capacities']
    N, M = weights.shape
    M_real = M - 1
    
    # Initialize assortment probabilities (logits)
    assortment_logits = np.zeros((N, M_real))
    
    # Generate random permutations once
    orderings = [np.random.permutation(N) for _ in range(num_permutations)]
    
    best_assortment = None
    best_value = -np.inf
    
    for iteration in range(num_iterations):
        # Convert logits to probabilities using sigmoid
        assortment_probs = 1 / (1 + np.exp(-assortment_logits))
        
        # Compute gradient using finite differences (simple but effective)
        gradient = np.zeros((N, M_real))
        epsilon = 0.01
        
        for i in range(N):
            for j in range(M_real):
                # Current value
                current_assortment = (assortment_probs > 0.5).astype(int)
                current_value = np.mean([
                    simulate_ordering(current_assortment, weights, capacities, perm)
                    for perm in orderings
                ])
                
                # Perturbed value
                assortment_logits[i, j] += epsilon
                perturbed_probs = 1 / (1 + np.exp(-assortment_logits))
                perturbed_assortment = (perturbed_probs > 0.5).astype(int)
                perturbed_value = np.mean([
                    simulate_ordering(perturbed_assortment, weights, capacities, perm)
                    for perm in orderings
                ])
                assortment_logits[i, j] -= epsilon
                
                # Gradient
                gradient[i, j] = (perturbed_value - current_value) / epsilon
        
        # Update logits
        assortment_logits += learning_rate * gradient
        
        # Track best solution
        current_assortment = (assortment_probs > 0.5).astype(int)
        current_value = np.mean([
            simulate_ordering(current_assortment, weights, capacities, perm)
            for perm in orderings
        ])
        
        if current_value > best_value:
            best_value = current_value
            best_assortment = current_assortment.copy()
    
    return best_assortment
