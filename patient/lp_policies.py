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
                 n_permutations=10, temperature=0.5, epsilon=0.1, 
                 fairness_threshold=None, fairness_penalty=0.5,
                 norm_type='linf', device='cpu', max_shown=25):
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
            max_shown: Maximum number of providers to show each patient (row sum constraint)
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
        self.max_shown = max_shown

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


    def compute_objective(self, X_constrained, weights, permutations=None, return_cluster_utilities=False):
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
        patient_utilities = self._compute_batch_permutations_differentiable(
            X_constrained, weights, permutations
        )
        
        # Average across permutations: (N,)
        avg_patient_utilities = patient_utilities.mean(dim=0)
        avg_utility = avg_patient_utilities.sum()/len(avg_patient_utilities)

        if return_cluster_utilities and self.cluster_assignments is not None:
            cluster_avg_utilities = {}
            for c in self.clusters.tolist():
                mask = self.cluster_masks[c]
                cluster_avg_utilities[c] = avg_patient_utilities[mask].mean()
            return avg_utility, cluster_avg_utilities
        return avg_utility

    def _compute_batch_permutations_differentiable(self, X, weights, permutations):
        """
        Sequential-permutation differentiable simulator.
        Replaces the previous vectorized-but-non-sequential implementation.
        """
        B = len(permutations)  # batch size (number of permutations)
        N, M_minus_1 = X.shape
        M = self.n_options
        device = self.device

        perm_indices = torch.stack(permutations)          # (B, N)
        weights_perm = weights[perm_indices]              # (B, N, M)
        X_perm = X[perm_indices]                          # (B, N, M-1)

        # remaining capacity per permutation (B, M-1)
        remaining_capacity = self.capacities[:-1].unsqueeze(0).expand(B, -1).clone()

        # Prepare output utilities (B, N)
        patient_utils = torch.zeros(B, N, device=device)

        for t in range(N):
            # soft availability for the t-th patient in each permutation
            # Option A: soft availability (original style)
            # cap_avail = torch.sigmoid(remaining_capacity * 10.0)  # (B, M-1)
            # available_providers = X_perm[:, t, :] * cap_avail  # (B, M-1)
            mask = torch.sigmoid(20 * remaining_capacity)

            # Option B: hard availability mask (closer to discrete)
            available_providers = X_perm[:, t, :] * mask # (remaining_capacity > 0).float()  # (B, M-1)

            # combine with exit option
            available_full = torch.cat([
                available_providers,
                torch.ones(B, 1, device=device)
            ], dim=1)  # (B, M)

            logits = weights_perm[:, t, :] * available_full / (self.temperature + 1e-12)  # (B, M)
            selection_probs = torch.nn.functional.softmax(logits, dim=1)  # (B, M)

            # expected utility for these patients
            patient_utils[:, t] = (selection_probs * weights_perm[:, t, :]).sum(dim=1)

            # update remaining_capacity using soft decrement (differentiable)
            # subtract expected allocation from capacity
            remaining_capacity = remaining_capacity - selection_probs[:, :self.n_providers]
            remaining_capacity = torch.clamp(remaining_capacity, min=0.0)

        # patient_utils currently in permuted order; invert to original order
        inverse_perms = torch.argsort(perm_indices, dim=1)  # (B, N)
        batch_range = torch.arange(B, device=device).unsqueeze(1)  # (B, 1)
        patient_utils = patient_utils[batch_range, inverse_perms]  # (B, N)

        return patient_utils  # (B, N)

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
            
            # Compute objective with fairness penalty
            optimizer.zero_grad()
            obj_worst = self.compute_objective(X_soft, self.weights_original, fixed_perms)
            fairness_viol = torch.tensor(0.0)
            loss = -obj_worst

            loss.backward()
            optimizer.step()
            
            # Periodic evaluation
            if iteration % eval_frequency == 0 or iteration == n_iterations - 1:
                with torch.no_grad():
                    X_soft_eval = torch.sigmoid(X_logits)
                    
                    if self.fairness_threshold is not None:
                        obj_original, cluster_utils_orig = self.compute_objective(
                            X_soft_eval, self.weights_original, fixed_perms, 
                            return_cluster_utilities=True
                        )
                        fairness_viol_orig = self.compute_fairness_violation(cluster_utils_orig)
                    else:
                        obj_original = self.compute_objective(X_soft_eval, self.weights_original, fixed_perms)
                        fairness_viol_orig = torch.tensor(0.0)
                        cluster_utils_orig = {}
                    
                    # Track row sums after constraint
                    X_hard = self._get_hard_assignment(X_logits)
                    obj_disc = self.evaluate_discrete(X_hard.cpu().numpy(), self.weights_original, fixed_perms)
                    print("Iter {}: Continuous {}, Discrete {}".format(iteration+1,obj_worst,obj_disc))


                history['objective'].append(obj_original.item())
                history['worst_case_objective'].append(obj_worst.item())
                history['fairness_violation'].append(fairness_viol_orig.item())
                history['cluster_utilities'].append({k: v.item() for k, v in cluster_utils_orig.items()})
                history['iterations'].append(iteration)
                
        # Convert to hard assignment with constraint
        X_soft_final = torch.sigmoid(X_logits)
                
        return X_soft_final.detach().cpu().numpy(), history
    
    def evaluate_discrete(self, X_binary, weights=None, permutations=None):
        if weights is None:
            weights = self.weights_original
        if permutations is None:
            permutations = [torch.randperm(self.n_patients, device=self.device) for _ in range(self.n_permutations)]

        X_binary = torch.tensor(X_binary, dtype=torch.float32, device=self.device)
        all_utils = []
        for perm in permutations:
            utils = self._simulate_discrete_single_perm(X_binary, weights, perm)
            all_utils.append(utils)
        avg_patient_utilities = torch.stack(all_utils).mean(dim=0)
        return avg_patient_utilities.mean().item()

    def _simulate_discrete_single_perm(self, X, weights, perm):
        N = self.n_patients
        M = self.n_providers
        remaining_cap = self.capacities[:-1].clone()
        utilities = torch.zeros(N, device=self.device)
        for t in range(N):
            pidx = perm[t]
            patient_assort = X[pidx]  # (M,)
            has_cap = (remaining_cap > 0).float()
            avail = patient_assort * has_cap
            provider_utils = weights[pidx, :M] * avail
            exit_util = weights[pidx, M]
            all_utils = torch.cat([provider_utils, exit_util.unsqueeze(0)])
            sel = torch.argmax(all_utils)
            utilities[pidx] = all_utils[sel]
            if sel < M:
                remaining_cap[sel] -= 1
        return utilities

    def _get_hard_assignment(self, X_logits):
        N = self.n_patients
        M = self.n_providers
        X_hard = torch.zeros(N, M, device=self.device)
        for i in range(N):
            # if logits are a Parameter, move to cpu/numpy extraction not needed; use torch.topk directly
            _, top_idx = torch.topk(X_logits[i], k=min(self.max_shown, M))
            X_hard[i, top_idx] = 1.0
        return X_hard

    def _hard_assignment_with_max_shown(self, X_soft):
        """
        Convert soft assignments to hard binary assignments respecting max_shown.
        For each patient, select top max_shown providers by probability.
        
        Args:
            X_soft: (N, M-1) soft probabilities
            
        Returns:
            X_hard: (N, M-1) binary matrix
        """
        N, M = X_soft.shape
        X_hard = torch.zeros_like(X_soft)
        
        for i in range(N):
            # Get top max_shown indices
            top_k = min(self.max_shown, M)
            _, top_indices = torch.topk(X_soft[i], k=top_k)
            X_hard[i, top_indices] = 1.0
        
        return X_hard

def simulate_assortment(X_sol, weights_list, capacities, permutations=[]):
    N, M = X_sol.shape
    M_plus1 = M + 1
    K = len(weights_list)
    
    simulated_rewards = np.zeros(K)
    z = np.zeros((N,K))
    
    for k, weights in enumerate(weights_list):
        rewards_per_perm = []
        perm = permutations[k]
        remaining_capacity = capacities.copy()
        remaining_capacity.append(N)  # exit option, effectively infinite
        reward = 0.0
        
        for idx,i in enumerate(perm):
            # Determine available options (offered & capacity > 0)
            available = np.zeros(M_plus1)
            for j in range(M):
                if X_sol[i,j] == 1 and remaining_capacity[j] > 0:
                    available[j] = 1
            available[M] = 1  # exit always available
            

            # Choose the option with max weight
            choices = np.where(available)[0]
            selected = choices[np.argmax(weights[i, choices])]
            # if k == 1 and idx<=2:
            #     print("Avaialble",available)
            #     print("Choice {}".format(selected))

            reward += weights[i, selected]
            z[i,k] =weights[i, selected] 
            
            # Update capacity if not exit
            if selected < M:
                remaining_capacity[selected] -= 1
        
        rewards_per_perm.append(reward)
        
        simulated_rewards[k] = np.mean(rewards_per_perm)
    # print(z[permutations[1],1])
    return simulated_rewards

def create_random_weights(weights,epsilon):
    noisy = weights + np.random.uniform(-epsilon, epsilon, weights.shape)
    noisy = np.clip(noisy, 0, 1)
    margin = 1e-6   # much larger than bonus
    noisy = margin + (1 - 2*margin) * noisy
    bonus_eps = 1e-12
    J = weights.shape[1]
    bonus = bonus_eps * np.arange(J)[None, :]
    noisy = noisy + bonus
    return noisy 

def evaluate_fixed_X(X_sol, weights_list, capacities, permutations):
    N, M = X_sol.shape
    M_plus1 = M + 1
    K = len(weights_list)

    model = gp.Model("evaluate_fixed_X")
    model.Params.LogToConsole = 0

    # Variables (same as before)
    y = model.addVars(N, M_plus1, K, vtype=GRB.BINARY, name="y")
    z = model.addVars(N, K, vtype=GRB.CONTINUOUS, name="z")
    c = model.addVars(N, M, K, vtype=GRB.CONTINUOUS, name="c")

    for k in range(K):
        weights_k = weights_list[k]
        ordering = permutations[k]

        # One selection per patient
        for i in range(N):
            model.addConstr(gp.quicksum(y[i,j,k] for j in range(M_plus1)) == 1)

        # Sequence + capacities
        for t in range(N):
            patient = ordering[t]

            for j in range(M):
                if t == 0:
                    # initial capacity
                    model.addConstr(c[t,j,k] == capacities[j])
                else:
                    prev_patient = ordering[t-1]
                    model.addConstr(c[t,j,k] == c[t-1,j,k] - y[prev_patient,j,k])

                # availability constraints
                if X_sol[patient, j] == 0:
                    model.addConstr(y[patient,j,k] == 0)
                else:
                    model.addConstr(y[patient,j,k] <= c[t,j,k])  # capacity >0 if chosen

            # exit option always available
            pass

        # link utility
        for i in range(N):
            model.addConstr(z[i,k] == gp.quicksum(weights_k[i,j] * y[i,j,k]
                                                  for j in range(M_plus1)))

        # rationality (Big-M)
        bigM = 1
        for t in range(N):
            patient = ordering[t]
            for j in range(M_plus1):
                for l in range(M_plus1):
                    if j == l:
                        continue
                    if l < M:
                        if X_sol[patient, l] == 1:
                            model.addConstr(z[patient,k] >=
                                weights_k[patient,l] -
                                bigM * (1 - y[patient,j,k]))
                    else:
                        # exit always available
                        model.addConstr(z[patient,k] >=
                            weights_k[patient,l] -
                            bigM * (1 - y[patient,j,k]))

    # objective value for fixed X
    model.setObjective(gp.quicksum(z[i,k]
                        for i in range(N) for k in range(K)), GRB.MAXIMIZE)

    model.optimize()

    # Extract y, z, c, objective
    y_sol = np.zeros((N, M_plus1, K))
    z_sol = np.zeros((N, K))
    c_sol = np.zeros((N, M, K))

    feasible = model.status == GRB.OPTIMAL

    if feasible:
        for i in range(N):
            for k in range(K):
                z_sol[i,k] = z[i,k].X
                for j in range(M_plus1):
                    y_sol[i,j,k] = y[i,j,k].X
            for j in range(M):
                for k in range(K):
                    c_sol[i,j,k] = c[i,j,k].X

        return {
            "feasible": True,
            "objective": model.ObjVal,
            "y": y_sol,
            "z": z_sol,
            "c": c_sol,
        }
    else:
        return {"feasible": False}


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
    max_shown = parameters['max_shown']
    epsilon = parameters.get('noise', 0.0)

    N, M_plus1 = weights.shape
    M = M_plus1 - 1

    # For simplicity: single scenario
    K = 20
    weights_list = []
    orderings = []
    for i in range(K):
        # np.random.seed(i)
        weights_list.append(create_random_weights(weights,epsilon))
        orderings.append(np.random.permutation(N))
    
    model = gp.Model("assortment_sequential")
    model.Params.LogToConsole = 0

    # --- Variables ---
    X = model.addVars(N, M, vtype=GRB.BINARY, name="X")            # shared assortment
    y = model.addVars(N, M_plus1, K, vtype=GRB.BINARY, name="y")   # selection per scenario
    z = model.addVars(N, K, vtype=GRB.CONTINUOUS, name="z")        # utility per scenario
    c = model.addVars(N, M, K, vtype=GRB.CONTINUOUS, name="c")     # remaining capacity per timestep

    # --- Constraints ---
    for k in range(K):
        weights_k = weights_list[k]
        ordering = orderings[k]

        # 1. One selection per patient
        for i in range(N):
            model.addConstr(gp.quicksum(y[i,j,k] for j in range(M_plus1)) == 1)

        # 2. Max_shown per patient
        for i in range(N):
            model.addConstr(gp.quicksum(X[i,j] for j in range(M)) <= max_shown)

        # 3. Initialize capacities at timestep 0
        for j in range(M):
            model.addConstr(c[0,j,k] == capacities[j])

        # 5. Cannot pick unavailable providers (offered + available capacity)
        for t in range(N):
            patient = ordering[t]
            for j in range(M):
                if t == 0:
                    model.addConstr(c[t,j,k] == capacities[j])
                else:
                    prev_patient = ordering[t-1]
                    model.addConstr(c[t,j,k] == c[t-1,j,k] - y[prev_patient,j,k])

                # Availability constraint
                model.addConstr(y[patient,j,k] <= X[patient,j])
                model.addConstr(y[patient,j,k] <= c[t,j,k])

        # 6. Link utility
        for i in range(N):
            model.addConstr(z[i,k] == gp.quicksum(weights_k[i,j]*y[i,j,k] for j in range(M_plus1)))

        # 7. Rationality constraints (best among available)
        bigM = 1
        for t in range(N):
            patient = ordering[t]
            for j in range(M_plus1):
                for l in range(M_plus1):
                    if j == l:
                        continue
                    if l < M:
                        # only enforce if l is offered
                        model.addConstr(
                                                        z[patient,k] >= weights_k[patient,l]
                                                                    - bigM*(1 - y[patient,j,k])
                                                                    - bigM*(1 - X[patient,l])
                                                                    - bigM*(1 - c[t,l,k])   # effectively disables if capacity = 0
                                                    )                    
                    else:
                        # exit option always available
                        model.addConstr(z[patient,k] >= weights_k[patient,l] - bigM*(1 - y[patient,j,k]))

    # --- Objective ---
    model.setObjective(gp.quicksum(z[i,k] for i in range(N) for k in range(K)), GRB.MAXIMIZE)

    # --- Solve ---
    model.optimize()

    # --- Extract solution ---
    X_sol = np.zeros((N,M), dtype=int)
    z_sol = np.zeros((N,K))
    y_sol = np.zeros((N,M,K))
    c_sol = np.zeros((N,M,K))
    objective_value = None
    if model.status == GRB.OPTIMAL:
        for i in range(N):
            for j in range(M):
                X_sol[i,j] = int(X[i,j].X)
                for k in range(K):
                    y_sol[i,j,k] = float(y[i,j,k].X)
                    c_sol[i,j,k] = float(c[i,j,k].X)
            for j in range(K):
                z_sol[i,j] = float(z[i,j].X)
        objective_value = model.ObjVal
    print(objective_value/(K*N)) 
    res = simulate_assortment(X_sol, weights_list, capacities=[1]*M, permutations=orderings)
    print(np.mean(res)/20,np.std(res)/20**.5,res)    
    
    return X_sol

def gradient_policy_fast(parameters,fairness_threshold=None,cluster_by_patient=None):
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
    max_shown = parameters['max_shown']
    epsilon = parameters.get('noise', 0.0)

    N, M_plus1 = weights.shape
    M = M_plus1 - 1

    # For simplicity: single scenario
    K = 20
    weights_list = []
    orderings = []
    for i in range(K):
        # np.random.seed(i)
        weights_list.append(create_random_weights(weights,epsilon))
        orderings.append(np.random.permutation(N))
    
    model = gp.Model("assortment_sequential")
    model.Params.LogToConsole = 0

    # --- Variables ---
    X = model.addVars(N, M, vtype=GRB.BINARY, name="X")            # shared assortment
    y = model.addVars(N, M_plus1, K, vtype=GRB.BINARY, name="y")   # selection per scenario
    z = model.addVars(N, K, vtype=GRB.CONTINUOUS, name="z")        # utility per scenario
    c = model.addVars(N, M, K, vtype=GRB.CONTINUOUS, name="c")     # remaining capacity per timestep

    # --- Constraints ---
    for k in range(K):
        weights_k = weights_list[k]
        ordering = orderings[k]

        # 1. One selection per patient
        for i in range(N):
            model.addConstr(gp.quicksum(y[i,j,k] for j in range(M_plus1)) == 1)

        # 2. Max_shown per patient
        for i in range(N):
            model.addConstr(gp.quicksum(X[i,j] for j in range(M)) <= max_shown)

        # 3. Initialize capacities at timestep 0
        for j in range(M):
            model.addConstr(c[0,j,k] == capacities[j])

        # 5. Cannot pick unavailable providers (offered + available capacity)
        for t in range(N):
            patient = ordering[t]
            for j in range(M):
                if t == 0:
                    model.addConstr(c[t,j,k] == capacities[j])
                else:
                    prev_patient = ordering[t-1]
                    model.addConstr(c[t,j,k] == c[t-1,j,k] - y[prev_patient,j,k])

                # Availability constraint
                model.addConstr(y[patient,j,k] <= X[patient,j])
                model.addConstr(y[patient,j,k] <= c[t,j,k])

        # 6. Link utility
        for i in range(N):
            model.addConstr(z[i,k] == gp.quicksum(weights_k[i,j]*y[i,j,k] for j in range(M_plus1)))

        # 7. Rationality constraints (best among available)
        bigM = 1
        for t in range(N):
            patient = ordering[t]
            for j in range(M_plus1):
                for l in range(M_plus1):
                    if j == l:
                        continue
                    if l < M:
                        # only enforce if l is offered
                        model.addConstr(
                                                        z[patient,k] >= weights_k[patient,l]
                                                                    - bigM*(1 - y[patient,j,k])
                                                                    - bigM*(1 - X[patient,l])
                                                                    - bigM*(1 - c[t,l,k])   # effectively disables if capacity = 0
                                                    )                    
                    else:
                        # exit option always available
                        model.addConstr(z[patient,k] >= weights_k[patient,l] - bigM*(1 - y[patient,j,k]))

    # --- Objective ---
    model.setObjective(gp.quicksum(z[i,k] for i in range(N) for k in range(K)), GRB.MAXIMIZE)

    # --- Solve ---
    model.optimize()

    # --- Extract solution ---
    X_sol = np.zeros((N,M), dtype=int)
    z_sol = np.zeros((N,K))
    y_sol = np.zeros((N,M,K))
    c_sol = np.zeros((N,M,K))
    objective_value = None
    if model.status == GRB.OPTIMAL:
        for i in range(N):
            for j in range(M):
                X_sol[i,j] = int(X[i,j].X)
                for k in range(K):
                    y_sol[i,j,k] = float(y[i,j,k].X)
                    c_sol[i,j,k] = float(c[i,j,k].X)
            for j in range(K):
                z_sol[i,j] = float(z[i,j].X)
        objective_value = model.ObjVal
    print(objective_value/(K*N)) 
    res = simulate_assortment(X_sol, weights_list, capacities=[1]*M, permutations=orderings)
    print(np.mean(res)/20,np.std(res)/20**.5,res)    
    
    return X_sol