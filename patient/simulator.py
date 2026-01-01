import numpy as np 
import random 
from copy import deepcopy 
import time 
from patient.utils import safe_min, safe_max, safe_var, one_shot_policy
from patient.ordering_policies import compute_optimal_order
from patient.semi_synthetic import generate_semi_synthetic_theta_workload
import os
import json


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

class Patient:
    """Class to represent the Patient and their information"""

    def __init__(self, provider_rewards, idx):
        self.provider_rewards = provider_rewards
        self.idx = idx 
    
    def get_random_outcome(self, menu):
        """Given a menu, get a random outcome; either match or exit

        Arguments:
            menu: List of providers who are available/presented

        Returns: Integer, -1 is exit
            And other numbers signify a particular provider"""

        provider_probs = np.where(menu == 1, self.provider_rewards, -1)
        max_loc = np.argmax(provider_probs)
        return max_loc 

class Simulator():
    """Simulator class that allows for evaluation of policies
        Both with and without re-entry"""

    def __init__(self,num_patients,num_providers,provider_capacity,num_trials,utility_function,order,noise,online_arrival,new_provider,average_distance,max_shown,online_scale,seed):
        self.num_patients = num_patients
        self.num_providers = num_providers
        self.provider_max_capacity = provider_capacity
        self.provider_max_capacities = [provider_capacity for i in range(self.num_providers+1)]
        self.provider_max_capacities[-1] = self.num_patients
        self.utility_function = utility_function
        self.order = order 
        self.num_trials = num_trials
        self.seed = seed
        self.noise = noise 
        self.online_arrival = online_arrival
        self.new_provider = new_provider
        self.average_distance = average_distance
        self.max_shown = max_shown 
        self.online_scale = online_scale

    def step(self,patient_num,provider_list):
        """Update the workload by provider, after a patient receives a menu
        
        Arguments:
            patient_num: Integer, which pateint we're looking at
            provider_list: 0-1 List of providers
        
        Returns: List of provider workloads, the available providers, 
            and an integer for the chosen provider"""

        chosen_provider = self.all_patients[patient_num].get_random_outcome(provider_list)
        self.provider_capacities[chosen_provider] -= 1

        return chosen_provider

    def reset_patient_order(self,trial_num):
        if self.order == "uniform":
            self.patient_order = np.random.permutation(list(range(self.num_patients)))
        elif self.order == "custom":
            if self.custom_patient_order == []:
                self.patient_order = np.random.permutation(list(range(self.num_patients)))
            else:
                self.patient_order = self.custom_patient_order[trial_num]
        elif self.order == "proportional":
            max_rewards = [np.mean(i.provider_rewards) for i in self.all_patients]
            max_rewards = np.array(max_rewards)/np.sum(max_rewards)
            self.patient_order = np.random.choice(list(range(len(max_rewards))), size=len(max_rewards), replace=False, p=max_rewards)

    def reset_patient_utility(self):
        self.all_patients = []

        if self.utility_function == 'uniform':
            for i in range(self.num_patients):
                utilities = [np.random.random() for j in range(self.num_providers+1)]       
                self.all_patients.append(Patient(utilities,i))
        elif self.utility_function == 'normal':
            means = [np.random.random() for i in range(self.num_providers+1)]
            for i in range(self.num_patients):
                std = 0.1
                utilities = [np.clip(np.random.normal(means[j],std),0,1) for j in range(self.num_providers+1)]     
                self.all_patients.append(Patient(utilities,i))
        elif self.utility_function == 'semi_synthetic':
            if os.path.exists("../../data/{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers)) and self.average_distance == 20.2:
                data = json.load(open("../../data/{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers)))
                theta = np.array(data[0]) 
                workloads = np.array(data[1])
            else:
                print("Generating dataset!")
                theta, workloads,random_patients, random_providers = generate_semi_synthetic_theta_workload(self.num_patients,self.num_providers,average_distance=self.average_distance)
                data = [theta.tolist(),workloads.tolist()]
                json.dump(data,open("../../data/{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers),"w"))
                json.dump(random_patients,open("../../data/patient_data_{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers),"w"))
                json.dump(random_providers,open("../../data/provider_data_{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers),"w"))
                print("Wrote dataset!")
            theta = np.hstack([theta, np.full((theta.shape[0], 1), 0.25)])
            for i in range(self.num_patients):
                utilities = theta[i]  
                self.all_patients.append(Patient(utilities,i))
        elif self.utility_function == 'semi_synthetic_comorbidity':
            if os.path.exists("../../data/{}_{}_{}_comorbidity.json".format(self.seed,self.num_patients,self.num_providers)) and self.average_distance == 20.2:
                data = json.load(open("../../data/{}_{}_{}_comorbidity.json".format(self.seed,self.num_patients,self.num_providers)))
                theta = np.array(data[0]) 
                workloads = np.array(data[1])
            else:
                theta, workloads,random_patients, random_providers = generate_semi_synthetic_theta_workload(self.num_patients,self.num_providers,comorbidities=True,average_distance=self.average_distance)
                data = [theta.tolist(),workloads.tolist()]
                json.dump(data,open("../../data/{}_{}_{}_comorbidity.json".format(self.seed,self.num_patients,self.num_providers),"w"))
                json.dump(random_patients,open("../../data/patient_data_{}_{}_{}_comorbidity.json".format(self.seed,self.num_patients,self.num_providers),"w"))
                json.dump(random_providers,open("../../data/provider_data_{}_{}_{}_comorbidity.json".format(self.seed,self.num_patients,self.num_providers),"w"))
            theta = np.hstack([theta, np.full((theta.shape[0], 1), 0.25)])
            for i in range(self.num_patients):
                utilities = theta[i]  
                self.all_patients.append(Patient(utilities,i))
        else:
            raise Exception("Utility Function {} not found".format(self.utility_function))

    def reset_initial(self):
        self.provider_capacities = deepcopy(self.provider_max_capacities)

def get_target_plus_random(row,scale_ours):
    scale_others = 1/scale_ours

    row = deepcopy(row)
    row[0][:-1]*=scale_ours 
    row[0][-1] *= scale_others
    return row 


def run_heterogenous_policy(env, policy, seed, num_trials, per_epoch_function=None,use_real=False):
    """Wrapper to run policies without needing to go through boilerplate code
    
    Arguments:
        env: Simulator environment
        policy: Function that takes in environment, state, budget, and lambda
            produces action as a result
        seed: Random seed for run
        num_trials: Number of trials to run
        per_epoch_function: Optional function to call per epoch
    
    Returns: 
        patient_results: List of lists, where each sublist contains tuples of 
                        (chosen_provider, reward) for each patient in that trial
    """
    N = env.num_patients
    M = env.num_providers   
    online_arrival = env.online_arrival
    online_scale = env.online_scale

    random.seed(seed)
    np.random.seed(seed)
    env.reset_initial()
    env.reset_patient_utility()

    patient_results = []
    patient_orders = []
    patient_options = []
    time_taken = 0
    
    per_epoch_results = None 
    weights_noisy = [p.provider_rewards for p in env.all_patients]
    weights_noisy = np.array(weights_noisy)

    if per_epoch_function is not None and not online_arrival and not use_real:
        parameters = {'weights': weights_noisy, 
                        'capacities': env.provider_capacities, 
                        'online_arrival': env.online_arrival,
                        'max_shown': env.max_shown,
                        'noise': env.noise}
        per_epoch_results = per_epoch_function(parameters)
    
    new_provider_mode = env.new_provider
    new_providers = []
    new_provider_idx = np.random.randint(M)
    all_weights = []

    for trial_num in range(num_trials):
        if online_arrival:
            print("On trial {}".format(trial_num))
        env.num_providers = M 
        env.num_patients = N
        env.reset_patient_order(trial_num)
        env.reset_initial()
        noise = env.noise 
        np.random.seed(trial_num)
        weights = create_random_weights(weights_noisy,noise)
        env.patient_order = np.random.permutation(N)
        weights = np.clip(weights,0,1)
        all_weights.append(weights.tolist())

        for i in range(len(env.all_patients)):
            env.all_patients[i].provider_rewards = weights[i]
        if use_real and per_epoch_function is not None and not online_arrival :
            parameters = {'weights': weights, 
                          'capacities': env.provider_capacities, 
                          'online_arrival': env.online_arrival, 
                          'max_shown': env.max_shown,
                          'noise': env.noise}
            per_epoch_results = per_epoch_function(parameters)

        
        start = time.time()
        
        unmatched_patients = []
        patient_results_trial = []
        patient_options_trial = [[] for i in range(N)]

        memory = None 
                
        # If so, randomly hide one provider for the first phase
        if new_provider_mode:
            env.provider_capacities[new_provider_idx] = 0
            new_providers.append(new_provider_idx)
        else:
            new_providers.append(-1)
        available_providers = (np.array(env.provider_capacities) > 0).astype(int)

        patient_results_trial = [() for i in range(env.num_patients)]
        for t in range(env.num_patients):
            current_patient = env.all_patients[env.patient_order[t]]
            R = 0
            # Get policy decision
            if online_arrival:
                if use_real:
                    selected_providers = per_epoch_function({'weights': weights[env.patient_order[t:t+1]], 
                                        'max_capacity': env.provider_max_capacity, 
                                        'online_arrival': env.online_arrival, 
                                        'capacities': env.provider_capacities,
                                        'noise': env.noise,
                                        'max_shown': env.max_shown})[0]
                else:
                    selected_providers = per_epoch_function({'weights': get_target_plus_random(weights[env.patient_order[t:t+1]], online_scale),
                                                            'max_capacity': env.provider_max_capacity, 
                                                            'online_arrival': env.online_arrival, 
                                                            'capacities': env.provider_capacities,
                                                            'noise': env.noise,
                                                            'max_shown': env.max_shown})[0]
            elif policy == one_shot_policy:
                selected_providers = per_epoch_results[current_patient.idx]
            else:
                selected_providers, memory = policy(env, current_patient, available_providers, 
                                                    memory, per_epoch_results)

            initial_selected = np.concatenate([selected_providers, [1]])
            total_ones = np.sum(initial_selected)
            if total_ones > env.max_shown+1:
                first_N = initial_selected[:-1]
                one_indices = np.flatnonzero(first_N)

                keep_indices = np.random.choice(one_indices, size=env.max_shown, replace=False)
                first_N[:] = 0
                first_N[keep_indices] = 1
                initial_selected = np.concatenate([first_N, [1]])

            # Apply menu size constraint if needed
            selected_provider_to_all = np.multiply(initial_selected, available_providers)
            patient_options_trial[env.patient_order[t]] = [i for i in range(len(selected_provider_to_all)) if selected_provider_to_all[i] == 1]

            # Execute step
            chosen_provider = env.step(env.patient_order[t], selected_provider_to_all)
            # Record results
            reward = current_patient.provider_rewards[chosen_provider]

            if new_provider_mode:
                patient_results_trial[env.patient_order[t]] = (chosen_provider, chosen_provider,reward)
            else:
                patient_results_trial[env.patient_order[t]] = (chosen_provider, reward)
            
            # Update availability if capacity reached
            if env.provider_capacities[chosen_provider] == 0:
                available_providers[chosen_provider] = 0
            env.unmatched_patients = unmatched_patients
        if new_provider_mode:
            env.provider_capacities[new_provider_idx] = available_providers[new_provider_idx] = int(round(np.sqrt(N)))
            new_weights = np.zeros((N,3))
            for idx,i in enumerate(patient_results_trial):
                new_weights[idx,0] = i[2]
            new_weights[:,1] = weights[:,new_provider_idx]

            new_capacities = [N,env.provider_capacities[new_provider_idx]]

            if use_real:
                parameters = {'weights': new_weights, 'capacities': new_capacities, 'online_arrival': env.online_arrival, 'max_shown': env.max_shown, 'noise': noise}
            else:
                parameters = {'weights': create_random_weights(new_weights,noise),
                            'capacities': new_capacities, 'online_arrival': env.online_arrival, 'max_shown': env.max_shown, 'noise': noise}
            
            single_provider_results = per_epoch_function(parameters)

            env.reset_patient_order(trial_num+num_trials)
            for t in range(env.num_patients):
                current_patient = env.all_patients[env.patient_order[t]]
                selected_providers = single_provider_results[current_patient.idx]
                assert len(selected_providers) == 2, "Expected 2 providers in new-provider stage"

                selected_provider_to_all = np.array(selected_providers) * np.array([1, available_providers[new_provider_idx]])
                current_reward = new_weights[env.patient_order[t], 0]


                # Record results
                if current_patient.provider_rewards[new_provider_idx] > current_reward and available_providers[new_provider_idx] > 0:
                    reward = current_patient.provider_rewards[new_provider_idx]

                    old_provider = patient_results_trial[env.patient_order[t]][0]
                    patient_results_trial[env.patient_order[t]] = (old_provider,np.int64(new_provider_idx),current_reward,reward)
                    env.provider_capacities[new_provider_idx] -= 1
                    available_providers[new_provider_idx] -= 1

                    # Update availability if capacity reached
                    if env.provider_capacities[new_provider_idx] == 0:
                        available_providers[new_provider_idx] = 0
        time_taken += time.time() - start 
        patient_results.append(patient_results_trial)
        patient_orders.append(env.patient_order.tolist())
        patient_options.append(patient_options_trial)

    return patient_results, patient_orders, new_providers, per_epoch_results, patient_options, all_weights



def run_multi_seed(seed_list,policy,parameters,per_epoch_function=None,use_real=False):
    """Run multiple seeds of trials for a given policy
    
    Arguments:
        seed_list: List of integers, which seeds to run for
        policy: Function which maps (RMABSimulator,state vector,budget,lamb,memory,per_epoch_results) to actions and memory
            See baseline_policies.py for examples
        parameters: Dictionary with keys for each parameter
    
    Returns: 3 things, the scores dictionary with reward, etc. 
        memories: List of memories (things that the policy might store between timesteps)
        simulator: RMABSimulator object
    """
    
    scores = {
        'patient_utilities': [],
        'chosen_providers': [], 
        'num_matches': [],
        'assortments': [],
        'patient_orders': [],
        'new_providers': [],
        'options': [],
        'weights': [],
    }

    num_patients = parameters['num_patients']
    num_providers = parameters['num_providers']
    provider_capacity = parameters['provider_capacity']
    utility_function = parameters['utility_function']
    order = parameters['order']
    num_trials = parameters['num_trials']
    noise = parameters['noise']
    online_arrival = parameters['online_arrival']
    new_provider = parameters['new_provider']
    average_distance = parameters['average_distance']
    max_shown = parameters['max_shown']
    online_scale = parameters['online_scale']
    verbose = parameters['verbose']

    for seed in seed_list:
        simulator = Simulator(num_patients,num_providers,provider_capacity,num_trials,utility_function,order,noise,online_arrival,new_provider,average_distance,max_shown,online_scale,seed)
        patient_results, patient_orders, new_providers, assortment, options, weights = run_heterogenous_policy(simulator,policy,seed,num_trials,per_epoch_function=per_epoch_function,use_real=use_real) 

        if new_provider:
            def double_tuple(t):
                if len(t) == 1:
                    return (t[0],t[0])
                return t
            patient_utilities = [[double_tuple(j[2:]) for j in i] for i in patient_results]
            chosen_providers = [[(int(j[0]),int(j[1])) for j in i] for i in patient_results]
        else:
            patient_utilities = [[j[1] for j in i] for i in patient_results]
            chosen_providers = [[int(j[0]) for j in i] for i in patient_results]

        num_matches = [len([j for j in i if j[0] < num_providers]) for i in patient_results]
        
        scores['patient_utilities'].append(patient_utilities)
        scores['assortments'].append(np.array(assortment).tolist())
        scores['chosen_providers'].append(chosen_providers)
        scores['num_matches'].append(num_matches)
        scores['patient_orders'].append(patient_orders)
        scores['new_providers'].append(new_providers)

        if verbose:
            scores['options'].append(options)
            scores['weights'].append(weights)

    return scores, simulator

