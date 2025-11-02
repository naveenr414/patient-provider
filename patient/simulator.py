import numpy as np 
import random 
from copy import deepcopy 
import time 
from patient.utils import safe_min, safe_max, safe_var, one_shot_policy
from patient.ordering_policies import compute_optimal_order
from patient.semi_synthetic import generate_semi_synthetic_theta_workload
import os
import json

class Patient:
    """Class to represent the Patient and their information"""

    def __init__(self, provider_rewards, idx):
        self.provider_rewards = provider_rewards
        self.all_provider_rewards = provider_rewards
        self.idx = idx 
    
    def get_random_outcome(self, menu):
        """Given a menu, get a random outcome; either match or exit

        Arguments:
            menu: List of providers who are available/presented

        Returns: Integer, -1 is exit
            And other numbers signify a particular provider"""

        provider_probs = np.where(menu == 1, self.all_provider_rewards, -1)
        max_loc = np.argmax(provider_probs)
        return max_loc 

class Simulator():
    """Simulator class that allows for evaluation of policies
        Both with and without re-entry"""

    def __init__(self,num_patients,num_providers,provider_capacity,num_trials,utility_function,order,noise,seed):
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
            max_rewards = [np.mean(i.all_provider_rewards) for i in self.all_patients]
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
            if os.path.exists("../../data/{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers)):
                data = json.load(open("../../data/{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers)))
                theta = np.array(data[0]) 
                workloads = np.array(data[1])
            else:
                print("Generating dataset!")
                theta, workloads,random_patients, random_providers = generate_semi_synthetic_theta_workload(self.num_patients,self.num_providers)
                data = [theta.tolist(),workloads.tolist()]
                json.dump(data,open("../../data/{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers),"w"))
                json.dump(random_patients,open("../../data/patient_data_{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers),"w"))
                json.dump(random_providers,open("../../data/provider_data_{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers),"w"))
                print("Wrote dataset!")
            for i in range(self.num_patients):
                utilities = theta[i]  
                self.all_patients.append(Patient(utilities,i))
        elif self.utility_function == 'semi_synthetic_comorbidity':
            if os.path.exists("../../data/{}_{}_{}_comorbidity.json".format(self.seed,self.num_patients,self.num_providers)):
                data = json.load(open("../../data/{}_{}_{}_comorbidity.json".format(self.seed,self.num_patients,self.num_providers)))
                theta = np.array(data[0]) 
                workloads = np.array(data[1])
            else:
                theta, workloads,random_patients, random_providers = generate_semi_synthetic_theta_workload(self.num_patients,self.num_providers,comorbidities=True)
                data = [theta.tolist(),workloads.tolist()]
                json.dump(data,open("../../data/{}_{}_{}_comorbidity.json".format(self.seed,self.num_patients,self.num_providers),"w"))
                json.dump(random_patients,open("../../data/patient_data_{}_{}_{}_comorbidity.json".format(self.seed,self.num_patients,self.num_providers),"w"))
                json.dump(random_providers,open("../../data/provider_data_{}_{}_{}_comorbidity.json".format(self.seed,self.num_patients,self.num_providers),"w"))
            for i in range(self.num_patients):
                utilities = theta[i]  
                self.all_patients.append(Patient(utilities,i))
        else:
            raise Exception("Utility Function {} not found".format(self.utility_function))

    def reset_initial(self):
        self.provider_capacities = deepcopy(self.provider_max_capacities)

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

    random.seed(seed)
    np.random.seed(seed)
    env.reset_initial()
    env.reset_patient_utility()

    patient_results = []
    time_taken = 0
    
    per_epoch_results = None 
    weights = [p.provider_rewards for p in env.all_patients]
    weights = np.array(weights)

    if per_epoch_function is not None:
        if use_real:
            per_epoch_results = per_epoch_function(weights,env.provider_max_capacity)
        else:
            noise = env.noise 
            weight_noisy = weights+np.random.normal(0,noise,weights.shape)
            per_epoch_results = per_epoch_function(weight_noisy,env.provider_max_capacity)


    for trial_num in range(num_trials):
        env.num_providers = M 
        env.num_patients = N
        env.reset_patient_order(trial_num)
        env.reset_initial()
        start = time.time()
        
        available_providers = np.array([1 for i in range(len(env.provider_capacities))])
        unmatched_patients = []
        patient_results_trial = []

        memory = None 
        
        for t in range(env.num_patients):
            current_patient = env.all_patients[env.patient_order[t]]
            
            # Get policy decision
            if policy == one_shot_policy:
                selected_providers = per_epoch_results[current_patient.idx]
            else:
                selected_providers, memory = policy(env, current_patient, available_providers, 
                                                    memory, per_epoch_results)

            # Apply menu size constraint if needed
            selected_provider_to_all = np.multiply(np.concatenate([selected_providers,[1]]), available_providers)
            
            # Execute step
            chosen_provider = env.step(env.patient_order[t], selected_provider_to_all)

            # Record results
            if np.sum(selected_provider_to_all) == 0:
                # No providers available - patient unmatched
                patient_results_trial.append((chosen_provider, -0.01))
                unmatched_patients.append(env.patient_order[t])
            elif chosen_provider >= 0:
                # Successful match
                reward = current_patient.provider_rewards[chosen_provider]
                patient_results_trial.append((chosen_provider, reward))
                
                # Update availability if capacity reached
                if env.provider_capacities[chosen_provider] == 0:
                    available_providers[chosen_provider] = 0
            else:
                # Patient chose to exit
                patient_results_trial.append((chosen_provider, 0))
                unmatched_patients.append(env.patient_order[t])
            
            env.unmatched_patients = unmatched_patients

        time_taken += time.time() - start 
        patient_results.append(patient_results_trial)
    

    return patient_results, per_epoch_results



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
    }

    num_patients = parameters['num_patients']
    num_providers = parameters['num_providers']
    provider_capacity = parameters['provider_capacity']
    utility_function = parameters['utility_function']
    order = parameters['order']
    num_trials = parameters['num_trials']
    noise = parameters['noise']

    for seed in seed_list:
        simulator = Simulator(num_patients,num_providers,provider_capacity,num_trials,utility_function,order,noise,seed)
        patient_results, assortment = run_heterogenous_policy(simulator,policy,seed,num_trials,per_epoch_function=per_epoch_function,use_real=use_real) 

        patient_utilities = [[j[1] for j in i] for i in patient_results]
        chosen_providers = [[int(j[0]) for j in i] for i in patient_results]
        num_matches = [len([j for j in i if j[0] < num_providers]) for i in patient_results]
        
        scores['patient_utilities'].append(patient_utilities)
        scores['assortments'].append(np.array(assortment).tolist())
        scores['chosen_providers'].append(chosen_providers)
        scores['num_matches'].append(num_matches)

    return scores, simulator

