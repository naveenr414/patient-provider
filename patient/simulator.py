import numpy as np 
import random 
from copy import deepcopy 
import time 
from patient.utils import safe_min, safe_max, safe_var
from patient.ordering_policies import compute_optimal_order
from patient.semi_synthetic import generate_semi_synthetic_theta_workload
import os
import json

class Patient:
    """Class to represent the Patient and their information"""

    def __init__(self,patient_vector,provider_rewards,choice_model_settings,idx,workload):
        self.patient_vector = patient_vector
        self.all_provider_rewards = provider_rewards
        self.provider_rewards = deepcopy(provider_rewards)
        self.choice_model_settings = choice_model_settings
        self.idx = idx 
        self.workload = workload 
    
    def get_random_outcome(self,menu,choice_model):
        """Given a menu, get a random outcome; either match or exit

        Arguments:
            menu: List of providers who are available/presented

        Returns: Integer, -1 is exit
            And other numbers signify a particular provider"""

        if choice_model == "uniform_choice" or choice_model == "threshold":
            provider_probs = [self.all_provider_rewards[i] if menu[i] == 1 else -1 for i in range(len(menu))]
            
            rand_num = np.random.random()
            if rand_num < self.choice_model_settings['true_top_choice_prob'] and np.max(provider_probs) > -1:
                return np.argmax(provider_probs)

            return -1    

        elif choice_model == "mnl":
            provider_probs = [self.all_provider_rewards[i] if menu[i] == 1 else -1 for i in range(len(menu))]
            provider_probs += [self.choice_model_settings['exit_option']]

            probs = [np.exp(i) if i >= 0 else 0 for i in provider_probs]
            probs /= np.sum(probs)

            random_choice = np.random.choice(list(range(len(probs))),p=probs)
            if random_choice == len(self.all_provider_rewards):
                return -1 
            if menu[random_choice] == 0:
                return -1 
            else:
                return random_choice
        elif choice_model == "mnl_max":
            provider_probs = [self.all_provider_rewards[i] if menu[i] == 1 else -1 for i in range(len(menu))]
            if max(provider_probs) > 0:
                provider_probs = [max(provider_probs)]
            
            provider_probs += [self.choice_model_settings['exit_option']]

            probs = [np.exp(i) if i >= 0 else 0 for i in provider_probs]
            probs /= np.sum(probs)

            random_choice = np.random.choice(list(range(len(probs))),p=probs)
            if random_choice == len(self.all_provider_rewards):
                return -1 
            if menu[random_choice] == 0:
                return -1 
            else:
                return random_choice


        else:
            raise Exception("{} choice model not found".format(choice_model))     

class Simulator():
    """Simulator class that allows for evaluation of policies
        Both with and without re-entry"""

    def __init__(self,num_patients,num_providers,provider_capacity,choice_model_settings,choice_model,context_dim,max_menu_size,utility_function,order,num_repetitions,previous_patients_per_provider,num_trials,batch_size,seed):
        self.num_patients = num_patients*num_repetitions
        self.num_providers = num_providers
        self.provider_max_capacity = provider_capacity
        self.choice_model_settings = choice_model_settings
        self.choice_model = choice_model 
        self.context_dim = context_dim 
        self.provider_max_capacities = [provider_capacity for i in range(self.num_providers)]
        self.context_dim = context_dim
        self.max_menu_size = max_menu_size 
        self.utility_function = utility_function
        self.order = order 
        self.num_repetitions = num_repetitions
        self.previous_patients_per_provider = previous_patients_per_provider
        self.num_trials = num_trials
        self.seed = seed
        self.batch_size = batch_size

        self.matched_pairs = []
        self.unmatched_pairs = []
        self.preference_pairs = []
        self.unmatched_patients = []
        self.raw_matched_pairs = []
        self.custom_patient_order = []

    def step(self,patient_num,provider_list):
        """Update the workload by provider, after a patient receives a menu
        
        Arguments:
            patient_num: Integer, which pateint we're looking at
            provider_list: 0-1 List of providers
        
        Returns: List of provider workloads, the available providers, 
            and an integer for the chosen provider"""

        chosen_provider = self.all_patients[patient_num].get_random_outcome(provider_list,self.choice_model)

        if chosen_provider >= 0:
            patient_utility = self.all_patients[patient_num].all_provider_rewards[chosen_provider]
            self.provider_workloads[chosen_provider].append(patient_utility)
            self.provider_capacities[chosen_provider] -= 1
        
        available_providers = [1 if i > 0 else 0 for i in self.provider_capacities]
        self.raw_matched_pairs.append((patient_num,chosen_provider))
        return self.provider_workloads, available_providers, chosen_provider

    def reset_patient_order(self,trial_num):
        if self.order == "random":
            self.patient_order = np.random.permutation(list(range(self.num_patients)))
        elif self.order == "optimal":
            self.patient_order = compute_optimal_order(self)
        elif self.order == "custom":
            if self.custom_patient_order == []:
                self.patient_order = np.random.permutation(list(range(self.num_patients)))
            else:
                self.patient_order = self.custom_patient_order[trial_num]


    def reset_patient_utility(self):
        self.all_patients = []

        if self.utility_function == 'uniform':
            for i in range(self.num_patients):
                patient_vector = np.random.random(self.context_dim)
                utilities = [np.random.random() for j in range(self.num_providers)]       
                workload = np.random.random()     
                self.all_patients.append(Patient(patient_vector,utilities,self.choice_model_settings,i,workload))
        elif self.utility_function == 'normal':
            means = [np.random.random() for i in range(self.num_providers)]
            for i in range(self.num_patients):
                patient_vector = np.random.random(self.context_dim)
                utilities = [np.clip(np.random.normal(means[j],0.1),0,1) for j in range(self.num_providers)]     
                workload = np.random.random()       
                self.all_patients.append(Patient(patient_vector,utilities,self.choice_model_settings,i,workload))
        elif self.utility_function == 'fixed':
            a = 0.4
            rewards = np.array([[1,1,1,1],[a,a,a,a],[a,a,a,a],[a,a,a,a]])
            for i in range(self.num_patients):
                patient_vector = np.random.random(self.context_dim)
                utilities = rewards[i]          
                self.all_patients.append(Patient(patient_vector,utilities,self.choice_model_settings,i))
        elif self.utility_function == 'semi_synthetic':
            if os.path.exists("../../data/{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers)):
                data = json.load(open("../../data/{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers)))
                theta = np.array(data[0]) 
                workloads = np.array(data[1])
            else:
                theta, workloads,random_patients, random_providers = generate_semi_synthetic_theta_workload(self.num_patients,self.num_providers)
                data = [theta.tolist(),workloads.tolist()]
                json.dump(data,open("../../data/{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers),"w"))
                json.dump(random_patients,open("../../data/patient_data_{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers),"w"))
                json.dump(random_providers,open("../../data/provider_data_{}_{}_{}.json".format(self.seed,self.num_patients,self.num_providers),"w"))
            for i in range(self.num_patients):
                patient_vector = np.random.random(self.context_dim)
                utilities = theta[i]  
                workload = workloads[i] 
                self.all_patients.append(Patient(patient_vector,utilities,self.choice_model_settings,i,workload))
        else:
            raise Exception("Utility Function {} not found".format(self.utility_function))
        
        if self.choice_model == "threshold":
            threshold = self.all_patients[0].choice_model_settings['exit_option']
            for i in range(len(self.all_patients)):
                self.all_patients[i].all_provider_rewards = [int(j>threshold)*j for j in self.all_patients[i].all_provider_rewards]
        
        self.patients = self.all_patients[:self.num_patients//self.num_repetitions]

    def reset_initial(self):
        self.provider_capacities = deepcopy(self.provider_max_capacities)
        self.provider_workloads = [[] for i in range(self.num_providers)]
        self.provider_previous_workloads = []
        for i in range(self.num_providers):
            workloads = np.sum([np.random.random() for i in range(self.previous_patients_per_provider)])
            self.provider_previous_workloads.append(workloads)
        self.provider_vectors = [np.random.random(self.context_dim) for i in range(self.num_providers)]
        self.utility_coefficients = np.random.random(self.context_dim)

def run_heterogenous_policy(env,policy,seed,num_trials,per_epoch_function=None,second_seed=None):
    """Wrapper to run policies without needing to go through boilerplate code
    
    Arguments:
        env: Simulator environment
        n_episodes: How many episodes to run for each epoch
            T = n_episodes * episode_len
        n_epochs: Number of different epochs/cohorts to run
        discount: Float, how much to discount rewards by
        policy: Function that takes in environment, state, budget, and lambda
            produces action as a result
        seed: Random seed for run
        lamb: Float, tradeoff between matching, activity
        should_train: Should we train the policy; if so, run epochs to train first
    
    Returns: Two things
        matching reward - Numpy array of Epochs x T, with rewards for each combo (this is R_{glob})
        activity rate - Average rate of engagement across all volunteers (this is R_{i}(s_{i},a_{i}))
    """
    N         = env.num_patients
    M         = env.num_providers   
    T         = env.num_patients

    random.seed(seed)
    np.random.seed(seed)
    env.reset_initial()
    env.reset_patient_utility()

    all_trials_workload = []
    initial_workloads = [deepcopy(env.provider_previous_workloads) for _ in range(num_trials)]
    final_workloads = [deepcopy(env.provider_previous_workloads) for _ in range(num_trials)]
    patient_results = []
    time_taken = 0

    if second_seed != None:
        random.seed(second_seed)
        np.random.seed(second_seed)
    
    per_epoch_results = None 
    if per_epoch_function != None and env.num_repetitions == 1:
        per_epoch_results = per_epoch_function(env)

    top_b = np.zeros((env.num_patients,env.num_providers))
    num_times_in_position = np.zeros((env.num_patients,env.num_patients))
    matches_per = np.zeros((env.num_patients,env.num_providers))

    for trial_num in range(num_trials):
        env.num_providers = M 
        env.num_patients = N
        env.reset_patient_order(trial_num)
        env.reset_initial()
        start = time.time()
        available_providers  = [1 for i in range(len(env.provider_capacities))]
        matched_pairs = []
        preference_pairs = []
        unmatched_patients = []
        patient_results_trial = []

        for repetition in range(env.num_repetitions):
            memory = None 
            env.patients = env.all_patients[repetition*env.num_patients//env.num_repetitions:(repetition+1)*env.num_patients//env.num_repetitions]
            env.num_providers = sum(available_providers)
            idx_to_provider_num = []
            for i in range(len(available_providers)):
                if available_providers[i] == 1:
                    idx_to_provider_num.append(i)
            
            if env.num_providers == 0:
                break 

            for i in range(len(env.patients)):
                env.patients[i].provider_rewards = np.array(env.patients[i].all_provider_rewards)[np.array(available_providers) == 1]

            if per_epoch_function != None and env.num_repetitions > 1:
                per_epoch_results = per_epoch_function(env)

            current_patients_sorted = sorted([env.all_patients[env.patient_order[t]].idx for t in range(repetition*env.num_patients//env.num_repetitions,(repetition+1)*env.num_patients//env.num_repetitions)])
            for t in range(repetition*env.num_patients//env.num_repetitions,(repetition+1)*env.num_patients//env.num_repetitions):
                current_patient = env.all_patients[env.patient_order[t]]
                current_patient.idx = current_patients_sorted.index(current_patient.idx)
                num_times_in_position[current_patient.idx][t] += 1
                selected_providers,memory = policy(env,current_patient,available_providers,memory,deepcopy(per_epoch_results))

                selected_provider_to_all = np.zeros(len(available_providers))
                selected_provider_to_all[idx_to_provider_num] = selected_providers


                selected_provider_to_all *= np.array(available_providers)
                if np.sum(selected_provider_to_all) > env.max_menu_size:
                    indices = np.where(selected_provider_to_all == 1)[0]  # Get indices of elements that are 1
                    to_set_zero = indices[env.max_menu_size:]
                    selected_provider_to_all[to_set_zero] = 0 
                provider_workloads, available_providers,chosen_provider = env.step(env.patient_order[t],selected_provider_to_all)

                top_b[current_patient.idx][np.argmax(np.array(available_providers)*np.array(current_patient.provider_rewards))] += 1

                if np.sum(selected_provider_to_all) == 0:
                    patient_results_trial.append(-0.01)
                elif chosen_provider >= 0:
                    patient_results_trial.append(current_patient.provider_rewards[chosen_provider])
                    matches_per[current_patient.idx][chosen_provider] += 1/num_trials
                                        
                if chosen_provider >= 0:
                    matched_pairs.append((current_patient.patient_vector,env.provider_vectors[chosen_provider]))
                    final_workloads[trial_num][chosen_provider] += current_patient.workload
                    for i in range(len(selected_provider_to_all)):
                        if selected_provider_to_all[i] == 1 and i != chosen_provider:
                            preference_pairs.append((current_patient.patient_vector,env.provider_vectors[chosen_provider],env.provider_vectors[i]))
                else:
                    unmatched_patients.append(env.patient_order[t])
                env.unmatched_patients = unmatched_patients
        time_taken += time.time()-start 
        all_trials_workload.append(deepcopy(provider_workloads))
        patient_results.append(deepcopy(patient_results_trial))
    
    print("Took {} time".format(time_taken))

    return all_trials_workload, patient_results, initial_workloads, final_workloads, matches_per


def run_multi_seed(seed_list,policy,parameters,per_epoch_function=None):
    """Run multiple seeds of trials for a given policy
    
    Arguments:
        seed_list: List of integers, which seeds to run for
        policy: Function which maps (RMABSimulator,state vector,budget,lamb,memory,per_epoch_results) to actions and memory
            See baseline_policies.py for examples
        parameters: Dictionary with keys for each parameter
        should_train: Boolean; if we're training, then we run the policy for the training period
            Otherwise, we just skip these + run only fo the testing period
    
    Returns: 3 things, the scores dictionary with reward, etc. 
        memories: List of memories (things that the policy might store between timesteps)
        simulator: RMABSimulator object
    """
    
    scores = {
        'matches': [],
        'patient_utilities': [], 
        'provider_minimums': [],
        'provider_minimums_all': [],
        'provider_gaps': [],
        'provider_gaps_all': [],
        'provider_variance': [],
        'provider_variance_all': [],
        'provider_workloads': [], 
        'initial_workloads': [], 
        'final_workloads': [],
        'matches_per': [],
    }

    num_patients = parameters['num_patients']
    num_providers = parameters['num_providers']
    provider_capacity = parameters['provider_capacity']
    top_choice_prob = parameters['top_choice_prob']
    choice_model = parameters['choice_model']
    exit_option = parameters['exit_option']
    num_trials = parameters['num_trials']
    context_dim = parameters['context_dim']
    true_top_choice_prob = parameters['true_top_choice_prob']
    num_repetitions = parameters['num_repetitions']
    context_dim = parameters['context_dim']
    max_menu_size = parameters['max_menu_size']
    utility_function = parameters['utility_function']
    order = parameters['order']
    previous_patients_per_provider = parameters['previous_patients_per_provider']
    batch_size = parameters['batch_size']

    choice_model_settings = {
        'top_choice_prob': top_choice_prob,
        'true_top_choice_prob': true_top_choice_prob, 
        'exit_option': exit_option
    }

    for seed in seed_list:
        simulator = Simulator(num_patients,num_providers,provider_capacity,choice_model_settings,choice_model,context_dim,max_menu_size,utility_function,order,num_repetitions,previous_patients_per_provider,num_trials,batch_size,seed)

        policy_results, patient_results, initial_workloads, final_workloads, matches_per = run_heterogenous_policy(simulator,policy,seed,num_trials,per_epoch_function=per_epoch_function,second_seed=seed) 
        utilities_by_provider = policy_results

        num_matches = [len([j for j in i if j != []]) for i in utilities_by_provider]
        patient_utilities = [sum([np.sum(j) if len(j)>0 else 0 for j in i]) for i in utilities_by_provider]
        
        list_of_utilities = [[j for j in i if j>=0] for i in patient_results]
        list_of_utilities_all = [[max(j,0) for j in i] for i in patient_results]
        min_utilities = [safe_min(i) for i in list_of_utilities]
        min_utilities_all = [safe_min(i) for i in list_of_utilities_all]
        max_gap = [safe_max(i)-safe_min(i) for i in list_of_utilities]
        max_gap_all = [safe_max(i)-safe_min(i)for i in list_of_utilities_all]

        variance = [safe_var(i) for i in list_of_utilities]
        variance_all = [safe_var(i) for i in list_of_utilities_all]

        if len(np.array(utilities_by_provider).shape) == 3:
            provider_workloads = [[len(j) for j in i] for i in utilities_by_provider]
            provider_workloads = np.mean(np.array(provider_workloads),axis=0).tolist()
        else:
            provider_workloads = [sum([len(j) for j in i])/len(i) for i in np.array(utilities_by_provider).T]

        scores['initial_workloads'].append(initial_workloads)
        scores['final_workloads'].append(final_workloads)
        scores['matches'].append(num_matches)
        scores['patient_utilities'].append(patient_utilities)
        scores['provider_workloads'].append(provider_workloads)
        scores['matches_per'].append(matches_per.tolist())

        scores['provider_minimums'].append(min_utilities)
        scores['provider_minimums_all'].append(min_utilities_all)

        scores['provider_gaps'].append(max_gap)
        scores['provider_gaps_all'].append(max_gap_all)

        scores['provider_variance'].append(variance)
        scores['provider_variance_all'].append(variance_all)


    return scores, simulator

