import numpy as np 
import random 
from copy import deepcopy 
import time 
from patient.utils import compute_utility

class Patient:
    """Class to represent the Patient and their information"""

    def __init__(self,patient_vector,provider_rewards,choice_model_settings,idx):
        self.patient_vector = patient_vector
        self.provider_rewards = provider_rewards
        self.choice_model_settings = choice_model_settings
        self.idx = idx 
    
    def get_random_outcome(self,menu,choice_model):
        """Given a menu, get a random outcome; either match or exit

        Arguments:
            menu: List of providers who are available/presented

        Returns: Integer, -1 is exit
            And other numbers signify a particular provider"""

        if choice_model == "uniform_choice":
            provider_probs = [self.provider_rewards[i] if menu[i] == 1 else -1 for i in range(len(menu))]
            
            rand_num = np.random.random()

            if rand_num < self.choice_model_settings['top_choice_prob'] and np.max(provider_probs) > -1:
                return np.argmax(provider_probs)

            return -1    
        elif choice_model == "mnl":
            provider_probs = [self.provider_rewards[i] if menu[i] == 1 else -1 for i in range(len(menu))]
            provider_probs += [self.choice_model_settings['exit_option']]

            probs = [np.exp(i) if i >= 0 else 0 for i in provider_probs]
            probs /= np.sum(probs)

            random_choice = np.random.choice(list(range(len(probs))),p=probs)
            if random_choice == len(self.provider_rewards):
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

    def __init__(self,num_patients,num_providers,provider_capacity,choice_model_settings,choice_model,context_dim):
        self.num_patients = num_patients 
        self.num_providers = num_providers
        self.provider_max_capacity = provider_capacity
        self.choice_model_settings = choice_model_settings
        self.choice_model = choice_model 
        self.context_dim = context_dim 
        self.provider_max_capacities = [provider_capacity for i in range(self.num_providers)]
        self.context_dim = context_dim 

        self.matched_pairs = []
        self.unmatched_pairs = []
        self.preference_pairs = []
        self.unmatched_patients = []
        self.raw_matched_pairs = []

    def step(self,patient_num,provider_list):
        chosen_provider = self.patients[patient_num].get_random_outcome(provider_list,self.choice_model)
        if chosen_provider >= 0:
            patient_utility = self.patients[patient_num].provider_rewards[chosen_provider]
            self.provider_workloads[chosen_provider].append(patient_utility)
            self.provider_capacities[chosen_provider] -= 1
        
        available_providers = [1 if i > 0 else 0 for i in self.provider_capacities]
        self.raw_matched_pairs.append((patient_num,chosen_provider))
        return self.provider_workloads, available_providers, chosen_provider

    def reset_patients(self):
        self.patient_order = np.random.permutation(list(range(self.num_patients)))
        self.patients = []
        for i in range(self.num_patients):
            patient_vector = np.random.random(self.context_dim)
            utilities = [compute_utility(patient_vector,self.provider_vectors[j],self.utility_coefficients,self.context_dim) for j in range(self.num_providers)]
            self.patients.append(Patient(patient_vector,utilities,self.choice_model_settings,i))
        self.patient_order = np.random.permutation(list(range(self.num_patients)))


    def reset_initial(self):
        self.provider_capacities = deepcopy(self.provider_max_capacities)
        self.provider_workloads = [[] for i in range(self.num_providers)]
        self.provider_vectors = [np.random.random(self.context_dim) for i in range(self.num_providers)]
        self.utility_coefficients = np.random.random(self.context_dim)

def run_heterogenous_policy(env,policy,seed,num_trials,per_epoch_function=None):
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
    P         = env.num_providers   
    T         = env.num_patients

    random.seed(seed)
    np.random.seed(seed)
    env.reset_initial()
    provider_workloads = [[] for i in range(P)]
    time_taken = 0

    for _ in range(num_trials):
        env.reset_patients()

        patient_list = env.patients 

        start = time.time()
        available_providers  = [1 if i > 0 else 0 for i in env.provider_capacities]
        memory = None 

        matched_pairs = []
        unmatched_pairs = []
        preference_pairs = []
        unmatched_patients = []

        for t in range(0, T):
            current_patient = patient_list[env.patient_order[t]]
            selected_providers,memory = policy(env,current_patient,available_providers,memory,per_epoch_function)

            selected_providers *= np.array(available_providers)

            provider_workloads, available_providers,chosen_provider = env.step(env.patient_order[t],selected_providers)

            if chosen_provider >= 0:
                matched_pairs.append((current_patient.patient_vector,env.provider_vectors[chosen_provider]))
                unmatched_pairs.append((current_patient.patient_vector,env.provider_vectors[np.random.choice([i for i in range(len(env.provider_vectors)) if i != chosen_provider])]))

                for i in range(len(selected_providers)):
                    if selected_providers[i] == 1 and i != chosen_provider:
                        preference_pairs.append((current_patient.patient_vector,env.provider_vectors[chosen_provider],env.provider_vectors[i]))
            else:
                unmatched_patients.append(env.patient_order[t])
            env.unmatched_patients = unmatched_patients
        env.matched_pairs += matched_pairs 
        env.unmatched_pairs += unmatched_pairs 
        env.preference_pairs += preference_pairs 
        time_taken += time.time()-start 

    print("Took {} time".format(time_taken))

    return provider_workloads


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
        'provider_workloads': [], 
    }

    num_patients = parameters['num_patients']
    num_providers = parameters['num_providers']
    provider_capacity = parameters['provider_capacity']
    top_choice_prob = parameters['top_choice_prob']
    choice_model = parameters['choice_model']
    exit_option = parameters['exit_option']
    num_trials = parameters['num_trials']
    context_dim = parameters['context_dim']

    choice_model_settings = {
        'top_choice_prob': top_choice_prob, 
        'exit_option': exit_option
    }

    for seed in seed_list:
        simulator = Simulator(num_patients,num_providers,provider_capacity,choice_model_settings,choice_model,context_dim)

        policy_results = run_heterogenous_policy(simulator,policy,seed,num_trials,per_epoch_function=per_epoch_function)
        utilities_by_provider = policy_results

        num_matches = sum([len(i) for i in utilities_by_provider])
        patient_utilities = sum([sum(i) for i in utilities_by_provider])
        provider_workloads = [len(i) for i in utilities_by_provider]

        scores['matches'].append(num_matches)
        scores['patient_utilities'].append(patient_utilities)
        scores['provider_workloads'].append(provider_workloads)

    return scores, simulator