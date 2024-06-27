import numpy as np 
import random 
from copy import deepcopy 
import time 

class Patient:
    """Class to represent the Patient and their information"""

    def __init__(self,provider_rewards,choice_model_settings,idx):
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
            
            if np.random.random() < self.choice_model_settings['top_choice_prob'] and np.max(provider_probs) > -1:
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

    def __init__(self,num_patients,num_providers,provider_capacity,choice_model_settings,choice_model):
        self.num_patients = num_patients 
        self.num_providers = num_providers
        self.provider_max_capacity = provider_capacity
        self.choice_model_settings = choice_model_settings
        self.choice_model = choice_model 

        self.patients = []
        for i in range(num_patients):
            utilities = [np.random.random() for _ in range(num_providers)]
            self.patients.append(Patient(utilities,choice_model_settings,i))
        
        self.provider_max_capacities = [provider_capacity for i in range(self.num_providers)]
        self.provider_capacities = deepcopy(self.provider_max_capacities)

        self.patient_order = np.random.permutation(list(range(num_patients)))
        self.provider_workloads = [[] for i in range(num_providers)]
    
    def step(self,patient_num,provider_list):
        chosen_provider = self.patients[patient_num].get_random_outcome(provider_list,self.choice_model)
        if chosen_provider >= 0:
            patient_utility = self.patients[patient_num].provider_rewards[chosen_provider]
            self.provider_workloads[chosen_provider].append(patient_utility)
            self.provider_capacities[chosen_provider] -= 1
        
        available_providers = [1 if i > 0 else 0 for i in self.provider_capacities]
        return self.provider_workloads, available_providers

    def reset_all(self):
        self.provider_capacities = deepcopy(self.provider_max_capacities)

        self.patient_order = np.random.permutation(list(range(self.num_patients)))
        self.provider_workloads = [[] for i in range(self.num_providers)]

def run_heterogenous_policy(env,policy,seed,per_epoch_function=None):
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

    env.reset_all()

    provider_workloads = [[] for i in range(P)]
    patient_list = env.patients 

    start = time.time()
    available_providers  = [1 if i > 0 else 0 for i in env.provider_capacities]
    memory = None 
    for t in range(0, T):
        current_patient = patient_list[env.patient_order[t]]
        selected_providers,memory = policy(env,current_patient,available_providers,memory,per_epoch_function)
        selected_providers *= np.array(available_providers)

        provider_workloads, available_providers = env.step(env.patient_order[t],selected_providers)


    time_taken = time.time()-start 
    env.time_taken = time_taken 

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

    choice_model_settings = {
        'top_choice_prob': top_choice_prob, 
        'exit_option': exit_option
    }

    for seed in seed_list:
        simulator = Simulator(num_patients,num_providers,provider_capacity,choice_model_settings,choice_model)

        policy_results = run_heterogenous_policy(simulator,policy,seed,per_epoch_function=per_epoch_function)
        utilities_by_provider = policy_results

        num_matches = sum([len(i) for i in utilities_by_provider])
        patient_utilities = sum([sum(i) for i in utilities_by_provider])
        provider_workloads = [len(i) for i in utilities_by_provider]

        scores['matches'].append(num_matches)
        scores['patient_utilities'].append(patient_utilities)
        scores['provider_workloads'].append(provider_workloads)

    return scores, simulator