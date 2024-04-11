import numpy as np 
import random 
from copy import deepcopy 

class Patient:
    """Class to represent the Patient and their information"""

    def __init__(self,provider_rewards,exit_option,discount,idx):
        self.discount = discount
        self.provider_rewards = provider_rewards
        self.exit_option = exit_option 
        self.discount = discount 
        self.idx = idx 
    
    def get_all_probabilities(self):
        return np.sum(self.provider_rewards) + self.exit_option + self.discount*(np.max(self.provider_rewards))

    def get_all_probabilities(self,menu):
        return np.sum(np.array(self.provider_rewards)[menu]) + self.exit_option + self.discount*(np.max(self.provider_rewards))

    def get_match_prob_provider(self,provider_idx,menu):
        return self.provider_rewards[provider_idx]/self.get_all_probabilities(menu)
    
    def get_exit_probability(self):
        return self.exit_option/self.get_all_probabilities()

    def get_random_outcome(self,menu,reenter=False):
        """Given a menu, get a random outcome; either match, re-enter, or exit

        Arguments:
            menu: List of providers who are available/presented

        Returns: Integer, -1 is exit, len(self.provider_rewards) is continue/re-enter
            And other numbers signify a particular provider"""

        provider_probs = [self.provider_rewards[i] if i in menu else 0 for i in range(len(self.provider_rewards))]
        all_options = [self.exit_option] + provider_probs
        if reenter:
            all_options += [self.discount*(np.max(self.provider_rewards))] 
        all_options = np.array(all_options)/np.sum(all_options)
        outcome = np.random.choice(list(range(len(all_options))), 1,
            p=all_options)[0]-1
        return outcome 

class Simulator():
    """Simulator class that allows for evaluation of policies
        Both with and without re-entry"""

    def __init__(self,num_patients,num_providers):
        self.num_patients = num_patients 
        self.num_providers = num_providers

        self.patients = []
        for i in range(num_patients):
            utilities = [np.random.random() for i in range(num_providers)]
            discount = np.random.random() 
            exit_option = np.random.random() 
            self.patients.append(Patient(utilities,exit_option,discount,i))
        
        self.provider_max_capacities = [2 for i in range(self.num_providers)]
        self.provider_capacities = deepcopy(self.provider_max_capacities)


    def simulate_no_renetry(self,policy,seed_list=[42]):
        """Evaluate a policy without re-entry
        
        Arguments:
            policy: Some function that takes in patient, provider capacity, provider max_capacity
                and outputs a menu
        
        Returns: List of number of matches per seed"""

        all_matches = []
        utilities = []

        for seed in seed_list:
            np.random.seed(seed)
            random.seed(seed)
            self.__init__(self.num_patients,self.num_providers)

            self.provider_capacities = deepcopy(self.provider_max_capacities)
            num_matches = 0
            curr_utilities = []
            for i in range(self.num_patients):
                menu = policy(self.patients[i],self.provider_capacities,self.provider_max_capacities)
                menu = [i for i in menu if self.provider_capacities[i] > 0]
                outcome = self.patients[i].get_random_outcome(menu)

                if outcome >= 0 and outcome < self.num_providers:
                    num_matches += 1 
                    self.provider_capacities[int(outcome)] -= 1
            
                if outcome == -1:
                    curr_utilities.append(self.patients[i].exit_option)
                elif outcome < self.num_providers:
                    curr_utilities.append(self.patients[i].provider_rewards[outcome])
            all_matches.append(num_matches)
            utilities.append(curr_utilities)
        return {'matches': all_matches, 'patient_utilities': utilities}

    def simulate_with_renetry(self,policy,seed_list=[42]):
        """Evaluate a policy with re-entry
        
        Arguments:
            policy: Some function that takes in patient, provider capacity, provider max_capacity
                and outputs a menu
        
        Returns: Dictionary: 'matches' is number of matches for each seed
            'waittimes' is the list of waittimes for each patient for each seed"""


        all_scores = {'matches': [], 'waittimes': [],'patient_utilities': []}

        for seed in seed_list:
            np.random.seed(seed)
            random.seed(seed)
            self.provider_capacities = deepcopy(self.provider_max_capacities)

            num_matches = 0
            curr_patients = []
            waittimes = []
            patient_utilities = []
            for i in range(self.num_patients):
                curr_patients.append(self.patients[i])
                new_curr_patients = []

                for p in curr_patients:
                    menu = policy(self.patients[i],self.provider_capacities,self.provider_max_capacities)
                    menu = [i for i in menu if self.provider_capacities[i] > 0]
                    outcome = p.get_random_outcome(menu,reenter=True)

                    if outcome >= 0 and outcome < self.num_providers:
                        num_matches += 1 
                        self.provider_capacities[int(outcome)] -= 1
                        waittimes.append(i - p.idx)
                        patient_utilities.append(self.patients[i].provider_rewards[outcome])
                    elif outcome == self.num_providers:
                        new_curr_patients.append(p)
                    else:
                        waittimes.append(i - p.idx)
                        patient_utilities.append(self.patients[i].exit_option)
                curr_patients = new_curr_patients
            for p in curr_patients: 
                waittimes.append(i-p.idx)
                patient_utilities.append(p.exit_option)
                    
            all_scores['matches'].append(num_matches)
            all_scores['waittimes'].append(waittimes)
            all_scores['patient_utilities'].append(patient_utilities)
        all_scores['patient_utilities'] = np.array(all_scores['patient_utilities']).flatten()
        return all_scores  