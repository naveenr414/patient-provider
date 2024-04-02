import random 
import numpy as np 

def random_policy(patient,provider_capacities):
    available_providers = [i for i in range(len(provider_capacities)) if random.random() < 0.5]
    return available_providers

def greedy_policy(patient,provider_capacities):
    return list(range(len(provider_capacities)))

def brute_force_discount(patient,provider_capacities):
    num_providers = len(provider_capacities)

    def discount_function(x):
        return np.exp(-(1-x/2))

    max_rep = []
    max_prob = 0

    for i in range(2**num_providers):
        binary_rep = [int(j) for j in bin(i)[2:].zfill(num_providers)]
        lst_rep = [i for i in range(len(binary_rep)) if binary_rep[i]]
        prob_match = 0

        for j in range(num_providers):
            if binary_rep[j]*provider_capacities[j] > 0:
                prob_match += patient.get_match_prob_provider(j,lst_rep)*(1-discount_function(2-provider_capacities[j]))
        
        if prob_match > max_prob:
            max_prob = prob_match 
            max_rep = lst_rep 
    
    return max_rep 

def brute_force_discount_lamb(patient,provider_capacities):
    num_providers = len(provider_capacities)
    lamb = 0.5

    def discount_function(x):
        return np.exp(-(1-x/2))

    max_rep = []
    max_prob = 0

    for i in range(2**num_providers):
        binary_rep = [int(j) for j in bin(i)[2:].zfill(num_providers)]
        lst_rep = [i for i in range(len(binary_rep)) if binary_rep[i]]
        prob_match = 0

        for j in range(num_providers):
            if binary_rep[j]*provider_capacities[j] > 0:
                prob_match += patient.get_match_prob_provider(j,lst_rep)*(1-discount_function(2-provider_capacities[j]))
        prob_match -= lamb * patient.discount*np.max(patient.provider_rewards)/(patient.get_all_probabilities(lst_rep))

        if prob_match > max_prob:
            max_prob = prob_match 
            max_rep = lst_rep 
    
    return max_rep 