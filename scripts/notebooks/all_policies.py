# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: patient
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append('/usr0/home/naveenr/projects/patient_provider')

import numpy as np
import random 
import matplotlib.pyplot as plt
import argparse
import secrets
import json

from patient.simulator import run_multi_seed
from patient.baseline_policies import random_policy, greedy_policy
from patient.online_policies import p_approximation, p_approximation_balance, p_approximation_with_additions, p_approximation_with_additions_extra_provider
from patient.offline_policies import offline_solution, offline_solution_balance, offline_learning_solution, offline_solution_more_patients
from patient.utils import get_save_path, delete_duplicate_results, restrict_resources
from patient.learning import guess_coefficients

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 43
    num_patients = 50
    num_providers = 25
    provider_capacity = 1
    top_choice_prob = 0.5
    choice_model = "uniform_choice"
    utility_function = "uniform"
    out_folder = "policy_comparison"
    exit_option = 0.5
    true_top_choice_prob = 0.5
    num_repetitions = 1
    num_trials = 10
    context_dim = 5
    max_menu_size = 5
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--n_patients',         '-N', help='Number of patients', type=int, default=100)
    parser.add_argument('--n_providers',        help='Number of providers', type=int, default=100)
    parser.add_argument('--n_trials',          help='Number of trials ', type=int, default=2)
    parser.add_argument('--top_choice_prob',          help='Probability of picking top choice', type=float, default=0.75)
    parser.add_argument('--true_top_choice_prob',          help='Probability of picking top choice', type=float, default=0.75)
    parser.add_argument('--context_dim',          help='Context dim for patients and providers', type=int, default=5)
    parser.add_argument('--max_menu_size',          help='Context dim for patients and providers', type=int, default=5)
    parser.add_argument('--num_repetitions',          help='Context dim for patients and providers', type=int, default=10)
    parser.add_argument('--provider_capacity', help='Provider Capacity', type=int, default=5)
    parser.add_argument('--choice_model', help='Which choice model for patients', type=str, default='uniform_choice')
    parser.add_argument('--exit_option', help='What is the value of the exit option', type=float, default=0.5)
    parser.add_argument('--out_folder', help='Which folder to write results to', type=str, default='policy_comparison')
    parser.add_argument('--utility_function', help='Which folder to write results to', type=str, default='uniform')

    args = parser.parse_args()

    seed = args.seed
    num_patients = args.n_patients
    num_providers = args.n_providers 
    provider_capacity = args.provider_capacity
    top_choice_prob = args.top_choice_prob
    choice_model = args.choice_model
    exit_option = args.exit_option
    out_folder = args.out_folder
    num_trials = args.n_trials 
    context_dim = args.context_dim 
    num_repetitions = args.num_repetitions
    true_top_choice_prob = args.true_top_choice_prob
    max_menu_size = args.max_menu_size
    utility_function = args.utility_function

save_name = secrets.token_hex(4)  
# -

results = {}
results['parameters'] = {'seed'      : seed,
        'num_patients'    : num_patients,
        'num_providers': num_providers, 
        'provider_capacity'    : provider_capacity,
        'top_choice_prob': top_choice_prob, 
        'choice_model': choice_model,
        'exit_option': exit_option,
        'num_trials': num_trials,
        'context_dim': context_dim, 
        'true_top_choice_prob': true_top_choice_prob, 
        'num_repetitions': num_repetitions, 
        'max_menu_size': max_menu_size, 
        'utility_function': utility_function} 

# ## Baselines

seed_list = [seed]
restrict_resources()

# +
policy = random_policy
name = "random"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'])

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))

# +
policy = greedy_policy
name = "greedy"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'])

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))
# -

# ## Offline

# +
policy = offline_solution
name = "offline_solution"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'])

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))

# +
policy = p_approximation_with_additions
name = "offline_solution_swaps"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'])

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))

# +
policy = offline_solution_more_patients
name = "more_patients_than_providers"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'])

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))
# -

# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))


