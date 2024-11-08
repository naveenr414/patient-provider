# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import numpy as np
import random 
import matplotlib.pyplot as plt
import argparse
import secrets
import json
import sys
import math 

from patient.simulator import run_multi_seed
from patient.baseline_policies import *
from patient.lp_policies import *
from patient.group_based_policies import *
from patient.ordering_policies import *
from patient.provider_policies import *
from patient.utils import get_save_path, delete_duplicate_results, restrict_resources, one_shot_policy

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 44
    num_patients = 10
    num_providers = 10
    provider_capacity = 1
    top_choice_prob = 0.5
    choice_model = "uniform_choice"
    utility_function = "semi_synthetic"
    out_folder = "policy_comparison"
    exit_option = 0.5
    true_top_choice_prob = 0.5
    num_repetitions = 1
    num_trials = 1000
    context_dim = 5
    max_menu_size = 25
    previous_patients_per_provider = 10
    order="custom"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--n_patients',         '-N', help='Number of patients', type=int, default=100)
    parser.add_argument('--n_providers',        help='Number of providers', type=int, default=100)
    parser.add_argument('--n_trials',          help='Number of trials ', type=int, default=2)
    parser.add_argument('--top_choice_prob',          help='Probability of picking top choice', type=float, default=0.75)
    parser.add_argument('--true_top_choice_prob',          help='Probability of picking top choice', type=float, default=0.75)
    parser.add_argument('--context_dim',          help='Context dim for patients and providers', type=int, default=5)
    parser.add_argument('--max_menu_size',          help='Context dim for patients and providers', type=int, default=50)
    parser.add_argument('--num_repetitions',          help='Context dim for patients and providers', type=int, default=1)
    parser.add_argument('--previous_patients_per_provider',          help='Context dim for patients and providers', type=int, default=1)
    parser.add_argument('--provider_capacity', help='Provider Capacity', type=int, default=5)
    parser.add_argument('--choice_model', help='Which choice model for patients', type=str, default='uniform_choice')
    parser.add_argument('--exit_option', help='What is the value of the exit option', type=float, default=0.5)
    parser.add_argument('--out_folder', help='Which folder to write results to', type=str, default='policy_comparison')
    parser.add_argument('--utility_function', help='Which folder to write results to', type=str, default='uniform')
    parser.add_argument('--order', help='Which folder to write results to', type=str, default='random')

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
    order = args.order
    previous_patients_per_provider = args.previous_patients_per_provider

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
        'utility_function': utility_function, 
        'order': order, 
        'previous_patients_per_provider': previous_patients_per_provider} 

# ## Baselines

seed_list = [seed]
# restrict_resources()

# +
policy = random_policy
name = "random"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'])

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
results['{}_final_workloads'.format(name)] = rewards['final_workloads']

np.sum(rewards['matches'])/(num_patients*num_repetitions*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_repetitions*num_trials*len(seed_list))

# +
policy = all_ones_policy
name = "greedy_basic"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'])

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
results['{}_final_workloads'.format(name)] = rewards['final_workloads']

np.mean(results['{}_minimums_all'.format(name)]),np.mean(results['{}_gaps_all'.format(name)]),np.mean(results['{}_variance_all'.format(name)])

# +
policy = greedy_policy
name = "greedy"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'])

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))
# -

if 2**(num_patients*num_providers)*2**(num_patients)*math.factorial(num_patients) < 100000:
    policy = one_shot_policy
    per_epoch_function = optimal_policy
    name = "optimal"
    print("{} policy".format(name))

    rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

    results['{}_matches'.format(name)] = rewards['matches']
    results['{}_utilities'.format(name)] = rewards['patient_utilities']
    results['{}_workloads'.format(name)] = rewards['provider_workloads']

    results['{}_minimums'.format(name)] = rewards['provider_minimums']
    results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
    results['{}_gaps'.format(name)] = rewards['provider_gaps']
    results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
    results['{}_variance'.format(name)] = rewards['provider_variance']
    results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
    results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
    results['{}_final_workloads'.format(name)] = rewards['final_workloads']

    print(np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list)))

# +
policy = one_shot_policy
per_epoch_function = optimal_order_policy
name = "optimal_order"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
results['{}_final_workloads'.format(name)] = rewards['final_workloads']

print(np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list)))
# -

# ## Offline

# +
policy = one_shot_policy
per_epoch_function = lp_policy
name = "lp"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
results['{}_final_workloads'.format(name)] = rewards['final_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list)),np.max(np.mean(np.array(rewards['final_workloads'])[0],axis=0))

# +
policy = one_shot_policy
per_epoch_function = lp_workload_policy
name = "lp"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
results['{}_final_workloads'.format(name)] = rewards['final_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list)),np.max(np.mean(np.array(rewards['final_workloads'])[0],axis=0)),np.max(np.mean(np.array(rewards['final_workloads'])[0],axis=0))

# +
policy = one_shot_policy
per_epoch_function = lp_multiple_match_policy
name = "lp_multiple_match"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
results['{}_final_workloads'.format(name)] = rewards['final_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))
# -

if choice_model == 'threshold':
    policy = one_shot_policy 
    per_epoch_function = lp_threshold
    name = "lp_threshold"
    print("{} policy".format(name))

    rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

    results['{}_matches'.format(name)] = rewards['matches']
    results['{}_utilities'.format(name)] = rewards['patient_utilities']
    results['{}_workloads'.format(name)] = rewards['provider_workloads']

    results['{}_minimums'.format(name)] = rewards['provider_minimums']
    results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
    results['{}_gaps'.format(name)] = rewards['provider_gaps']
    results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
    results['{}_variance'.format(name)] = rewards['provider_variance']
    results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
    results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
    results['{}_final_workloads'.format(name)] = rewards['final_workloads']

    print(np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list)))

# +
policy = one_shot_policy 
per_epoch_function = lp_more_patients_policy
name = "lp_more_patients"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
results['{}_final_workloads'.format(name)] = rewards['final_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))

# +
policy = one_shot_policy
per_epoch_function = lp_fairness_policy
name = "lp_fairness"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
results['{}_final_workloads'.format(name)] = rewards['final_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))

# +
policy = one_shot_policy
per_epoch_function = group_based_policy
name = "group_based"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
results['{}_final_workloads'.format(name)] = rewards['final_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))

# +
policy = one_shot_policy
per_epoch_function = group_based_unidirectional_policy
name = "group_based_unidirectional"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
results['{}_final_workloads'.format(name)] = rewards['final_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))

# +
policy = one_shot_policy 
per_epoch_function = provider_focused_policy
name = "provider_focused"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
results['{}_final_workloads'.format(name)] = rewards['final_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))

# +
policy = one_shot_policy 
per_epoch_function = provider_focused_less_interference_policy
name = "provider_focused_less_interference"
print("{} policy".format(name))

rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_workloads'.format(name)] = rewards['provider_workloads']

results['{}_minimums'.format(name)] = rewards['provider_minimums']
results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
results['{}_variance'.format(name)] = rewards['provider_variance']
results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
results['{}_initial_workloads'.format(name)] = rewards['initial_workloads']
results['{}_final_workloads'.format(name)] = rewards['final_workloads']

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))
# -

# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))


