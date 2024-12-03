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
from patient.utils import get_save_path, delete_duplicate_results, restrict_resources, one_shot_policy, MyEncoder

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 43
    num_patients = 3
    num_providers = 3
    provider_capacity = 1
    top_choice_prob = 0.5
    true_top_choice_prob = 0.5
    choice_model = "uniform_choice"
    exit_option = 0.5
    utility_function = "normal"
    out_folder = "policy_comparison"
    num_repetitions = 1
    num_trials = 100
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
    parser.add_argument('--previous_patients_per_provider',          help='Context dim for patients and providers', type=int, default=10)
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
restrict_resources()

# +
policy = random_policy
name = "random"
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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
    results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list)),np.max(np.mean(np.array(rewards['final_workloads'])[0],axis=0))

# +
policy = one_shot_policy
per_epoch_function = lp_workload_policy
name = "lp_workload"
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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
    results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list))

# +
# policy = one_shot_policy 
# for lamb in [0.25,0.5,1,2,4]:
#     per_epoch_function = provider_focused_linear_regularization_policy(lamb)
#     name = "provider_focused_linear_regularization_{}".format(lamb)
#     print("{} policy".format(name))

#     rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

#     results['{}_matches'.format(name)] = rewards['matches']
#     results['{}_utilities'.format(name)] = rewards['patient_utilities']
#     results['{}_workloads'.format(name)] = rewards['provider_workloads']

#     results['{}_minimums'.format(name)] = rewards['provider_minimums']
#     results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
#     results['{}_gaps'.format(name)] = rewards['provider_gaps']
#     results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
#     results['{}_variance'.format(name)] = rewards['provider_variance']
#     results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
#     results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

#     print(lamb,np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list)))

# +
# policy = one_shot_policy 
# for lamb in [0,0.1,0.25,0.5]:#,1,2,4]:
#     per_epoch_function = provider_focused_log_regularization_policy(lamb)
#     name = "provider_focused_log_regularization_{}".format(lamb)
#     print("{} policy".format(name))

#     rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

#     results['{}_matches'.format(name)] = rewards['matches']
#     results['{}_utilities'.format(name)] = rewards['patient_utilities']
#     results['{}_workloads'.format(name)] = rewards['provider_workloads']

#     results['{}_minimums'.format(name)] = rewards['provider_minimums']
#     results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
#     results['{}_gaps'.format(name)] = rewards['provider_gaps']
#     results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
#     results['{}_variance'.format(name)] = rewards['provider_variance']
#     results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
#     results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

#     print(lamb,np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list)))

# +
# policy = one_shot_policy 
# per_epoch_function = gradient_descent_policy
# name = "gradient_descent"
# print("{} policy".format(name))

# rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

# results['{}_matches'.format(name)] = rewards['matches']
# results['{}_utilities'.format(name)] = rewards['patient_utilities']
# results['{}_workloads'.format(name)] = rewards['provider_workloads']

# results['{}_minimums'.format(name)] = rewards['provider_minimums']
# results['{}_minimums_all'.format(name)] = rewards['provider_minimums_all']
# results['{}_gaps'.format(name)] = rewards['provider_gaps']
# results['{}_gaps_all'.format(name)] = rewards['provider_gaps_all']
# results['{}_variance'.format(name)] = rewards['provider_variance']
# results['{}_variance_all'.format(name)] = rewards['provider_variance_all']
# results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

# print(np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list)))

# +
policy = one_shot_policy 
per_epoch_function = gradient_descent_policy_2
name = "gradient_descent_2"
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
results['{}_workload_diff'.format(name)] = [max(rewards['final_workloads'][0][i])-max(rewards['initial_workloads'][0][i]) for i in range(len(rewards['final_workloads'][0]))]

print(np.sum(rewards['matches'])/(num_patients*num_trials*len(seed_list)),np.sum(rewards['patient_utilities'])/(num_patients*num_trials*len(seed_list)))


# -

def objective_old(z, theta, p, sorted_theta,lamb=1, smooth_reg='entropy', epsilon=1e-5):
    # Reparameterize x using sigmoid
    x = torch.sigmoid(z)  # x is now bounded in [0, 1]
    
    # Compute the sum of x across all columns for each row
    row_sums = torch.sum(x, dim=0, keepdim=True)  # Shape: (rows, 1)
    
    # Normalize x by row sums
    normalized_x = x / (p*torch.maximum(row_sums, torch.tensor(1.0, device=x.device)))*(1-(1-p)**(torch.maximum(row_sums, torch.tensor(1.0, device=x.device)))) 

    sorted_normalized_x = normalized_x.gather(1, sorted_theta)

    # Compute cumulative products (1 - normalized_x) along rows
    one_minus_sorted = 1 - sorted_normalized_x
    cumprods = torch.cumprod(one_minus_sorted, dim=1)

    # Shift the cumulative products to use for the original scaling (prepending 1 for first index)
    shifted_cumprods = torch.cat([torch.ones(cumprods.size(0), 1, device=cumprods.device), cumprods[:, :-1]], dim=1)

    # Apply the cumulative product scaling to the original indices
    scaled_normalized_x = sorted_normalized_x * shifted_cumprods

    # Scatter back to the original positions
    normalized_x = torch.zeros_like(normalized_x)
    normalized_x.scatter_(1, sorted_theta, scaled_normalized_x)
    # Normalize row-wise

    prod = p*torch.sum(normalized_x,dim=0)

    # Compute numerator for the first term (using normalized x)
    term1_num = prod * torch.sum(normalized_x * theta, dim=0)

    term1_den = torch.sum(normalized_x, dim=0) + 1e-8  # Avoid division by zero
    term1_den = torch.maximum(term1_den,torch.tensor(1.0, device=x.device))

    # Compute the main term
    term1 = (term1_num / term1_den)
        
    term1 = torch.sum(term1) / theta.shape[1]  # Normalize by number of columns

    reg_term = 0
    # Add smooth regularization term
    if smooth_reg == 'logit' and lamb > 0:
        reg_term = torch.sum(torch.logit(x, eps=epsilon) ** 2)  # Logit-based penalty
    elif smooth_reg == 'entropy' and lamb > 0:
        reg_term = -torch.sum(x * torch.log(x + epsilon) + (1 - x) * torch.log(1 - x + epsilon))  # Entropy-based penalty
    loss = term1 - lamb * reg_term

    return loss



def objective2(z, theta, p, sorted_theta,lamb=1, smooth_reg='entropy', epsilon=1e-5):
    x = torch.sigmoid(z)
        
    rows_with_top_i = torch.zeros((theta.shape[1], theta.shape[0]), device=theta.device)
    argsorted = torch.argsort(theta, dim=1, descending=True)

    # Mask elements in `argsorted` where x is 0
    mask = x.gather(1, argsorted) != 0

    # Filter out -1 indices
    filtered_argsorted = [
        torch.masked_select(argsorted[i], mask[i]) for i in range(len(argsorted))
    ]
    is_top_k = torch.zeros((theta.shape[0], theta.shape[1], theta.shape[0]), device=theta.device)

    # We need to update rows_with_top_i for each provider from their rank onwards
    for patient in range(len(filtered_argsorted)):
        # For each provider at the current rank
        for rank in range(len(filtered_argsorted[patient])):
            provider = filtered_argsorted[patient][rank]
            # Update rows_with_top_i for this provider starting from the current rank onwards
            rows_with_top_i[provider, rank:] += 1
            is_top_k[patient,provider,rank:] +=  1 / (theta.shape[0] - 1)


    # Normalize rows_with_top_i
    rows_with_top_i /= (theta.shape[0] - 1)

    # Step 4: Normalize `x`
    normalized_x = torch.zeros_like(x)
    delta = rows_with_top_i[None, :, :] - is_top_k

    delta = torch.roll(delta, shifts=1, dims=2)  # Shift columns to the right (dims=2 corresponds to columns)
    delta[:, :, 0] = 0  # Set the leftmost column to 1


    delta = 1-p*delta
    delta_raised = torch.cumprod(delta,dim=2)
    # Raise delta to successive powers along the third dimension
    delta_swapped = delta_raised.permute(0, 2, 1)

    normalized_x = x[:,None,:] * delta_swapped

    print("Normalized x {}".format(normalized_x))

    sorted_normalized_x = normalized_x.gather(dim=2, index=sorted_theta.unsqueeze(1).expand(-1, normalized_x.size(1), -1))

    # Compute cumulative products (1 - normalized_x) along rows
    one_minus_sorted = 1 - sorted_normalized_x
    cumprods = torch.cumprod(one_minus_sorted, dim=2)

    # # Shift the cumulative products to use for the original scaling (prepending 1 for first index)
    shifted_cumprods = torch.cat([torch.ones(cumprods.size(0), cumprods.size(1) ,1,device=cumprods.device), cumprods[:,:,:-1]], dim=2)


    # Apply the cumulative product scaling to the original indices
    scaled_normalized_x = sorted_normalized_x * shifted_cumprods
    scaled_normalized_x = torch.mean(scaled_normalized_x,dim=1)

    # Scatter back to the original positions
    normalized_x = torch.zeros_like(scaled_normalized_x)
    normalized_x.scatter_(1, sorted_theta, scaled_normalized_x)
    # Normalize row-wise
    # Compute numerator for the first term (using normalized x)
    term = p * torch.sum(normalized_x * theta, dim=0)
        
    term = torch.sum(term) / theta.shape[1]  # Normalize by number of columns

    reg_term = 0
    # Add smooth regularization term
    if smooth_reg == 'logit' and lamb > 0:
        reg_term = torch.sum(torch.logit(x, eps=epsilon) ** 2)  # Logit-based penalty
    elif smooth_reg == 'entropy' and lamb > 0:
        reg_term = -torch.sum(x * torch.log(x + epsilon) + (1 - x) * torch.log(1 - x + epsilon))  # Entropy-based penalty
    loss = term - lamb * reg_term

    return loss


if is_jupyter:
    theta = [p.provider_rewards for p in simulator.patients]
    theta = torch.Tensor(theta)
    sorted_theta = torch.argsort(theta, dim=1,descending=True)  # Sorting indices of `theta` row-wise
    opt_tensor = torch.Tensor(lp_policy(simulator))
    ones_tensor = torch.Tensor(np.ones(opt_tensor.shape))
    # x = torch.Tensor(gradient_descent_policy_2(simulator))
    p = true_top_choice_prob

if is_jupyter:
    print(objective2(ones_tensor*10000-10000/2,theta,p,sorted_theta,0))
    # print(objective2(x*10000-10000/2,theta,p,sorted_theta,0))
    print(objective2(opt_tensor*10000-10000/2,theta,p,sorted_theta,0))

# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'),cls=MyEncoder)


