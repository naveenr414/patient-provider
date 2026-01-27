# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py  
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
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

import sys
sys.path.append('/usr0/home/naveenr/projects/patient_provider')

from patient.simulator import run_multi_seed
from patient.baseline_policies import *
from patient.lp_policies import *
from patient.utils import get_save_path, delete_duplicate_results, restrict_resources, one_shot_policy, MyEncoder

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 43
    num_patients = 1225
    num_providers = 700
    provider_capacity = 1
    noise = 0.1
    fairness_constraint = -1
    num_trials = 25
    utility_function = "semi_synthetic_comorbidity"
    order="uniform"
    online_arrival = True   
    new_provider = False 
    out_folder = "dynamic"
    average_distance = 20.2
    max_shown = 25
    online_scale = 1
    verbose = False 
    num_samples = 10
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--n_patients',         '-N', help='Number of patients', type=int, default=100)
    parser.add_argument('--n_providers',        help='Number of providers', type=int, default=100)
    parser.add_argument('--max_shown',        help='How many providers to show at most', type=int, default=25)
    parser.add_argument('--num_samples',        help='How many samples to run', type=int, default=10)
    parser.add_argument('--provider_capacity', help='Provider Capacity', type=int, default=1)
    parser.add_argument('--noise', help='Noise in theta', type=float, default=0.1)
    parser.add_argument('--average_distance', help='Maximum distance patients are willing to go', type=float, default=20.2)
    parser.add_argument('--fairness_constraint', help='Maximum difference in average utility between groups', type=float, default=-1)
    parser.add_argument('--num_trials', help='Number of trials', type=int, default=100)
    parser.add_argument('--utility_function', help='Which folder to write results to', type=str, default='uniform')
    parser.add_argument('--online_scale', help='Which folder to write results to', type=float, default=1)
    parser.add_argument('--order', help='Which folder to write results to', type=str, default='uniform')
    parser.add_argument("--online_arrival",action="store_true",help="Patients arrive one-by-one")
    parser.add_argument("--new_provider",action="store_true",help="Are we simulating a new provider matching")
    parser.add_argument("--verbose",action="store_true",help="Are we storing all the information")
    parser.add_argument('--out_folder', help='Which folder to write results to', type=str, default='policy_comparison')

    args = parser.parse_args()

    seed = args.seed
    num_patients = args.n_patients
    num_providers = args.n_providers 
    num_trials = args.num_trials
    noise = args.noise
    average_distance = args.average_distance
    fairness_constraint = args.fairness_constraint
    provider_capacity = args.provider_capacity
    max_shown = args.max_shown
    utility_function = args.utility_function
    order = args.order
    online_arrival = args.online_arrival
    new_provider = args.new_provider 
    online_scale = args.online_scale
    verbose = args.verbose 
    out_folder = args.out_folder
    num_samples = args.num_samples
    
assert not(online_arrival and new_provider)
save_name = secrets.token_hex(4)  
# -

results = {}
results['parameters'] = {'seed'      : seed,
        'num_patients'    : num_patients,
        'num_providers': num_providers, 
        'provider_capacity'    : provider_capacity,
        'max_shown': max_shown,
        'utility_function': utility_function, 
        'order': order, 
        'num_trials': num_trials, 
        'noise': noise, 
        'average_distance': average_distance,
        'online_arrival': online_arrival,
        'new_provider': new_provider,
        'fairness_constraint': fairness_constraint,
        'online_scale': online_scale, 
        'verbose': verbose, 
        'num_samples': num_samples} 

# ## Baselines

seed_list = [seed]
restrict_resources()

if fairness_constraint == -1:
    policy = one_shot_policy
    per_epoch_function = lambda m: greedy_justified(m,K=num_samples)
    name = "greedy_justified"
    print("{} policy".format(name))

    rewards, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function)

    for key in rewards:
        results['{}_{}'.format(name,key)] = rewards[key]
    print("Matches {}, Utilities {}".format(np.mean(results['{}_num_matches'.format(name)])/num_patients,np.mean(results['{}_patient_utilities'.format(name)])))

# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'),cls=MyEncoder)
