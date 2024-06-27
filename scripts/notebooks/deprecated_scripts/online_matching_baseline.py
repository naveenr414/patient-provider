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

from patient.simulator import Simulator
from patient.policy import *
from patient.utils import get_save_path, delete_duplicate_results

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed_list        = list(range(42,52))
    num_patients = 100
    num_providers = 20
    provider_capacity = 5
    out_folder = "online_baseline"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_patients',         '-N', help='Number of patients', type=int, default=100)
    parser.add_argument('--n_providers',         '-V', help='Number of providers', type=int, default=20)
    parser.add_argument('--provider_capacity',    '-P', help='Provider Capacity', type=int, default=5)
    parser.add_argument('--out_folder', help='Which folder to write results to', type=str, default='online_baseline')

    args = parser.parse_args()

    num_patients = args.n_patients
    num_providers = args.n_providers 
    provider_capacity = args.provider_capacity
    out_folder = args.out_folder

save_name = secrets.token_hex(4)  
# -

s = Simulator(num_patients,num_providers,provider_capacity)

seed_list = list(range(42,52))

results = {}
results['parameters'] = {'seed_list'      : seed_list,
        'num_patients'    : num_patients,
        'num_providers': num_providers, 
        'provider_capacity'    : provider_capacity,} 

# ## No Re-Entry

# +
policy = random_policy
name = "random"

rewards = s.simulate_no_renetry(policy,seed_list=seed_list)
results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_gaps'.format(name)] = rewards['provider_gaps']

np.mean(rewards['matches']), np.mean(rewards['patient_utilities']), np.mean(rewards['provider_gaps'])

# +
policy = max_match_prob
name = "match_prob"

rewards = s.simulate_no_renetry(policy,seed_list=seed_list)
results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_gaps'.format(name)] = rewards['provider_gaps']

np.mean(rewards['matches']), np.mean(rewards['patient_utilities']), np.mean(rewards['provider_gaps'])

# +
policy = max_patient_utility
name = "utility"

rewards = s.simulate_no_renetry(policy,seed_list=seed_list)
results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_gaps'.format(name)] = rewards['provider_gaps']

np.mean(rewards['matches']), np.mean(rewards['patient_utilities']), np.mean(rewards['provider_gaps'])
# -

# ## Re-Entry

# +
policy = random_policy
name = "random_reentry"

rewards = s.simulate_with_renetry(policy,seed_list=seed_list)
results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_waittimes'.format(name)] = rewards['waittimes']

np.mean(rewards['matches']), np.mean(rewards['waittimes']), np.mean(rewards['patient_utilities']), np.mean(rewards['provider_gaps'])

# +
policy = max_match_prob
name = "match_prob_reentry"

rewards = s.simulate_with_renetry(policy,seed_list=seed_list)
results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_waittimes'.format(name)] = rewards['waittimes']

np.mean(rewards['matches']), np.mean(rewards['waittimes']), np.mean(rewards['patient_utilities']), np.mean(rewards['provider_gaps'])

# +
policy = max_match_prob
name = "utility_reentry"

rewards = s.simulate_with_renetry(policy,seed_list=seed_list)
results['{}_matches'.format(name)] = rewards['matches']
results['{}_utilities'.format(name)] = rewards['patient_utilities']
results['{}_gaps'.format(name)] = rewards['provider_gaps']
results['{}_waittimes'.format(name)] = rewards['waittimes']

np.mean(rewards['matches']), np.mean(rewards['waittimes']), np.mean(rewards['patient_utilities']), np.mean(rewards['provider_gaps'])

# +
name = "alpha_lambda_reentry"

results['{}_matches'.format(name)] = []
results['{}_utilities'.format(name)] = []
results['{}_gaps'.format(name)] = []
results['{}_waittimes'.format(name)] = []


for lamb in [0,0.25,0.5,1,5,10]:
    for alpha in [0,0.25,0.5,1,5,10]:
        policy = max_patient_utility_with_waittime_alpha_lambda(alpha,lamb)

        rewards = s.simulate_with_renetry(policy,seed_list=seed_list)
        temp_dict = {}
        results['{}_matches'.format(name)].append({'alpha': alpha,'lamb': lamb,'matches': rewards['matches']}) 
        results['{}_utilities'.format(name)].append({'alpha': alpha,'lamb': lamb,'utilities': rewards['patient_utilities']}) 
        results['{}_gaps'.format(name)].append({'alpha': alpha,'lamb': lamb,'gaps': rewards['provider_gaps']}) 
        results['{}_waittimes'.format(name)].append({'alpha': alpha,'lamb': lamb,'gaps': rewards['waittimes']}) 
# -

# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))
