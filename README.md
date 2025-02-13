# Assortment Optimization for Patient-Provider Matching
![Paper Overview](figs/pull_figure.png)

This repository contains the implementation for the paper "Assortment Optimization for Patient-Provider Matching"

This work was done by [Naveen Raman](https://naveenraman.com/) and [Holly Wiberg](https://hwiberg.github.io).

#### TL;DR 
Rising provider turnover forces healthcare administrators to frequently rematch patients to available providers, which can be cumbersome and labor-intensive.
To reduce the burden of rematching, we study algorithms for matching patients and providers through assortment optimization.
We develop a patient-provider matching model in which we simultaneously offer each patient a menu of providers, and patients subsequently respond and select providers. 
We study policies for assortment optimization and characterize their performance under different problem settings. 
We demonstrate that the selection of assortment policy is highly dependent on problem specifics and, in particular, on a patient's willingness to match and the ratio between patients and providers .
On real-world data, we show that our best policy can improve match quality by 13\% over a greedy solution by tailoring assortment sizes based on patient characteristics.

## Setup

#### Installation
To run experiments with our patient-provider matching setup, first clone this repository
```$ git clone https://github.com/naveenr414/patient-provider``` 

Then to install, run the following: 
```$ conda env create --file environment.yaml
$ conda activate pateint
$ pip install -e .
$ bash scripts/bash_scripts/main_scripts/create_folders.sh
```

This will create a new environment, called `patient`, from which to run the code
To test whether the installation was successful, run 
```import patient
```

#### Evaluating Policies
To evaluate policies, run the `scripts/notebooks/All Policies.ipynb` notebook. 
This notebook evaluates all policies based on a set of parameters, and writes results to the `results/${out_folder}` folder, where out_folder is a parameter. 
For example, to run the random_policy, you can run the following: 
```
import random 
import numpy as np
from patient.simulator import run_multi_seed
from patient.utils import one_shot_policy

def random_policy(simulator):
    """Randomly give a menu of available providers
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, 0-1 vector of which providers to show """
    random_matrix = np.random.random((simulator.num_patients,simulator.num_providers))
    random_provider = np.round(random_matrix)
    return random_provider 

seed_list=[43]
policy = one_shot_policy
per_epoch_function = random_policy
parameters={'seed': 43,
  'num_patients': 20,
  'num_providers': 20,
  'provider_capacity': 1,
  'top_choice_prob': 0.75,
  'choice_model': 'uniform_choice',
  'exit_option': 0.5,
  'num_trials': 100,
  'context_dim': 5,
  'true_top_choice_prob': 0.75,
  'num_repetitions': 1,
  'max_menu_size': 1000,
  'utility_function': 'semi_synthetic_comorbidity',
  'order': 'custom',
  'previous_patients_per_provider': 10,
  'batch_size': 1,
  'fairness_weight': 0}

rewards, simulator = run_multi_seed(seed_list,policy,parameters,per_epoch_function)
print(np.mean(rewards['patient_utilities']))
```

#### Re-Running Experiments 
All bash scripts for experiments can be found in the `scripts/bash_scripts/main_scripts` folder. 
To run all the experiments, run `bash scripts/bash_scripts/main_scripts/run_all_policies.sh`

#### Running custom policies
To run custom policies, define a function that takes in a simulator, and returns a matrix of assortments
For example, to define the random policy: 
```
def random_policy(simulator):
    """Randomly give a menu of available providers
    
    Arguments:
        patient: Patient object with information on their utilities
        provider_capacities: List of integers, how much space each provider has
        
    Returns: List of integers, 0-1 vector of which providers to show """
    random_matrix = np.random.random((simulator.num_patients,simulator.num_providers))
    random_provider = np.round(random_matrix)
    return random_provider 
```

#### Results
Our results are available in a zip file [here](https://cmu.box.com/s/9oaq5oce9s4q3i0iai9jjwe8cbwv90aj). 