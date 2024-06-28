#!/bin/bash 

seed=42

echo "N Patients"
for n_patients in 5 
do 
    n_providers=${n_patients}
    conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob 0.75 --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 1
done 

echo "N Providers"
n_patients=100
for n_providers in 5
do 
    for provider_capacity in 1
    do 
        conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity ${provider_capacity} --top_choice_prob 0.75 --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 1
    done 
done 

n_patients=25
n_providers=25
for choice_prob in 0.1
do 
    n_providers=${n_patients}
    conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 1
done 

for exit_option in 0.1
do 
    n_providers=${n_patients}
    conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob 0.75 --choice_model mnl --exit_option ${exit_option} --out_folder policy_comparison --context_dim 5 --n_trials 1
done 

n_patients=10
n_providers=50
provider_capacity=2
for n_trials in 1
do 
    n_providers=${n_patients}
    conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity ${provider_capacity} --top_choice_prob 0.75 --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials ${n_trials}
done 