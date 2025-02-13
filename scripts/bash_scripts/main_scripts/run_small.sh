#!/bin/bash 

cd scripts/notebooks

for seed in $(seq 43 57); 
do 
    echo ${seed}

    n_patients=2
    choice_prob=0.5
    utility_function=uniform
    
    for n_providers in 2 3 4 5
    do 
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder small --context_dim 5 --n_trials 100 --utility_function ${utility_function}
    done 

    n_providers=2
    for n_patients in 2 3 4 5
    do 
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder small --context_dim 5 --n_trials 100 --utility_function ${utility_function}
    done   
done 
