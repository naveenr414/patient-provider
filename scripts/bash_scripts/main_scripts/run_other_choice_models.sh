#!/bin/bash 
cd scripts/notebooks

for seed in $(seq 43 57); 
do 
    echo ${seed}

    n_patients=25
    n_providers=25
    utility_function=uniform
    exit_option=0.5
    max_menu_size=50

    for choice_prob in 0.1 0.25 0.5 0.75 0.9
    do 
        for exit_option in 0.1 0.25 0.5 0.75
        do 
            python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model threshold --exit_option ${exit_option} --out_folder misspecification --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size}
        done 
    done 

    for choice_prob in 0.1 0.25 0.5 0.75 0.9
    do 
        for exit_option in 0.1 0.25 0.5 0.75
        do 
            python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model mnl --exit_option ${exit_option} --out_folder misspecification --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size}
        done 
    done 

    n_patients=25
    n_providers=25
    utility_function=uniform
    exit_option=0.5
    for choice_prob in 0.25 0.5 0.75
    do 
        for shift in -0.25 -0.1 0 0.1 0.25
        do 
            true_choice_prob=$(echo "$choice_prob + $shift" | bc)
            python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${true_choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder misspecification --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size}
        done 
    done 
done 
