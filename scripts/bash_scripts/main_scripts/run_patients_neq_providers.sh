#!/bin/bash 

cd scripts/notebooks

for seed in $(seq 43 57); 
do 
    echo ${seed}

    n_providers=25
    choice_prob=0.5
    max_menu_size=500
    utility_function=uniform
    
    for n_patients in 25 50 100 150 200
    do 
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder providers_patients --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size}
    done 

    utility_function=normal
    
    for n_patients in 25 50 100 150 200
    do 
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder providers_patients --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size}
    done 

    max_menu_size=50
    n_providers=10
    for n_patients in 70 80 # 10 20 30 40 50 60
    do 
        for utility_function in uniform 
        do 
            for choice_prob in 0.1 0.3 0.5 0.7 0.9
            do 
                python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder providers_patients --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size}
            done 
        done 
    done

    max_menu_size=50
    n_providers=10
    for n_patients in 10 20 30 40 50 60 70 80
    do 
        for utility_function in normal 
        do 
            for choice_prob in 0.1 0.3 0.5 0.7 0.9
            do 
                python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder providers_patients --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size}
            done 
        done 
    done
done 
