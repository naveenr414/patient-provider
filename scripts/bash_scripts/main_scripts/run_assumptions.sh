#!/bin/bash 
cd scripts/notebooks

for seed in $(seq 43 57); 
do 
    echo ${seed}

    n_patients=25
    n_providers=25
    max_menu_size=50
    utility_function=uniform

    for choice_prob in 0.1 0.25 0.5 0.75 0.9
    do 
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --assumption_relaxation dynamic_lp
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --order proportional
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --provider_capacity 2
    done 

    choice_prob=0.5
    for shift in 0.0 0.05 0.1 0.2
    do 
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --assumption_relaxation misspecified_theta_${shift}
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --assumption_relaxation varied_p_${shift}
    done 
done 