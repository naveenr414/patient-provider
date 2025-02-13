#!/bin/bash 
cd scripts/notebooks

for seed in $(seq 43 57); 
do 
    echo ${seed}

    choice_model=uniform_choice
    top_choice_prob=0.75
    true_top_choice_prob=0.75
    exit_option=0.5
    max_menu_size=2000

    n_patients=1225
    n_providers=700

    for utility_function in semi_synthetic_comorbidity 
    do 
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${top_choice_prob} --true_top_choice_prob ${true_top_choice_prob} --choice_model ${choice_model} --exit_option ${exit_option} --out_folder semi_synthetic --utility_function ${utility_function} --max_menu_size ${max_menu_size}  --batch 1
    done

    utility_function=semi_synthetic_comorbidity

    for fairness_weight in 0.1 0.25 0.5 0.75 0.9
    do 
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${top_choice_prob} --true_top_choice_prob ${true_top_choice_prob} --choice_model ${choice_model} --exit_option ${exit_option} --out_folder semi_synthetic --utility_function ${utility_function} --max_menu_size ${max_menu_size} --fairness_weight ${fairness_weight}
    done 

    for batch_size in 2 3 4 5
    do
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${top_choice_prob} --true_top_choice_prob ${true_top_choice_prob} --choice_model ${choice_model} --exit_option ${exit_option} --out_folder semi_synthetic --utility_function ${utility_function} --max_menu_size ${max_menu_size} --batch ${batch_size}
    done 

    for max_menu_size in 5 10 20 40
    do 
        python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${top_choice_prob} --true_top_choice_prob ${true_top_choice_prob} --choice_model ${choice_model} --exit_option ${exit_option} --out_folder semi_synthetic --utility_function ${utility_function} --max_menu_size ${max_menu_size}
    done 
done 
