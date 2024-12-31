#!/bin/bash 

for session in 1 2 3
do
    tmux new-session -d -s match_${session}
    tmux send-keys -t match_${session} ENTER 
    tmux send-keys -t match_${session} "cd ~/projects/patient_provider/scripts/notebooks" ENTER

    for start_seed in 42 45 48 51 54
    do 
        seed=$((${session}+${start_seed}))
        echo ${seed}

        utility_function=semi_synthetic
        choice_model=uniform_choice
        top_choice_prob=0.75
        true_top_choice_prob=0.75
        exit_option=0.5
        max_menu_size=1000

        # TODO: Change this to 1225, 700
        n_patients=205
        n_providers=140

        tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${top_choice_prob} --true_top_choice_prob ${true_top_choice_prob} --choice_model ${choice_model} --exit_option ${exit_option} --out_folder semi_synthetic --utility_function ${utility_function} --max_menu_size ${max_menu_size}" ENTER

        for batch_size in 2 3 4 5
        do
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${top_choice_prob} --true_top_choice_prob ${true_top_choice_prob} --choice_model ${choice_model} --exit_option ${exit_option} --out_folder semi_synthetic --utility_function ${utility_function} --max_menu_size ${max_menu_size} --batch ${batch_size}" ENTER
        done 

        for max_menu_size in 5 10 20 40
        do 
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${top_choice_prob} --true_top_choice_prob ${true_top_choice_prob} --choice_model ${choice_model} --exit_option ${exit_option} --out_folder semi_synthetic --utility_function ${utility_function} --max_menu_size ${max_menu_size}" ENTER
        done 

    done 
done 
