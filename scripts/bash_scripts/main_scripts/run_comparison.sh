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

        n_patients=25
        n_providers=25
        max_menu_size=50

        for utility_function in uniform normal 
        do 
            for choice_prob in 0.1 0.25 0.5 0.75 0.9
            do 
                tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size}" ENTER
            done 
        done 

        utility_function=uniform
        for choice_prob in 0.1 0.25 0.5 0.75 0.9
        do 
            for fairness_weight in 0.1 0.25 0.5 0.75 0.9
            do 
                tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --fairness_weight ${fairness_weight}" ENTER
            done 
        done    
    done 
done 
