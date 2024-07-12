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

        # Vary # of patients
        for n_patients in 5 10 20 50 100 
        do 
            n_providers=${n_patients}
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob 0.75 --true_top_choice_prob 0.75 --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 1" ENTER
        done 

        # Vary # of providers
        n_patients=100
        for n_providers in 5 10 25 50 100 200
        do 
            for provider_capacity in 1 2 5
            do 
                tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity ${provider_capacity} --top_choice_prob 0.75 --true_top_choice_prob 0.75 --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 1" ENTER
            done 
        done 

        # Vary choice prob
        n_patients=25
        n_providers=25
        for choice_prob in 0.1 0.25 0.5 0.75 0.9
        do 
            n_providers=${n_patients}
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 1" ENTER
        done 

        # Misalignment/specification in choice prob
        n_patients=25
        n_providers=25
        for choice_prob in 0.1 0.25 0.45 0.55 0.75 0.9
        do 
            n_providers=${n_patients}
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob 0.5 --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 1" ENTER
        done 

        # MNL Model
        for exit_option in 0.1 0.25 0.5 0.75 0.9
        do 
            n_providers=${n_patients}
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob 0.75 --true_top_choice_prob 0.75 --choice_model mnl --exit_option ${exit_option} --out_folder policy_comparison --context_dim 5 --n_trials 1" ENTER
        done 

        # Ability to learn
        n_patients=10
        n_providers=50
        provider_capacity=2
        for n_trials in 1 2 5 10 20
        do 
            n_providers=${n_patients}
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity ${provider_capacity} --top_choice_prob 0.75 --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials ${n_trials}" ENTER
        done 

        # Impact of menu size
        n_patients=50
        n_providers=50
        for menu_size in 3 5 10
        do
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob 0.75 --true_top_choice_prob 0.75 --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 1 --max_menu_size ${menu_size}" ENTER
        done 

    done 
done 
