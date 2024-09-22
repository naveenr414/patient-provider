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
        utility_function=uniform
        exit_option=0.5

        for choice_prob in 0.1 0.25 0.5 0.75 0.9
        do 
            for shift in -0.5 -0.25 -0.1 0.1 0.25 0.5
            do 
                top_choice_prob=$(echo "$choice_prob + $shift" | bc)
                
                # Check if top_choice_prob is within [0, 1]
                if (( $(echo "$top_choice_prob >= 0" | bc -l) )) && (( $(echo "$top_choice_prob <= 1" | bc -l) ))
                then
                    tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${top_choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option ${exit_option} --out_folder misspecification --context_dim 5 --n_trials 100 --utility_function ${utility_function}" ENTER
                fi
            done
        done

        n_patients=25
        n_providers=5

        for choice_prob in 0.1 0.25 0.5 0.75 0.9
        do 
            for shift in -0.5 -0.25 -0.1 0.1 0.25 0.5
            do 
                top_choice_prob=$(echo "$choice_prob + $shift" | bc)
                
                # Check if top_choice_prob is within [0, 1]
                if (( $(echo "$top_choice_prob >= 0" | bc -l) )) && (( $(echo "$top_choice_prob <= 1" | bc -l) ))
                then
                    tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${top_choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option ${exit_option} --out_folder misspecification --context_dim 5 --n_trials 100 --utility_function ${utility_function}" ENTER
                fi
            done
        done

        n_patients=25
        n_providers=25
        for choice_prob in 0.1 0.25 0.5 0.75 0.9
        do 
            for exit_option in 0.1 0.25 0.5 0.75
            do 
                tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model mnl --exit_option ${exit_option} --out_folder misspecification --context_dim 5 --n_trials 100 --utility_function ${utility_function}" ENTER
            done 
        done         
    done 
done 
