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

        n_patients=2
        choice_prob=0.5
        utility_function=uniform
        
        for n_providers in 2 3 4 5
        do 
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder small --context_dim 5 --n_trials 100 --utility_function ${utility_function}" ENTER
        done 

        n_providers=2
        for n_patients in 2 3 4 5
        do 
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder small --context_dim 5 --n_trials 100 --utility_function ${utility_function}" ENTER
        done   
    done 
done 
