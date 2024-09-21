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

        # Vary theta distribution
        n_patients=25
        n_providers=25
        for choice_prob in 0.1 0.25 0.5 0.75 0.9
        do 
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 10 --utility_function normal" ENTER
        done 

        # for n_patients in 5 10 25 50 #100 
        # do 
        #     n_providers=${n_patients}
        #     tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob 0.5 --true_top_choice_prob 0.5 --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 10 --utility_function uniform" ENTER
        # done 

        # Vary # of providers
        n_patients=25
        for n_providers in 5 10 25 50 #100
        do 
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob 0.5 --true_top_choice_prob 0.5 --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 10 --utility_function uniform" ENTER
        done 

        # Vary # of Patients
        n_providers=25
        for n_patients in 5 10 25 50 #100
        do 
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob 0.5 --true_top_choice_prob 0.5 --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 10 --utility_function uniform" ENTER
        done 

        # # Vary p
        # n_patients=25
        # n_providers=25
        # for choice_prob in 0.1 0.25 0.5 0.75 0.9
        # do 
        #     tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder policy_comparison --context_dim 5 --n_trials 10 --utility_function uniform" ENTER
        # done 
    done 
done 
