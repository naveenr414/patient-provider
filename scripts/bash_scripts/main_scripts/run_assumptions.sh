#!/bin/bash 
cd scripts/notebooks

for session in 1 2 3
do
    tmux new-session -d -s match_${session}
    tmux send-keys -t match_${session} ENTER 
    tmux send-keys -t match_${session} "cd ~/projects/patient_provider/scripts/notebooks" ENTER

    for start_seed in  42 45 48 51 54
    do 
        seed=$((${session}+${start_seed}))

        n_patients=100
        n_providers=25
        max_menu_size=50
        utility_function=uniform

        for choice_prob in 0.5
        do 
            # tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --assumption_relaxation dynamic_lp" ENTER
        #     tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --order proportional" ENTER 
        #     tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --provider_capacity 2" ENTER
            tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --batch_size 2" ENTER
        done 

        # utility_function=normal
        # choice_prob=0.5
        # for std in 0 0.05 0.1 0.15 0.2 0.25
        # do 
        #     tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function normal --max_menu_size ${max_menu_size} --assumption_relaxation varied_std_${std}" ENTER 
        # done 

        # utility_function=uniform
        # for capacity in 3 4
        # do 
        #     tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --provider_capacity ${capacity}" ENTER
        # done 
        # n_providers=50
        # tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size}" ENTER
        # n_providers=75
        # tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size}" ENTER
        # n_providers=100
        # tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size}" ENTER

        # choice_prob=0.5
        # for shift in 0.0 0.05 0.1 0.2
        # do 
        #     tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --assumption_relaxation misspecified_theta_${shift}" ENTER 
        #     tmux send-keys -t match_${session} "conda activate patient; python all_policies.py --seed ${seed} --n_patients ${n_patients} --n_providers ${n_providers} --provider_capacity 1 --top_choice_prob ${choice_prob} --true_top_choice_prob ${choice_prob} --choice_model uniform_choice --exit_option 0.5 --out_folder assumptions --context_dim 5 --n_trials 100 --utility_function ${utility_function} --max_menu_size ${max_menu_size} --assumption_relaxation varied_p_${shift}" ENTER 
        # done 
    done 
done 