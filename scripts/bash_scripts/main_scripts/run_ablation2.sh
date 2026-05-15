environment=patient

mkdir -p results/ablations runs/logs

for seed in 42 43 44 45 46 47 48 49 50
do
    tmux new-session -d -s patient_ablation2_${seed}
    tmux send-keys -t patient_ablation2_${seed} ENTER
    tmux send-keys -t patient_ablation2_${seed} "source ~/.bashrc" ENTER
    tmux send-keys -t patient_ablation2_${seed} "cd scripts/notebooks" ENTER
    tmux send-keys -t patient_ablation2_${seed} "export PYTHONWARNINGS='ignore'" ENTER
    tmux send-keys -t patient_ablation2_${seed} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

    for noise in 0.01 0.1 0.2 0.3 0.4 0.5
    do
        : > runs/logs/error_ablation_${seed}_noise_${noise}.txt
        tmux send-keys -t patient_ablation2_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --noise ${noise} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ../../runs/logs/error_ablation_${seed}_noise_${noise}.txt 2>&1" ENTER
    done

    for max_shown in 5 10 25 50 100
    do
        : > runs/logs/error_ablation_${seed}_maxshown_${max_shown}.txt
        tmux send-keys -t patient_ablation2_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --max_shown ${max_shown} --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ../../runs/logs/error_ablation_${seed}_maxshown_${max_shown}.txt 2>&1" ENTER
    done

    for eps in 0.01 0.2 0.4
    do
        for k in 5 10 25
        do
            : > runs/logs/error_ablation_${seed}_eps_${eps}_k_${k}.txt
            tmux send-keys -t patient_ablation2_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --noise ${eps} --max_shown ${k} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ../../runs/logs/error_ablation_${seed}_eps_${eps}_k_${k}.txt 2>&1" ENTER
        done
    done

    for utility_function in uniform normal
    do
        : > runs/logs/error_ablation_${seed}_dist_${utility_function}.txt
        tmux send-keys -t patient_ablation2_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function ${utility_function} --order uniform --out_folder ablations >> ../../runs/logs/error_ablation_${seed}_dist_${utility_function}.txt 2>&1" ENTER
    done
done
