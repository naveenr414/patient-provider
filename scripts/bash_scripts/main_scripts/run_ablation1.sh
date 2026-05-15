environment=patient

mkdir -p results/ablations runs/logs

for seed in 42 43 44 45 46 47 48 49 50
do
    tmux new-session -d -s patient_ablation1_${seed}
    tmux send-keys -t patient_ablation1_${seed} ENTER
    tmux send-keys -t patient_ablation1_${seed} "source ~/.bashrc" ENTER
    tmux send-keys -t patient_ablation1_${seed} "cd scripts/notebooks" ENTER
    tmux send-keys -t patient_ablation1_${seed} "export PYTHONWARNINGS='ignore'" ENTER
    tmux send-keys -t patient_ablation1_${seed} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

    for average_distance in 1 5 10 15 20 25 30
    do
        : > runs/logs/error_ablation_${seed}_distance_${average_distance}.txt
        tmux send-keys -t patient_ablation1_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --average_distance ${average_distance} --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ../../runs/logs/error_ablation_${seed}_distance_${average_distance}.txt 2>&1" ENTER
    done

    for num_samples in 1 2 5 10 25
    do
        : > runs/logs/error_ablation_${seed}_samples_${num_samples}.txt
        tmux send-keys -t patient_ablation1_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations --num_samples ${num_samples} >> ../../runs/logs/error_ablation_${seed}_samples_${num_samples}.txt 2>&1" ENTER
    done

    for capacity_lambda in 1 2 3 4 5
    do
        : > runs/logs/error_ablation_${seed}_capacity_${capacity_lambda}.txt
        tmux send-keys -t patient_ablation1_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity ${capacity_lambda} --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ../../runs/logs/error_ablation_${seed}_capacity_${capacity_lambda}.txt 2>&1" ENTER
    done

    for num_patients in 800 1200 1600 2000
    do
        : > runs/logs/error_ablation_${seed}_n_${num_patients}.txt
        tmux send-keys -t patient_ablation1_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${num_patients} --n_providers 700 --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ../../runs/logs/error_ablation_${seed}_n_${num_patients}.txt 2>&1" ENTER
    done

    for noise in 0.01 0.1 0.2 0.3 0.4 0.5
    do
        : > runs/logs/error_ablation_${seed}_slow_noise_${noise}.txt
        tmux send-keys -t patient_ablation1_${seed} "conda activate ${environment}; python -u all_policies_slow.py --seed ${seed} --n_patients 10 --n_providers 5 --provider_capacity 1 --noise ${noise} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations --max_shown 2 >> ../../runs/logs/error_ablation_${seed}_slow_noise_${noise}.txt 2>&1" ENTER
        tmux send-keys -t patient_ablation1_${seed} "conda activate ${environment}; python -u all_policies_slow.py --seed ${seed} --n_patients 20 --n_providers 10 --provider_capacity 1 --noise ${noise} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations --max_shown 3 >> ../../runs/logs/error_ablation_${seed}_slow_noise_${noise}.txt 2>&1" ENTER
    done
done
