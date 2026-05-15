environment=patient
for seed in 42 43 44 45 46 47 48 49 50
do
    for average_distance in 1 5 10 15 20 25 30
    do
        tmux new-session -d -s patient_ablation_${seed}_distance_${average_distance}
        tmux send-keys -t patient_ablation_${seed}_distance_${average_distance} ENTER
        tmux send-keys -t patient_ablation_${seed}_distance_${average_distance} "source ~/.bashrc" ENTER
        tmux send-keys -t patient_ablation_${seed}_distance_${average_distance} "cd scripts/notebooks" ENTER
        tmux send-keys -t patient_ablation_${seed}_distance_${average_distance} "export PYTHONWARNINGS='ignore'" ENTER
        tmux send-keys -t patient_ablation_${seed}_distance_${average_distance} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
    done

    for num_samples in 1 2 5 10 25
    do
        tmux new-session -d -s patient_ablation_${seed}_samples_${num_samples}
        tmux send-keys -t patient_ablation_${seed}_samples_${num_samples} ENTER
        tmux send-keys -t patient_ablation_${seed}_samples_${num_samples} "source ~/.bashrc" ENTER
        tmux send-keys -t patient_ablation_${seed}_samples_${num_samples} "cd scripts/notebooks" ENTER
        tmux send-keys -t patient_ablation_${seed}_samples_${num_samples} "export PYTHONWARNINGS='ignore'" ENTER
        tmux send-keys -t patient_ablation_${seed}_samples_${num_samples} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
    done

    for capacity_lambda in 1 2 3 4 5
    do
        tmux new-session -d -s patient_ablation_${seed}_capacity_${capacity_lambda}
        tmux send-keys -t patient_ablation_${seed}_capacity_${capacity_lambda} ENTER
        tmux send-keys -t patient_ablation_${seed}_capacity_${capacity_lambda} "source ~/.bashrc" ENTER
        tmux send-keys -t patient_ablation_${seed}_capacity_${capacity_lambda} "cd scripts/notebooks" ENTER
        tmux send-keys -t patient_ablation_${seed}_capacity_${capacity_lambda} "export PYTHONWARNINGS='ignore'" ENTER
        tmux send-keys -t patient_ablation_${seed}_capacity_${capacity_lambda} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
    done

    for num_patients in 800 1200 1600 2000
    do
        tmux new-session -d -s patient_ablation_${seed}_n_${num_patients}
        tmux send-keys -t patient_ablation_${seed}_n_${num_patients} ENTER
        tmux send-keys -t patient_ablation_${seed}_n_${num_patients} "source ~/.bashrc" ENTER
        tmux send-keys -t patient_ablation_${seed}_n_${num_patients} "cd scripts/notebooks" ENTER
        tmux send-keys -t patient_ablation_${seed}_n_${num_patients} "export PYTHONWARNINGS='ignore'" ENTER
        tmux send-keys -t patient_ablation_${seed}_n_${num_patients} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
    done

    for noise in 0.01 0.1 0.2 0.3 0.4 0.5
    do
        tmux new-session -d -s patient_ablation_${seed}_noise_${noise}
        tmux send-keys -t patient_ablation_${seed}_noise_${noise} ENTER
        tmux send-keys -t patient_ablation_${seed}_noise_${noise} "source ~/.bashrc" ENTER
        tmux send-keys -t patient_ablation_${seed}_noise_${noise} "cd scripts/notebooks" ENTER
        tmux send-keys -t patient_ablation_${seed}_noise_${noise} "export PYTHONWARNINGS='ignore'" ENTER
        tmux send-keys -t patient_ablation_${seed}_noise_${noise} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

        tmux new-session -d -s patient_ablation_${seed}_slow_noise_${noise}
        tmux send-keys -t patient_ablation_${seed}_slow_noise_${noise} ENTER
        tmux send-keys -t patient_ablation_${seed}_slow_noise_${noise} "source ~/.bashrc" ENTER
        tmux send-keys -t patient_ablation_${seed}_slow_noise_${noise} "cd scripts/notebooks" ENTER
        tmux send-keys -t patient_ablation_${seed}_slow_noise_${noise} "export PYTHONWARNINGS='ignore'" ENTER
        tmux send-keys -t patient_ablation_${seed}_slow_noise_${noise} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
    done

    for max_shown in 5 10 25 50 100
    do
        tmux new-session -d -s patient_ablation_${seed}_maxshown_${max_shown}
        tmux send-keys -t patient_ablation_${seed}_maxshown_${max_shown} ENTER
        tmux send-keys -t patient_ablation_${seed}_maxshown_${max_shown} "source ~/.bashrc" ENTER
        tmux send-keys -t patient_ablation_${seed}_maxshown_${max_shown} "cd scripts/notebooks" ENTER
        tmux send-keys -t patient_ablation_${seed}_maxshown_${max_shown} "export PYTHONWARNINGS='ignore'" ENTER
        tmux send-keys -t patient_ablation_${seed}_maxshown_${max_shown} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
    done
done

for seed in 42 43 44 45 46 47 48 49 50
do
    echo ${seed}

    for average_distance in 1 5 10 15 20 25 30
    do
        : > runs/logs/error_ablation_${seed}_distance_${average_distance}.txt
        LOGFILE=../../runs/logs/error_ablation_${seed}_distance_${average_distance}.txt
        tmux send-keys -t patient_ablation_${seed}_distance_${average_distance} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --average_distance ${average_distance} --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ${LOGFILE} 2>&1"  ENTER
    done

    for num_samples in 1 2 5 10 25
    do
        : > runs/logs/error_ablation_${seed}_samples_${num_samples}.txt
        LOGFILE=../../runs/logs/error_ablation_${seed}_samples_${num_samples}.txt
        tmux send-keys -t patient_ablation_${seed}_samples_${num_samples} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations --num_samples ${num_samples} >> ${LOGFILE} 2>&1"  ENTER
    done

    for capacity_lambda in 1 2 3 4 5
    do
        : > runs/logs/error_ablation_${seed}_capacity_${capacity_lambda}.txt
        LOGFILE=../../runs/logs/error_ablation_${seed}_capacity_${capacity_lambda}.txt
        tmux send-keys -t patient_ablation_${seed}_capacity_${capacity_lambda} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity ${capacity_lambda} --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ${LOGFILE} 2>&1"  ENTER
    done

    for num_patients in 800 1200 1600 2000
    do
        : > runs/logs/error_ablation_${seed}_n_${num_patients}.txt
        LOGFILE=../../runs/logs/error_ablation_${seed}_n_${num_patients}.txt
        tmux send-keys -t patient_ablation_${seed}_n_${num_patients} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${num_patients} --n_providers 700 --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ${LOGFILE} 2>&1"  ENTER
    done

    for noise in 0.01 0.1 0.2 0.3 0.4 0.5
    do
        : > runs/logs/error_ablation_${seed}_noise_${noise}.txt
        LOGFILE=../../runs/logs/error_ablation_${seed}_noise_${noise}.txt
        tmux send-keys -t patient_ablation_${seed}_noise_${noise} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --noise ${noise} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ${LOGFILE} 2>&1"  ENTER

        : > runs/logs/error_ablation_${seed}_slow_noise_${noise}.txt
        LOGFILE=../../runs/logs/error_ablation_${seed}_slow_noise_${noise}.txt
        tmux send-keys -t patient_ablation_${seed}_slow_noise_${noise} "conda activate ${environment}; python -u all_policies_slow.py --seed ${seed} --n_patients 10 --n_providers 5 --provider_capacity 1 --noise ${noise} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations --max_shown 2 >> ${LOGFILE} 2>&1"  ENTER
        tmux send-keys -t patient_ablation_${seed}_slow_noise_${noise} "conda activate ${environment}; python -u all_policies_slow.py --seed ${seed} --n_patients 20 --n_providers 10 --provider_capacity 1 --noise ${noise} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations --max_shown 3 >> ${LOGFILE} 2>&1"  ENTER
    done

    for max_shown in 5 10 25 50 100
    do
        : > runs/logs/error_ablation_${seed}_maxshown_${max_shown}.txt
        LOGFILE=../../runs/logs/error_ablation_${seed}_maxshown_${max_shown}.txt
        tmux send-keys -t patient_ablation_${seed}_maxshown_${max_shown} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --max_shown ${max_shown} --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ${LOGFILE} 2>&1"  ENTER
    done

done
