: > runs/logs/error_ablation.txt
LOGFILE=../../runs/logs/error_ablation.txt

environment=patient
for seed in 42 43 44 45 46 47 48 49 50
do 
    tmux new-session -d -s patient_ablation_${seed}
    tmux send-keys -t patient_ablation_${seed} ENTER 
    tmux send-keys -t patient_ablation_${seed} "source ~/.bashrc" ENTER
    tmux send-keys -t patient_ablation_${seed} "cd scripts/notebooks" ENTER
    tmux send-keys -t patient_ablation_${seed} "export PYTHONWARNINGS='ignore'" ENTER
    tmux send-keys -t patient_ablation_${seed} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
done 

for seed in 42 43 44 45 46 47 48 49 50 
do 
    echo ${seed}
    # for num_patients in 800 1200 1600 2000 
    # do 
    #     tmux send-keys -t patient_ablation_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${num_patients} --n_providers 700 --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ${LOGFILE} 2>&1"  ENTER 
    # done 

    # for noise in 0.01 0.1 0.2 0.3 0.4 0.5
    # do 
    #     tmux send-keys -t patient_ablation_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --noise ${noise} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ${LOGFILE} 2>&1"  ENTER 
    # done

    # for noise in 0.01 0.1 0.2 0.3 0.4 0.5
    # do 
    #     tmux send-keys -t patient_ablation_${seed} "conda activate ${environment}; python -u all_policies_slow.py --seed ${seed} --n_patients 10 --n_providers 5 --provider_capacity 1 --noise ${noise} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations --max_shown 2 >> ${LOGFILE} 2>&1"  ENTER 
    # done 

    # for noise in 0.01 0.1 0.2 0.3 0.4 0.5
    # do 
    #     tmux send-keys -t patient_ablation_${seed} "conda activate ${environment}; python -u all_policies_slow.py --seed ${seed} --n_patients 20 --n_providers 10 --provider_capacity 1 --noise ${noise} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations --max_shown 3 >> ${LOGFILE} 2>&1"  ENTER 
    # done 

    # for max_shown in 5 10 25 50 100
    # do 
    #     tmux send-keys -t patient_ablation_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --max_shown ${max_shown} --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ${LOGFILE} 2>&1"  ENTER 
    # done  

    for average_distance in 1 5 10 15 20 25 30
    do 
        tmux send-keys -t patient_ablation_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --average_distance ${average_distance} --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ${LOGFILE} 2>&1"  ENTER 
    done 

    for num_samples in 1 2 5 10 25
    do 
        tmux send-keys -t patient_ablation_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations --num_samples ${num_samples} >> ${LOGFILE} 2>&1"  ENTER 
    done 

done 