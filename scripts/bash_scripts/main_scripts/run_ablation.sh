: > runs/logs/error_ablation.txt
LOGFILE=../../runs/logs/error_ablation.txt

environment=patient
tmux new-session -d -s patient_ablation
tmux send-keys -t patient_ablation ENTER 
tmux send-keys -t patient_ablation "source ~/.bashrc" ENTER
tmux send-keys -t patient_ablation "cd scripts/notebooks" ENTER
tmux send-keys -t patient_ablation "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t patient_ablation "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER


for seed in 43 
do 
    for num_patients in 800 1200 1600 2000 
    do 
        tmux send-keys -t patient_ablation "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${num_patients} --n_providers 700 --provider_capacity 1 --noise 0.1 --num_trials 100 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ${LOGFILE} 2>&1"  ENTER 
    done 

    for noise in 0 0.1 0.2 0.3 0.4 0.5
    do 
        tmux send-keys -t patient_ablation "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --noise ${noise} --num_trials 100 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ${LOGFILE} 2>&1"  ENTER 
    done 

    for average_distance in 1 5 10 15 20 25 30
    do 
        tmux send-keys -t patient_ablation "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients 1225 --n_providers 700 --provider_capacity 1 --average_distance ${average_distance} --noise 0.1 --num_trials 100 --utility_function semi_synthetic_comorbidity --order uniform --out_folder ablations >> ${LOGFILE} 2>&1"  ENTER 
    done 
done 