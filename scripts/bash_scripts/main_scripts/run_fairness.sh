: > runs/logs/error_fairness.txt
LOGFILE=../../runs/logs/error_fairness.txt

environment=patient
for seed in 42 43 44 45 46 47 48 49 50 
do 
    tmux new-session -d -s patient_fairness_${seed}
    tmux send-keys -t patient_fairness_${seed} ENTER 
    tmux send-keys -t patient_fairness_${seed} "source ~/.bashrc" ENTER
    tmux send-keys -t patient_fairness_${seed} "cd scripts/notebooks" ENTER
    tmux send-keys -t patient_fairness_${seed} "export PYTHONWARNINGS='ignore'" ENTER
    tmux send-keys -t patient_fairness_${seed} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
done 
N=1225
M=700

for seed in 42 43 44 45 46 47 48 49 50  
do 
    for max_diff in 0.01 0.02 0.05 0.1 0.2
    do 
        tmux send-keys -t patient_fairness_${seed} "conda activate ${environment}; python -u all_policies_small.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --fairness_constraint ${max_diff} --out_folder fairness >> ${LOGFILE} 2>&1"  ENTER 
        tmux send-keys -t patient_fairness_${seed} "conda activate ${environment}; python -u all_policies_small.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --fairness_constraint ${max_diff} --out_folder fairness >> ${LOGFILE} 2>&1"  ENTER 

    done 
done 