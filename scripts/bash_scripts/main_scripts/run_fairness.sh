: > runs/logs/error_fairness.txt
LOGFILE=../../runs/logs/error_fairness.txt

environment=patient
tmux new-session -d -s patient_fairness
tmux send-keys -t patient_fairness ENTER 
tmux send-keys -t patient_fairness "source ~/.bashrc" ENTER
tmux send-keys -t patient_fairness "cd scripts/notebooks" ENTER
tmux send-keys -t patient_fairness "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t patient_fairness "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

N=1225
M=700

for seed in 43 
do 
    for max_diff in 0.01 0.05 0.1 0.2 0.3 0.4 0.5
    do 
        tmux send-keys -t patient_fairness "conda activate ${environment}; python -u all_policies_small.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.1 --num_trials 100 --utility_function semi_synthetic_comorbidity --order uniform --fairness_constraint ${max_diff} --out_folder fairness >> ${LOGFILE} 2>&1"  ENTER 
    done 
done 