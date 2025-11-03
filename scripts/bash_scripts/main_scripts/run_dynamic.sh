: > runs/logs/error_dynamic.txt
LOGFILE=../../runs/logs/error_dynamic.txt

environment=patient
tmux new-session -d -s patient_dynamic
tmux send-keys -t patient_dynamic ENTER 
tmux send-keys -t patient_dynamic "source ~/.bashrc" ENTER
tmux send-keys -t patient_dynamic "cd scripts/notebooks" ENTER
tmux send-keys -t patient_dynamic "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t patient_dynamic "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

N=1225
M=700

for seed in 43 
do 
    # tmux send-keys -t patient_dynamic "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.1 --num_trials 100 --utility_function semi_synthetic_comorbidity --order uniform  --out_folder dynamic --online_arrival >> ${LOGFILE} 2>&1"  ENTER 
    tmux send-keys -t patient_dynamic "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.1 --num_trials 100 --utility_function semi_synthetic_comorbidity --order uniform  --out_folder dynamic --new_provider >> ${LOGFILE} 2>&1"  ENTER 

done 