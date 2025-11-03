: > runs/logs/error_basic.txt
LOGFILE=../../runs/logs/error_basic.txt

environment=patient
tmux new-session -d -s patient_basic
tmux send-keys -t patient_basic ENTER 
tmux send-keys -t patient_basic "source ~/.bashrc" ENTER
tmux send-keys -t patient_basic "cd scripts/notebooks" ENTER
tmux send-keys -t patient_basic "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t patient_basic "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

N=300
M=150

for seed in 43 
do 
    tmux send-keys -t patient_basic "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.1 --num_trials 100 --utility_function semi_synthetic_comorbidity --order uniform --out_folder baseline >> ${LOGFILE} 2>&1"  ENTER 
done 