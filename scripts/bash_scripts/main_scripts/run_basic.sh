: > runs/logs/error_basic.txt
LOGFILE=../../runs/logs/error_basic.txt

environment=patient
for seed in 42 43 44 45 46 47 48 49 50
do 
    tmux new-session -d -s patient_basic_${seed}
    tmux send-keys -t patient_basic_${seed} ENTER 
    tmux send-keys -t patient_basic_${seed} "source ~/.bashrc" ENTER
    tmux send-keys -t patient_basic_${seed} "cd scripts/notebooks" ENTER
    tmux send-keys -t patient_basic_${seed} "export PYTHONWARNINGS='ignore'" ENTER
    tmux send-keys -t patient_basic_${seed} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
done 

N=1225
M=700 

for seed in 42 43 44 45 46 47 48 49 50 
do 
    tmux send-keys -t patient_basic_${seed} "conda activate ${environment}; python -u all_policies.py --verbose --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder baseline >> ${LOGFILE} 2>&1"  ENTER 
    # tmux send-keys -t patient_basic_${seed} "conda activate ${environment}; python -u all_policies_small.py  --verbose  --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder baseline >> ${LOGFILE} 2>&1"  ENTER 
done 