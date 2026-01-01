: > runs/logs/error_dynamic.txt
LOGFILE=../../runs/logs/error_dynamic.txt

environment=patient
for seed in 42 43 44 45 46 47 48 49 50
do 
    tmux new-session -d -s patient_dynamic_${seed}
    tmux send-keys -t patient_dynamic_${seed} ENTER 
    tmux send-keys -t patient_dynamic_${seed} "source ~/.bashrc" ENTER
    tmux send-keys -t patient_dynamic_${seed} "cd scripts/notebooks" ENTER
    tmux send-keys -t patient_dynamic_${seed} "export PYTHONWARNINGS='ignore'" ENTER
    tmux send-keys -t patient_dynamic_${seed} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
done 

N=1225
M=700

for seed in 42 43 44 45 46 47 48 49 50 
do 
    tmux send-keys -t patient_dynamic_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform  --out_folder dynamic --new_provider >> ${LOGFILE} 2>&1"  ENTER 

    for online_scale in 0.1 0.25 0.5 0.75 0.9 1.0
    do 
        tmux send-keys -t patient_dynamic_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.25 --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform  --out_folder dynamic --online_scale ${online_scale} --online_arrival  >> ${LOGFILE} 2>&1"  ENTER 
    done 
done 