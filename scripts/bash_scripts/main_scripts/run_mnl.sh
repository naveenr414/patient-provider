environment=patient

for seed in 42 43 44 45 46 47 48 49 50
do
    for gamma in 0.01 0.25 0.5 0.75
    do
        : > runs/logs/error_mnl_${seed}_gamma_${gamma}.txt
        tmux new-session -d -s patient_mnl_${seed}_gamma_${gamma}
        tmux send-keys -t patient_mnl_${seed}_gamma_${gamma} ENTER
        tmux send-keys -t patient_mnl_${seed}_gamma_${gamma} "source ~/.bashrc" ENTER
        tmux send-keys -t patient_mnl_${seed}_gamma_${gamma} "cd scripts/notebooks" ENTER
        tmux send-keys -t patient_mnl_${seed}_gamma_${gamma} "export PYTHONWARNINGS='ignore'" ENTER
        tmux send-keys -t patient_mnl_${seed}_gamma_${gamma} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
    done
done

N=1225
M=700

for seed in 42 43 44 45 46 47 48 49 50
do
    for gamma in 0.01 0.25 0.5 0.75
    do
        LOGFILE=../../runs/logs/error_mnl_${seed}_gamma_${gamma}.txt
        tmux send-keys -t patient_mnl_${seed}_gamma_${gamma} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.25 --gamma ${gamma} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder mnl >> ${LOGFILE} 2>&1"  ENTER
    done
done
