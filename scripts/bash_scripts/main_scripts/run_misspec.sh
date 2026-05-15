environment=patient

# True epsilon is fixed at 0.25; policy_noise values correspond to errors of -0.2, -0.1, +0.1, +0.2
for seed in 42 43 44 45 46 47 48 49 50
do
    for policy_noise in 0.05 0.15 0.35 0.45
    do
        : > runs/logs/error_misspec_${seed}_pnoise_${policy_noise}.txt
        tmux new-session -d -s patient_misspec_${seed}_pnoise_${policy_noise}
        tmux send-keys -t patient_misspec_${seed}_pnoise_${policy_noise} ENTER
        tmux send-keys -t patient_misspec_${seed}_pnoise_${policy_noise} "source ~/.bashrc" ENTER
        tmux send-keys -t patient_misspec_${seed}_pnoise_${policy_noise} "cd scripts/notebooks" ENTER
        tmux send-keys -t patient_misspec_${seed}_pnoise_${policy_noise} "export PYTHONWARNINGS='ignore'" ENTER
        tmux send-keys -t patient_misspec_${seed}_pnoise_${policy_noise} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
    done
done

N=1225
M=700

for seed in 42 43 44 45 46 47 48 49 50
do
    for policy_noise in 0.05 0.15 0.35 0.45
    do
        LOGFILE=../../runs/logs/error_misspec_${seed}_pnoise_${policy_noise}.txt
        tmux send-keys -t patient_misspec_${seed}_pnoise_${policy_noise} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.25 --policy_noise ${policy_noise} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder misspec >> ${LOGFILE} 2>&1"  ENTER
    done
done
