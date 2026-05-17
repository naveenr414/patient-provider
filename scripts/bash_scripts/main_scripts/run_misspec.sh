environment=patient
N=1225
M=700
for seed in 42 43 44 45 46 47 48 49 50
do
for policy_noise in 0.05 0.15 0.35 0.45
do
        pnoise_tag=$(echo $policy_noise | tr '.' '_')
        SESSION="patient_misspec_${seed}_pnoise_${pnoise_tag}"
        : > runs/logs/error_misspec_${seed}_pnoise_${policy_noise}.txt
        tmux new-session -d -s $SESSION
        tmux send-keys -t $SESSION ENTER
        tmux send-keys -t $SESSION "source ~/.bashrc" ENTER
        tmux send-keys -t $SESSION "cd scripts/notebooks" ENTER
        tmux send-keys -t $SESSION "export PYTHONWARNINGS='ignore'" ENTER
        tmux send-keys -t $SESSION "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
done
done
N=1225
M=700
for seed in 42 43 44 45 46 47 48 49 50
do
for policy_noise in 0.05 0.15 0.35 0.45
do
        pnoise_tag=$(echo $policy_noise | tr '.' '_')
        SESSION="patient_misspec_${seed}_pnoise_${pnoise_tag}"
        LOGFILE=../../runs/logs/error_misspec_${seed}_pnoise_${policy_noise}.txt
        tmux send-keys -t $SESSION "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.25 --policy_noise ${policy_noise} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder misspec >> ${LOGFILE} 2>&1"  ENTER
done
done