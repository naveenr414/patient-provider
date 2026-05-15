environment=patient
N=1225
M=700

mkdir -p results/misspec results/mnl runs/logs

for seed in 42 43 44 45 46 47 48 49 50
do
    tmux new-session -d -s patient_extra_${seed}
    tmux send-keys -t patient_extra_${seed} ENTER
    tmux send-keys -t patient_extra_${seed} "source ~/.bashrc" ENTER
    tmux send-keys -t patient_extra_${seed} "cd scripts/notebooks" ENTER
    tmux send-keys -t patient_extra_${seed} "export PYTHONWARNINGS='ignore'" ENTER
    tmux send-keys -t patient_extra_${seed} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

    for error in -0.2 -0.1 0.1 0.2
    do
        : > runs/logs/error_misspec_${seed}_error_${error}.txt
        tmux send-keys -t patient_extra_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.25 --policy_noise $(echo "0.25 + ${error}" | bc) --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder misspec >> ../../runs/logs/error_misspec_${seed}_error_${error}.txt 2>&1" ENTER
    done

    for gamma in 0.01 0.25 0.5 0.75
    do
        : > runs/logs/error_mnl_${seed}_gamma_${gamma}.txt
        tmux send-keys -t patient_extra_${seed} "conda activate ${environment}; python -u all_policies.py --seed ${seed} --n_patients ${N} --n_providers ${M} --provider_capacity 1 --noise 0.25 --gamma ${gamma} --num_trials 25 --utility_function semi_synthetic_comorbidity --order uniform --out_folder mnl >> ../../runs/logs/error_mnl_${seed}_gamma_${gamma}.txt 2>&1" ENTER
    done
done
