#!/bin/bash

tmux new-session -d -s patient
tmux send-keys -t patient ENTER 
tmux send-keys -t patient "cd ~/projects/patient_provider/scripts/notebooks" ENTER

tmux send-keys -t patient "conda activate patient; python online_matching_baseline.py --n_patients 10 --n_providers 10 --provider_capacity 1" ENTER
tmux send-keys -t patient "conda activate patient; python online_matching_baseline.py --n_patients 10 --n_providers 10 --provider_capacity 2" ENTER
tmux send-keys -t patient "conda activate patient; python online_matching_baseline.py --n_patients 10 --n_providers 10 --provider_capacity 4" ENTER

tmux send-keys -t patient "conda activate patient; python online_matching_baseline.py --n_patients 40 --n_providers 20 --provider_capacity 1" ENTER
tmux send-keys -t patient "conda activate patient; python online_matching_baseline.py --n_patients 40 --n_providers 20 --provider_capacity 2" ENTER
tmux send-keys -t patient "conda activate patient; python online_matching_baseline.py --n_patients 40 --n_providers 20 --provider_capacity 4" ENTER

tmux send-keys -t patient "conda activate patient; python online_matching_baseline.py --n_patients 20 --n_providers 40 --provider_capacity 1" ENTER
tmux send-keys -t patient "conda activate patient; python online_matching_baseline.py --n_patients 20 --n_providers 40 --provider_capacity 2" ENTER
tmux send-keys -t patient "conda activate patient; python online_matching_baseline.py --n_patients 20 --n_providers 40 --provider_capacity 4" ENTER
