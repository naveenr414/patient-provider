#!/bin/bash 

# Synthetic Scripts
bash ../bash_scripts/main_scripts/run_small.sh
bash ../bash_scripts/main_scripts/run_comparison.sh
bash ../bash_scripts/main_scripts/run_other_choice_models.sh
bash ../bash_scripts/main_scripts/run_patients_neq_providers.sh
bash ../bash_scripts/main_scripts/run_misspecification.sh

# Semi-Synthetic Scripts
# bash ../bash_scripts/main_scripts/run_semi_synthetic.sh

# Ablation
# bash ../bash_scripts/main_scripts/run_multi_iteration.sh
# bash ../bash_scripts/main_scripts/run_ordering.sh
