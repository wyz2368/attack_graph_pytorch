#!/bin/bash
cd ../attackgraph/

# EGTA: Generate more policies.
tmux new-session -d -s 10_21_egta "python egta_main.py --run_name=10_21_egta --config_files=egta_do_dqn --env=run_env_B"


# QMix: Train an opponent classifier for Q-Mixing.
tmux new-session -d -s 10_21_classifier "python train_classifier_main.py --run_name=10_21_classifier --config_files=qmix_classifier"


# Policy Distillation.
tmux new-session -d -s 10_21_distill "python policy_distillation_main.py --run_name=10_21_distill --config_files=policy_distill"


# Soccer: Generate beta-final run results to be averaged together.
cd soccer/
tmux new-session -d -s 10_21_soccer_0 'python qmixing_main.py --run_name=10_21_soccer_0 --config_files=soccer_qmix --env=run_env_B'
tmux new-session -d -s 10_21_soccer_1 'python qmixing_main.py --run_name=10_21_soccer_1 --config_files=soccer_qmix --env=run_env_B'
tmux new-session -d -s 10_21_soccer_2 'python qmixing_main.py --run_name=10_21_soccer_2 --config_files=soccer_qmix --env=run_env_B'
tmux new-session -d -s 10_21_soccer_3 'python qmixing_main.py --run_name=10_21_soccer_3 --config_files=soccer_qmix --env=run_env_B'
tmux new-session -d -s 10_21_soccer_4 'python qmixing_main.py --run_name=10_21_soccer_4 --config_files=soccer_qmix --env=run_env_B'
