#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 09_27_state_enc_0 $'python evaluate_learning_algorithm_main.py --run_name=09_27_state_enc_0 --config_files=eval_dqn_state_enc --env=run_env_B \
     --config_overrides="DQNEncDec.q_lr=5e-6" \
     --config_overrides="DQNEncDec.encoder_lr=5e-6"'
tmux new-session -d -s 09_27_state_enc_1 $'python evaluate_learning_algorithm_main.py --run_name=09_27_state_enc_1 --config_files=eval_dqn_state_enc --env=run_env_B \
     --config_overrides="DQNEncDec.q_lr=5e-4" \
     --config_overrides="DQNEncDec.encoder_lr=5e-6"'
tmux new-session -d -s 09_27_state_enc_2 $'python evaluate_learning_algorithm_main.py --run_name=09_27_state_enc_2 --config_files=eval_dqn_state_enc --env=run_env_B \
     --config_overrides="DQNEncDec.q_lr=5e-4" \
     --config_overrides="DQNEncDec.encoder_lr=5e-4"'
tmux new-session -d -s 09_27_state_enc_3 $'python evaluate_learning_algorithm_main.py --run_name=09_27_state_enc_3 --config_files=eval_dqn_state_enc --env=run_env_B \
     --config_overrides="DQNEncDec.q_lr=5e-4" \
     --config_overrides="DQNEncDec.encoder_lr=5e-4" \
     --config_overrides="DQNEncDec.grad_norm_clipping=None"'
tmux new-session -d -s 09_27_state_enc_4 $'python evaluate_learning_algorithm_main.py --run_name=09_27_state_enc_4 --config_files=eval_dqn_state_enc --env=run_env_B \
     --config_overrides="DQNEncDec.q_lr=5e-4" \
     --config_overrides="DQNEncDec.encoder_lr=5e-4" \
     --config_overrides="DQNEncDec.grad_norm_clipping=1"'
