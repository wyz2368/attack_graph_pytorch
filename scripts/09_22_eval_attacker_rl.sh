#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 09_22_eval_0 "python evaluate_learning_algorithm_main.py --run_name=09_22_eval_0 --config_files=eval_local_test --env=run_env_B \
     --config_overrides='DQN.lr=5e-5' \
     --config_overrides='Learner.buffer_size=50000'"
tmux new-session -d -s 09_22_eval_1 "python evaluate_learning_algorithm_main.py --run_name=09_22_eval_1 --config_files=eval_local_test --env=run_env_B \
     --config_overrides='DQN.lr=5e-5' \
     --config_overrides='Learner.target_network_update_freq=1000'"
tmux new-session -d -s 09_22_eval_2 "python evaluate_learning_algorithm_main.py --run_name=09_22_eval_2 --config_files=eval_local_test --env=run_env_B \
     --config_overrides='DQN.lr=5e-5' \
     --config_overrides='Learner.buffer_size=50000' \
     --config_overrides='Learner.target_network_update_freq=1000'"
tmux new-session -d -s 09_22_eval_3 "python evaluate_learning_algorithm_main.py --run_name=09_22_eval_3 --config_files=eval_local_test --env=run_env_B \
     --config_overrides='DQN.lr=5e-4' \
     --config_overrides='Learner.buffer_size=50000'"
tmux new-session -d -s 09_22_eval_4 "python evaluate_learning_algorithm_main.py --run_name=09_22_eval_4 --config_files=eval_local_test --env=run_env_B \
     --config_overrides='DQN.lr=5e-4' \
     --config_overrides='Learner.target_network_update_freq=1000'"
tmux new-session -d -s 09_22_eval_5 "python evaluate_learning_algorithm_main.py --run_name=09_22_eval_5 --config_files=eval_local_test --env=run_env_B \
     --config_overrides='DQN.lr=5e-4' \
     --config_overrides='Learner.buffer_size=50000' \
     --config_overrides='Learner.target_network_update_freq=1000'"
