#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 09_24_eval_dqn "python evaluate_learning_algorithm_main.py --run_name=09_24_eval_dqn --config_files=eval_dqn --env=run_env_B"
tmux new-session -d -s 09_24_eval_state_enc "python evaluate_learning_algorithm_main.py --run_name=09_24_eval_state_enc --config_files=eval_dqn_state_enc --env=run_env_B"
