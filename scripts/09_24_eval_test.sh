#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 09_24_eval_test $'python evaluate_learning_algorithm_main.py --run_name=09_24_eval_test --config_files=eval_dqn_state_enc --env=run_env_B \
     --config_overrides="evaluate_algorithm.train_attacker=True" \
     --config_overrides="DQNEncDec.state_encoder_load_path=\'/home/mxsmith/projects/attack_graph/results/09_23_egta_state_enc/attacker_policies/att_str_epoch2.pkl\'"'
