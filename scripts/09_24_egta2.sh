#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 09_24_egta2_dqn "python egta_main.py --run_name=09_24_egta2_dqn --config_files=egta_do_dqn --env=run_env_B"
tmux new-session -d -s 09_24_egta2_state_enc "python egta_main.py --run_name=09_24_egta2_state_enc --config_files=egta_do_state_enc --env=run_env_B"
