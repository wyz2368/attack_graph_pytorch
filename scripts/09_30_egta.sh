#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 09_30_egta_state_enc "python egta_main.py --run_name=09_30_egta_state_enc --config_files=egta_do_state_enc --env=run_env_B"
