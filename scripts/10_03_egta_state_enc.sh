#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 10_03_egta_state_enc "python egta_main.py --run_name=10_03_egta_state_enc --config_files=egta_do_state_enc --env=run_env_B"
