#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 10_08_egta "python egta_main.py --run_name=10_08_egta_costly --config_files=egta_do_dqn --env=run_env_B_costly"
