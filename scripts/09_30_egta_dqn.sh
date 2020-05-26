#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 09_30_egta_dqn "python egta_main.py --run_name=09_30_egta_dqn --config_files=egta_do_dqn --env=run_env_B"
