#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 12_30_mo "python egta_main.py --run_name=12_30_mo --config_files=mo_test --algorithm=mixed-oracle --env=run_env_B"
