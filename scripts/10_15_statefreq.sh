#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 10_15_qmix_statefreq "python qmixing_main.py --run_name=10_15_qmix_statefreq --config_files=eval_qmix_def --env=run_env_B"
