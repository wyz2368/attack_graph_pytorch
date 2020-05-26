#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 10_18_qmix_substate "python qmixing_main.py --run_name=10_18_qmix_substate --config_files=eval_qmix_def --env=run_env_B"
