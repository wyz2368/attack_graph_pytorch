#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 10_09_qmix "python qmixing_main.py --run_name=10_09_qmix_costly --config_files=eval_qmix_costly --env=run_env_B_costly"
