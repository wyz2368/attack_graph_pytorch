#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 10_07_qmix "python qmixing_main.py --run_name=10_07_qmix --config_files=eval_qmix_def --env=run_env_B"
