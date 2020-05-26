#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 10_14_qmix_hard "python qmixing_main.py --run_name=10_14_qmix_hard --config_files=eval_qmix_def --env=run_env_B"
