#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 10_08_qmix "python qmixing_main.py --run_name=10_08_qmix --config_files=eval_qmix_def --env=run_env_B"