#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 10_01_qmix "python qmixing_main.py --run_name=10_01_qmix --config_files=eval_qmix --env=run_env_B"
