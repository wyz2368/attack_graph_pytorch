#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 09_30_qmix "python qmixing_main.py --run_name=09_30_qmix --config_files=eval_qmix --env=run_env_B"
