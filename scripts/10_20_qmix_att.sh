#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 10_20_qmix_att "python qmixing_main.py --run_name=10_20_qmix_att --config_files=eval_qmix --env=run_env_B"
