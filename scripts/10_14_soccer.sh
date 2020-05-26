#!/bin/bash
cd ../attackgraph/soccer/

tmux new-session -d -s 10_14_soccer "python qmixing_main.py --run_name=10_14_soccer --config_files=soccer_qmix --env=run_env_B"
