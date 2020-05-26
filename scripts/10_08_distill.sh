#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 10_08_distill "python policy_distillation_main.py --run_name=10_08_distill --config_files=policy_distill --env=run_env_B"
