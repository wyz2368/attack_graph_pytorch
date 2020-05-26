#!/bin/bash
cd ../attackgraph/

tmux new-session -d -s 10_15_qmix_explore_0 'python qmixing_main.py --run_name=10_15_qmix_explore_0 --config_files=eval_qmix_def --env=run_env_B \
     --config_overrides="pure/Learner.total_timesteps = 700000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.3"'
tmux new-session -d -s 10_15_qmix_explore_1 'python qmixing_main.py --run_name=10_15_qmix_explore_1 --config_files=eval_qmix_def --env=run_env_B \
     --config_overrides="pure/Learner.total_timesteps = 500000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.3" \
     --config_overrides="pure/DQN.lr = 5e-4"'
tmux new-session -d -s 10_15_qmix_explore_2 'python qmixing_main.py --run_name=10_15_qmix_explore_2 --config_files=eval_qmix_def --env=run_env_B \
     --config_overrides="pure/Learner.total_timesteps = 500000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.2" \
     --config_overrides="pure/DQN.lr = 5e-4"'
tmux new-session -d -s 10_15_qmix_explore_3 'python qmixing_main.py --run_name=10_15_qmix_explore_3 --config_files=eval_qmix_def --env=run_env_B \
     --config_overrides="pure/Learner.total_timesteps = 500000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.3" \
     --config_overrides="pure/DQN.lr = 5e-3"'
tmux new-session -d -s 10_15_qmix_explore_4 'python qmixing_main.py --run_name=10_15_qmix_explore_4 --config_files=eval_qmix_def --env=run_env_B \
     --config_overrides="pure/Learner.total_timesteps = 500000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.3" \
     --config_overrides="pure/DQN.lr = 5e-4" \
     --config_overrides="pure/DQN.grad_norm_clipping = None"'
