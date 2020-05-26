#!/bin/bash
cd ../attackgraph/


tmux new-session -d -s 01_07_qmix_explore_def_0 'python qmixing_main.py \
     --run_name=01_07_qmix_explore_def_0 \
     --env=run_env_B \
     --config_files=eval_qmix_def \
     --config_overrides="pure/Learner.total_timesteps = 5000000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.3" \
     --config_overrides="pure/DQN.lr = 5e-5" \
     --config_overrides="mix/Learner.total_timesteps = 5000000" \
     --config_overrides="mix/Learner.exploration_fraction = 0.3" \
     --config_overrides="mix/DQN.lr = 5e-5"'


tmux new-session -d -s 01_07_qmix_explore_def_1 'python qmixing_main.py \
     --run_name=01_07_qmix_explore_def_1 \
     --env=run_env_B \
     --config_files=eval_qmix_def \
     --config_overrides="pure/Learner.total_timesteps = 1000000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.3" \
     --config_overrides="pure/DQN.lr = 5e-5" \
     --config_overrides="mix/Learner.total_timesteps = 1000000" \
     --config_overrides="mix/Learner.exploration_fraction = 0.3" \
     --config_overrides="mix/DQN.lr = 5e-5"'


tmux new-session -d -s 01_07_qmix_explore_def_2 'python qmixing_main.py \
     --run_name=01_07_qmix_explore_def_2 \
     --env=run_env_B \
     --config_files=eval_qmix_def \
     --config_overrides="pure/Learner.total_timesteps = 700000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.3" \
     --config_overrides="pure/DQN.lr = 5e-5" \
     --config_overrides="mix/Learner.total_timesteps = 700000" \
     --config_overrides="mix/Learner.exploration_fraction = 0.3" \
     --config_overrides="mix/DQN.lr = 5e-5"'


tmux new-session -d -s 01_07_qmix_explore_def_3 'python qmixing_main.py \
     --run_name=01_07_qmix_explore_def_3 \
     --env=run_env_B \
     --config_files=eval_qmix_def \
     --config_overrides="pure/Learner.total_timesteps = 500000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.3" \
     --config_overrides="pure/DQN.lr = 5e-5" \
     --config_overrides="mix/Learner.total_timesteps = 500000" \
     --config_overrides="mix/Learner.exploration_fraction = 0.3" \
     --config_overrides="mix/DQN.lr = 5e-5"'

tmux new-session -d -s 01_07_qmix_explore_def_4 'python qmixing_main.py \
     --run_name=01_07_qmix_explore_def_4 \
     --env=run_env_B \
     --config_files=eval_qmix_def \
     --config_overrides="pure/Learner.total_timesteps = 400000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.3" \
     --config_overrides="pure/DQN.lr = 5e-5" \
     --config_overrides="mix/Learner.total_timesteps = 400000" \
     --config_overrides="mix/Learner.exploration_fraction = 0.3" \
     --config_overrides="mix/DQN.lr = 5e-5"'

tmux new-session -d -s 01_07_qmix_explore_def_5 'python qmixing_main.py \
     --run_name=01_07_qmix_explore_def_5 \
     --env=run_env_B \
     --config_files=eval_qmix_def \
     --config_overrides="pure/Learner.total_timesteps = 250000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.3" \
     --config_overrides="pure/DQN.lr = 5e-5" \
     --config_overrides="mix/Learner.total_timesteps = 250000" \
     --config_overrides="mix/Learner.exploration_fraction = 0.3" \
     --config_overrides="mix/DQN.lr = 5e-5"'
