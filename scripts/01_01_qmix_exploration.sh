#!/bin/bash
cd ../attackgraph/
source activate bgg

# Defender.
tmux new-session -d -s 01_01_qmix_explore_def_0 'python qmixing_main.py \
     --run_name=01_01_qmix_explore_def_0 \
     --env=run_env_B \
     --config_files=eval_qmix_def \
     --config_overrides="pure/Learner.total_timesteps = 10000000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.3" \
     --config_overrides="pure/DQN.lr = 5e-5"
     --config_overrides="mix/Learner.total_timesteps = 30000000" \
     --config_overrides="mix/Learner.exploration_fraction = 0.3" \
     --config_overrides="mix/DQN.lr = 5e-5"'

# Attacker.
tmux new-session -d -s 01_01_qmix_explore_att_0 'python qmixing_main.py \
     --run_name=01_01_qmix_explore_att_0 \
     --env=run_env_B \
     --config_files=eval_qmix \
     --config_overrides="pure/Learner.total_timesteps = 10000000" \
     --config_overrides="pure/Learner.exploration_fraction = 0.3" \
     --config_overrides="pure/DQN.lr = 5e-5"
     --config_overrides="mix/Learner.total_timesteps = 30000000" \
     --config_overrides="mix/Learner.exploration_fraction = 0.3" \
     --config_overrides="mix/DQN.lr = 5e-5"'
