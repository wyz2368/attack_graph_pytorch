#!/bin/bash
cd ../attackgraph/soccer/

tmux new-session -d -s 10_15_soccer_0 'python qmixing_main.py --run_name=10_15_soccer_0 --config_files=soccer_qmix --env=run_env_B \
     --config_overrides="Trainer.total_timesteps = 200000" \
     --config_overrides="Trainer.learning_starts = 10000"'

tmux new-session -d -s 10_15_soccer_1 'python qmixing_main.py --run_name=10_15_soccer_1 --config_files=soccer_qmix --env=run_env_B \
     --config_overrides="Trainer.total_timesteps = 200000" \
     --config_overrides="Trainer.learning_starts = 10000" \
     --config_overrides="DQN.lr = 5e-4"'

tmux new-session -d -s 10_15_soccer_2 'python qmixing_main.py --run_name=10_15_soccer_2 --config_files=soccer_qmix --env=run_env_B \
     --config_overrides="Trainer.total_timesteps = 200000" \
     --config_overrides="Trainer.learning_starts = 10000" \
     --config_overrides="DQN.lr = 5e-4" \
     --config_overrides="DQN.hidden_sizes = [5, 5]"'

tmux new-session -d -s 10_15_soccer_3 'python qmixing_main.py --run_name=10_15_soccer_3 --config_files=soccer_qmix --env=run_env_B \
     --config_overrides="Trainer.total_timesteps = 200000" \
     --config_overrides="Trainer.learning_starts = 10000" \
     --config_overrides="DQN.lr = 5e-3"'
