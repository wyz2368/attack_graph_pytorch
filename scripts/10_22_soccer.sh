#!/bin/bash
cd ../attackgraph/


# Soccer: Generate beta-final run results to be averaged together.
cd soccer/
tmux new-session -d -s 10_22_soccer_0 'python qmixing_main.py --run_name=10_22_soccer_0 --config_files=soccer_qmix --env=run_env_B \
     --config_overrides="pure/Trainer.total_timesteps = 10000" \
     --config_overrides="mix/Trainer.total_timesteps = 50000"'
tmux new-session -d -s 10_22_soccer_1 'python qmixing_main.py --run_name=10_22_soccer_1 --config_files=soccer_qmix --env=run_env_B \
     --config_overrides="pure/Trainer.total_timesteps = 10000" \
     --config_overrides="mix/Trainer.total_timesteps = 50000"'
tmux new-session -d -s 10_22_soccer_2 'python qmixing_main.py --run_name=10_22_soccer_2 --config_files=soccer_qmix --env=run_env_B \
     --config_overrides="pure/Trainer.total_timesteps = 10000" \
     --config_overrides="mix/Trainer.total_timesteps = 50000"'
tmux new-session -d -s 10_22_soccer_3 'python qmixing_main.py --run_name=10_22_soccer_3 --config_files=soccer_qmix --env=run_env_B \
     --config_overrides="pure/Trainer.total_timesteps = 10000" \
     --config_overrides="mix/Trainer.total_timesteps = 50000"'
tmux new-session -d -s 10_22_soccer_4 'python qmixing_main.py --run_name=10_22_soccer_4 --config_files=soccer_qmix --env=run_env_B \
     --config_overrides="pure/Trainer.total_timesteps = 10000" \
     --config_overrides="mix/Trainer.total_timesteps = 50000"'
