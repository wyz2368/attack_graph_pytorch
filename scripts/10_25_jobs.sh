#!/bin/bash
cd ../attackgraph/


# Policy Distillation.
tmux new-session -d -s 10_25_distill_0 "python policy_distillation_main.py --run_name=10_25_distill_0 --config_files=policy_distill"
tmux new-session -d -s 10_25_distill_1 "python policy_distillation_main.py --run_name=10_25_distill_1 --config_files=policy_distill"
tmux new-session -d -s 10_25_distill_2 "python policy_distillation_main.py --run_name=10_25_distill_2 --config_files=policy_distill"
tmux new-session -d -s 10_25_distill_3 "python policy_distillation_main.py --run_name=10_25_distill_3 --config_files=policy_distill"
tmux new-session -d -s 10_25_distill_4 "python policy_distillation_main.py --run_name=10_25_distill_4 --config_files=policy_distill"


# QMix: Attacer.
tmux new-session -d -s 10_25_qmix_att "python qmixing_main.py --run_name=10_25_qmix_att --config_files=eval_qmix --env=run_env_B"


# QMix: Defender. The goal of this job is to get an average of performance and get pure-strategy performances.
tmux new-session -d -s 10_25_qmix_def_0 "python qmixing_main.py --run_name=10_25_qmix_def_0 --config_files=eval_qmix_def"
tmux new-session -d -s 10_25_qmix_def_1 "python qmixing_main.py --run_name=10_25_qmix_def_1 --config_files=eval_qmix_def"
tmux new-session -d -s 10_25_qmix_def_2 "python qmixing_main.py --run_name=10_25_qmix_def_2 --config_files=eval_qmix_def"
tmux new-session -d -s 10_25_qmix_def_3 "python qmixing_main.py --run_name=10_25_qmix_def_3 --config_files=eval_qmix_def"
tmux new-session -d -s 10_25_qmix_def_4 "python qmixing_main.py --run_name=10_25_qmix_def_4 --config_files=eval_qmix_def"
