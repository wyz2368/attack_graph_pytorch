#!/bin/bash
cd ../attackgraph/

# Classifier.
tmux new-session -d -s 10_23_classifier "python train_classifier_main.py --run_name=10_23_classifier --config_files=qmix_classifier"


# Policy Distillation.
tmux new-session -d -s 10_23_distill "python policy_distillation_main.py --run_name=10_23_distill --config_files=policy_distill"


# QMix
tmux new-session -d -s 10_23_qmix_0 "python qmixing_main.py --run_name=10_23_qmix_0 --config_files=eval_qmix_def"
tmux new-session -d -s 10_23_qmix_1 "python qmixing_main.py --run_name=10_23_qmix_1 --config_files=eval_qmix_def"
tmux new-session -d -s 10_23_qmix_2 "python qmixing_main.py --run_name=10_23_qmix_2 --config_files=eval_qmix_def"
tmux new-session -d -s 10_23_qmix_3 "python qmixing_main.py --run_name=10_23_qmix_3 --config_files=eval_qmix_def"
tmux new-session -d -s 10_23_qmix_4 "python qmixing_main.py --run_name=10_23_qmix_4 --config_files=eval_qmix_def"
