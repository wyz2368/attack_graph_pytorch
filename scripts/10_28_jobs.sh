#!/bin/bash
cd ../attackgraph/


tmux new-session -d -s 10_28_classify_0 'python train_classifier_main.py --run_name=10_28_classify_0 --config_files=qmix_classifier \
     --config_overrides="supervised_learning.eval_freq=10" \
     --config_overrides="MLP.hidden_sizes=[200, 100]"'
tmux new-session -d -s 10_28_classify_1 'python train_classifier_main.py --run_name=10_28_classify_1 --config_files=qmix_classifier \
     --config_overrides="supervised_learning.eval_freq=10" \
     --config_overrides="MLP.hidden_sizes=[200, 200]"'
tmux new-session -d -s 10_28_classify_2 'python train_classifier_main.py --run_name=10_28_classify_2 --config_files=qmix_classifier \
     --config_overrides="supervised_learning.eval_freq=10" \
     --config_overrides="MLP.hidden_sizes=[50, 50]"'
tmux new-session -d -s 10_28_classify_3 'python train_classifier_main.py --run_name=10_28_classify_3 --config_files=qmix_classifier \
     --config_overrides="supervised_learning.eval_freq=10" \
     --config_overrides="MLP.hidden_sizes=[200, 128, 128]"'
tmux new-session -d -s 10_28_classify_4 'python train_classifier_main.py --run_name=10_28_classify_4 --config_files=qmix_classifier \
     --config_overrides="supervised_learning.eval_freq=10" \
     --config_overrides="supervised_learning.batch_size=128" \
     --config_overrides="MLP.hidden_sizes=[128, 128]"'
tmux new-session -d -s 10_28_classify_5 'python train_classifier_main.py --run_name=10_28_classify_5 --config_files=qmix_classifier \
     --config_overrides="supervised_learning.eval_freq=10" \
     --config_overrides="supervised_learning.batch_size=32" \
     --config_overrides="MLP.hidden_sizes=[128, 128]"'
tmux new-session -d -s 10_28_classify_6 'python train_classifier_main.py --run_name=10_28_classify_6 --config_files=qmix_classifier \
     --config_overrides="supervised_learning.eval_freq=10" \
     --config_overrides="supervised_learning.batch_size=256" \
     --config_overrides="MLP.hidden_sizes=[128, 128]"'
