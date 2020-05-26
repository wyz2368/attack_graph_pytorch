#!/bin/bash
cd ../attackgraph/


tmux new-session -d -s 10_31_classify_0 'python train_classifier_main.py --run_name=10_29_classify_0 --config_files=qmix_classifier \
     --config_overrides="supervised_learning.eval_freq=10" \
     --config_overrides="MLP.hidden_sizes=[200, 129, 129]"'
tmux new-session -d -s 10_31_classify_1 'python train_classifier_main.py --run_name=10_29_classify_1 --config_files=qmix_classifier \
     --config_overrides="supervised_learning.eval_freq=10" \
     --config_overrides="MLP.hidden_sizes=[200, 129, 129]"'
tmux new-session -d -s 10_31_classify_2 'python train_classifier_main.py --run_name=10_29_classify_2 --config_files=qmix_classifier \
     --config_overrides="supervised_learning.eval_freq=10" \
     --config_overrides="MLP.hidden_sizes=[200, 129, 129]"'
tmux new-session -d -s 10_31_classify_3 'python train_classifier_main.py --run_name=10_29_classify_3 --config_files=qmix_classifier \
     --config_overrides="supervised_learning.eval_freq=10" \
     --config_overrides="MLP.hidden_sizes=[200, 129, 129]"'
tmux new-session -d -s 10_31_classify_4 'python train_classifier_main.py --run_name=10_29_classify_4 --config_files=qmix_classifier \
     --config_overrides="supervised_learning.eval_freq=10" \
     --config_overrides="MLP.hidden_sizes=[200, 129, 129]"'
