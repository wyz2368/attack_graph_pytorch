""" Quick and dirty script to determine how similar two buffers are. """
import os
import os.path as osp

import numpy
import pickle
from tqdm import tqdm


buffer_paths = [
    "/home/mxsmith/projects/attack_graph/results/10_15_qmix_statefreq/09_30_egta_dqn_att_epoch1.best_response.replay_buffer.pkl",
    "/home/mxsmith/projects/attack_graph/results/10_15_qmix_statefreq/09_30_egta_dqn_att_epoch3.best_response.replay_buffer.pkl",
    "/home/mxsmith/projects/attack_graph/results/10_15_qmix_statefreq/09_30_egta_dqn_att_epoch7.best_response.replay_buffer.pkl"]
buffers = [pickle.load(open(x, "rb")) for x in buffer_paths]

print(f"Example state ([{len(buffers[0]._storage[0][0])}]): \n{buffers[0]._storage[0][0]}")


# State --> Freq.
state_freqs = [{}, {}, {}]

for buffer_i, buffer in enumerate(buffers):
    print(f"Reading buffer {buffer_i}")

    for experience in tqdm(buffer._storage):
        s, *_ = experience

        # Remove `timeLeft` from consideration.
        s = s[:90]

        s = str(s)

        if s not in state_freqs[buffer_i]:
            state_freqs[buffer_i][s] = 0
        state_freqs[buffer_i][s] += 1

    print(f"Found {len(state_freqs[buffer_i])} unique states.")

# Count number of states that all buffers share.
n = 0
shared_states = set()
for s in state_freqs[0].keys():
    counts = []
    for state_freq in state_freqs:
        if s in state_freq:
            counts += [state_freq[s]]
        else:
            counts += [0]
    n += min(counts)
    if min(counts) > 0:
        shared_states.add(s)
print(f"The buffers share {n} total states.")
print(f"The buffers share {len(shared_states)} unique states.")

# Count number of unique states.
unique_states = set()
for state_freq in state_freqs:
    unique_states.update(state_freq.keys())
n_states = len(unique_states)
print(f"These shared states are across {n_states} unique states.")

# Count the number of total states.
total_states = 0
for state_freq in state_freqs:
    for freq in state_freq.values():
        total_states += freq
print(f"There are {total_states} total states.")
