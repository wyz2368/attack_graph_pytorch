""" Learner worker processes. """
import os.path as osp
import sys
from multiprocessing import Process

import gin
from absl import flags
from tensorboardX import SummaryWriter

from attackgraph import settings, training


class LearnerWorker(Process):

    def __init__(self, job_queue, result_queue, is_attacker, opponent_mixed_strategy, epoch):
        super(LearnerWorker, self).__init__()
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.is_attacker = is_attacker
        self.opponent_mixed_strategy = opponent_mixed_strategy
        self.epoch = epoch

    def run(self):
        # Reparse the flags for this process.
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)
        # Reload gin configurations for this process.
        gin_files = [osp.join(settings.SRC_DIR, "configs", f"{x}.gin") for x in FLAGS.config_files]
        gin.parse_config_files_and_bindings(
            config_files=gin_files,
            bindings=FLAGS.config_overrides,
            skip_unknown=False)

        for job in iter(self.job_queue.get, None):
            game = job()
            writer = SummaryWriter(logdir=osp.join(settings.get_run_dir(), f"epoch_{self.epoch}"))

            best_deviation, report = training.train(
                game=game,
                identity=self.is_attacker,
                opponent_mix_str=self.opponent_mixed_strategy,
                epoch=self.epoch,
                writer=writer)

            self.result_queue.put((self.is_attacker, best_deviation, report))
