""" Empirical Game Theoritc Analysis (EGTA) algorithm's code. """
import logging
import multiprocessing
import os
import os.path as osp
import sys
import time

import dill as pickle
import gin
import numpy as np
import psutil
from absl import flags
from tensorboardX import SummaryWriter

import attackgraph.common.file_ops as fp
from attackgraph import DagGenerator as dag
from attackgraph import empirical_game
from attackgraph import gambit_analysis as ga
from attackgraph import settings, simulation, training, uniform_str_init, util
from attackgraph.common.cloudpickle_wrapper import CloudpickleWrapper
from attackgraph.rl.learner_worker import LearnerWorker
from attackgraph.meta_solvers import meta_method_selector

logger = logging.getLogger(__name__)
FLAGS = flags.FLAGS


@gin.configurable
def run(load_env, env_name, n_processes):
    """ Run the double-oracle algorithm. """
    # Create initial policies.
    fp.save_pkl(
        uniform_str_init.act_att,
        osp.join(settings.get_attacker_strategy_dir(), "att_str_epoch1.pkl"))
    fp.save_pkl(
        uniform_str_init.act_def,
        osp.join(settings.get_defender_strategy_dir(), "def_str_epoch1.pkl"))

    game = initialize(load_env=FLAGS.env, env_name=None, n_processes=n_processes)
    _run(game.env, game, meta_method_name=FLAGS.meta_method, n_processes=n_processes)


def initialize(load_env=None, env_name=None, n_processes: int = 1):
    logger.info("=======================================================")
    logger.info("=======Begin Initialization and first epoch============")
    logger.info("=======================================================")

    # Create Environment
    if isinstance(load_env, str):
        path = osp.join(settings.get_env_data_dir(), "{}.pkl".format(load_env))
        if not fp.isExist(path):
            raise ValueError("The env being loaded does not exist.")
        env = fp.load_pkl(path)
    else:
        # env is created and saved.
        env = dag.env_rand_gen_and_save(env_name)

    # save graph copy
    env.save_graph_copy()
    env.save_mask_copy()  # TODO: change transfer

    # create players and point to their env
    env.create_players()
    env.create_action_space()

    # print root node
    roots = env.get_Roots()
    logger.info(f"Root Nodes: {roots}")
    ed = env.get_ORedges()
    logger.info(f"Or edges: {ed}")

    # initialize game data
    game = empirical_game.EmpiricalGame(env)
    game.env.defender.set_env_belong_to(game.env)
    game.env.attacker.set_env_belong_to(game.env)

    # make no sense
    env.defender.set_env_belong_to(env)
    env.attacker.set_env_belong_to(env)

    # uniform strategy has been produced ahead of time
    logger.info("Epoch 1")
    epoch = 1
    epoch_dir = osp.join(settings.get_results_dir(), f"epoch_{epoch}")
    writer = SummaryWriter(logdir=epoch_dir)

    act_att = 'att_str_epoch1.pkl'
    act_def = 'def_str_epoch1.pkl'

    game.add_att_str(act_att)
    game.add_def_str(act_def)

    logger.info('Begin simulation for uniform strategy.')
    aReward, dReward = simulation.simulate_profile(
        env=game.env,
        game=game,
        nn_att=act_att,
        nn_def=act_def,
        n_episodes=game.num_episodes,
        n_processes=n_processes,
        save_dir=epoch_dir,
        summary_writer=writer)
    logger.info('Done simulation for uniform strategy.')

    game.init_payoffmatrix(dReward, aReward)
    ne = {}
    ne[0] = np.array([1], dtype=np.float32)
    ne[1] = np.array([1], dtype=np.float32)
    game.add_nasheq(epoch, ne)

    # save a copy of game data
    game_path = osp.join(settings.get_run_dir(), "game.pkl")
    fp.save_pkl(game, game_path)

    sys.stdout.flush()
    return game


def _run(env, game, meta_method_name, epoch: int = 1, game_path: str = None, n_processes: int = 1):
    assert n_processes > 0, "Invalid number of processors."
    if game_path is None:
        game_path = osp.join(settings.get_run_dir(), "game.pkl")

    logger.info("=======================================================")
    logger.info("===============Begin Running DO-EGTA===================")
    logger.info("=======================================================")

    proc = psutil.Process(os.getpid())
    result_dir = settings.get_run_dir()

    selector = meta_method_selector(meta_method_name)

    count = 80
    while count != 0:
        mem0 = proc.memory_info().rss

        # Fix opponent strategy.
        mix_str_def, mix_str_att = selector.sample(game, epoch)

        # Save mixed strategies.
        # with open(osp.join(result_dir, f"mix_defender.{epoch}.pkl"), "wb") as outfile:
        #     pickle.dump(mix_str_def, outfile)
        # with open(osp.join(result_dir, f"mix_attacker.{epoch}.pkl"), "wb") as outfile:
        #     pickle.dump(mix_str_att, outfile)
        # with open(osp.join(result_dir, f"payoff_defender.{epoch}.pkl"), "wb") as outfile:
        #     pickle.dump(game.payoffmatrix_def, outfile)
        # with open(osp.join(result_dir, f"payoff_attacker.{epoch}.pkl"), "wb") as outfile:
        #     pickle.dump(game.payoffmatrix_att, outfile)

        # Equilibrium pay-off.
        aPayoff, dPayoff = util.payoff_mixed_NE(game, epoch)
        game.att_payoff.append(aPayoff)
        game.def_payoff.append(dPayoff)

        # increase epoch
        epoch += 1
        logger.info("Epoch " + str(epoch))
        epoch_dir = osp.join(result_dir, f"epoch_{epoch}")

        # Summary writer for each epoch.
        writer = SummaryWriter(logdir=epoch_dir)

        # train and save RL agents

        # Train new best-response policies.
        if n_processes > 1:
            logger.info("Begining training attacker and defender in parallel.")
            time_training = time.time()
            job_queue = multiprocessing.SimpleQueue()
            result_queue = multiprocessing.SimpleQueue()

            attacker_trainer = LearnerWorker(job_queue, result_queue, 1, mix_str_def, epoch)
            defender_trainer = LearnerWorker(job_queue, result_queue, 0, mix_str_att, epoch)

            attacker_trainer.start()
            defender_trainer.start()

            # Submit training jobs on our game.
            for _ in range(2):
                job_queue.put(CloudpickleWrapper(game))
            # Send sentinel values to tell processes to cleanly shutdown (1 per worker).
            for _ in range(2):
                job_queue.put(None)

            attacker_trainer.join()
            defender_trainer.join()

            # Collect and report results. We need to sort the results because they may appear in any order.
            results = []
            for _ in range(2):
                results += [result_queue.get()]
            results = results if not results[0][0] else results[::-1]  # Put defender first then attacker.

            # Process results into expected variables for non-distributed.
            a_BD = results[1][1]
            d_BD = results[0][1]

            logger.info("Done training attacker and defender.")
            logger.info(f"Defender training report: \n{results[0][2]}")
            logger.info(f"Attacker training report: \n{results[1][2]}")
            time_training = time.time() - time_training

        else:
            logger.info("Begin training attacker......")
            time_train_attacker = time.time()
            a_BD, report = training.train(game, 1, mix_str_def, epoch, writer)
            time_train_attacker = time.time() - time_train_attacker
            logger.info(f"\n{report}")
            logger.info("Attacker training done......")

            logger.info("Begin training defender......")
            time_train_defender = time.time()
            d_BD, report = training.train(game, 0, mix_str_att, epoch, writer)
            time_train_defender = time.time() - time_train_defender
            logger.info(f"\n{report}")
            logger.info("Defender training done......")

        mem1 = proc.memory_info().rss

        game.att_BD_list.append(a_BD)
        game.def_BD_list.append(d_BD)

        mem2 = proc.memory_info().rss

        game.add_att_str("att_str_epoch" + str(epoch) + ".pkl")
        game.add_def_str("def_str_epoch" + str(epoch) + ".pkl")

        # simulate and extend the payoff matrix.
        time_extend_game = time.time()
        game = simulation.simulate_expanded_game(
            game=game,
            n_processes=n_processes,
            save_dir=epoch_dir,
            summary_writer=writer)
        time_extend_game = time.time() - time_extend_game
        mem3 = proc.memory_info().rss

        # find nash equilibrium using gambit analysis
        time_gambit = time.time()
        payoffmatrix_def = game.payoffmatrix_def
        payoffmatrix_att = game.payoffmatrix_att
        logger.info("Begin Gambit analysis.")
        nash_att, nash_def = ga.do_gambit_analysis(payoffmatrix_def, payoffmatrix_att)
        ga.add_new_NE(game, nash_att, nash_def, epoch)
        game.env.attacker.nn_att = None
        game.env.defender.nn_def = None
        fp.save_pkl(game, game_path)
        time_gambit = time.time() - time_gambit

        logger.info("RESULTS:")
        logger.info('  - a_BD_list: {}'.format(game.att_BD_list))
        logger.info('  - aPayoff: {}'.format(game.att_payoff))
        logger.info('  - d_BD_list: {}'.format(game.def_BD_list))
        logger.info('  - dPayoff: {}'.format(game.def_payoff))
        logger.info("MEM: {}, {}, {}.".format((mem1 - mem0) / mem0, (mem2 - mem0) / mem0, (mem3 - mem0) / mem0))
        logger.info("TIME: ")
        if n_processes == 1:
            logger.info(f"  - Training attacker: {time_train_attacker}")
            logger.info(f"  - Training defender: {time_train_defender}")
        else:
            logger.info(f"  - Training: {time_training}")
        logger.info(f"  - Extend game: {time_extend_game}")
        logger.info(f"  - Gambit: {time_gambit}")
        logger.info("Round_" + str(epoch) + " has done and game was saved.")
        logger.info("=======================================================")

        count -= 1
        sys.stdout.flush()  # TODO: make sure this is correct.

    logger.info("END: " + str(epoch))
    os._exit(os.EX_OK)
