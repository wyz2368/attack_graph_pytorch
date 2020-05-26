""" Training loop for a pytorch DQN agent. """
import gin


def get_dqn_policy(locals_: dict = {}, globals_: dict = {}):
    """ Returns a new DQN agent for training. """
    from attackgraph.rl.dqn.dqn import DQN
    best_responder = DQN(
        is_attacker=locals_["training_attacker"],
        input_size=locals_["observation_space"],
        hidden_sizes=gin.REQUIRED,
        output_size=locals_["n_actions"],
        parameter_noise=locals_["self"].param_noise)
    return best_responder


def get_dqn_enc_dec_policy(locals_: dict = {}, globals_: dict = {}):
    """ Returns a new DQN-EncDec agent for training. """
    from attackgraph.rl.dqn.dqn_encdec import DQNEncDec
    best_responder = DQNEncDec(
        is_attacker=locals_["training_attacker"],
        state_dim=locals_["observation_space"],
        hidden_sizes=gin.REQUIRED,
        action_dim=locals_["n_actions"],
        parameter_noise=locals_["self"].param_noise)
    return best_responder
