""" Wraps a multi-agent environment to feel like a single agent environment. """
import typing
from dataclasses import dataclass


@dataclass
class MultiToSingleAgentWrapper():
    """ Wraps a multi-agent environment to appear as a single-agent environment.

    Note: For ease of documentation we will refer to all other agents in the environment
    as opponents, even though they may be teammates.
    """

    env: typing.Any
    # The single agent's ID.
    agent_id: typing.Any
    # Dictionary of agent's IDs to themselves.
    opponents: typing.Dict

    def __post_init__(self):
        assert self.agent_id not in self.opponents, f"Single-agent ID `{self.agent_id} found in opponent IDs.`"

    def step(self, action, **kwargs):
        actions = {}
        # Add single-agent action.
        actions[self.agent_id] = action
        # Collect opponent actions.
        for opponent_id, opponent in self.opponents.items():
            actions[opponent_id] = opponent(self._state_cache[opponent_id])

        states, rewards, done, info = self.env.step(actions, **kwargs)
        state = self._cache_and_parse_state(states)
        return state, rewards[self.agent_id], done, info

    def reset(self, **kwargs):
        states = self.env.reset(**kwargs)
        state = self._cache_and_parse_state(states)

        for agent in self.opponents.values():
            if hasattr(agent, "begin_episode"):
                agent.begin_episode()

        return state

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def _cache_and_parse_state(self, states):
        agent_state = states[self.agent_id]
        del states[self.agent_id]
        self._state_cache = states
        return agent_state
