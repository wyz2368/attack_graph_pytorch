""" Grid-World Soccer.


Resources:
 - https://github.com/auputiger/MarkovGameLearning
 - https://www.aaai.org/Papers/ICML/2003/ICML03-034.pdf
 - https://www2.cs.duke.edu/courses/spring07/cps296.3/littman94markov.pdf
"""
import numpy as np
from gym import spaces


class GridWorldSoccer():
    """ Grid-World Soccer environment.

    Observation space (9): 8 for the field, and then an int for who has the ball.
    Action space (5): [Up, Down, Left, Right, Stick].

    """

    FIELD_DIMENSIONS = [2, 4]

    def __init__(self):
        # Player coordinates.
        self.player1 = [1, 1]
        self.player2 = [0, 2]
        # Ball ownership. Random initial owner.
        self.has_ball = np.random.choice([1, 2])

    def step(self, actions):
        """ Step environment forward.

        Both players choose actions simultaneously, however actions are not
        executed simultaneously. There is a 50% chance player 1 will go before
        player 2.

        """
        assert 1 in actions and 2 in actions
        self._update_state(actions)
        # State is player positions followed by ball position.
        state = self._get_state()
        # Zero-sum game.
        rewards = {1: self._get_reward(), 2: -1.0*self._get_reward()}
        # If the rewards aren't 0, then someone has scored and the episode is over.
        done = rewards[1] != 0
        return state, rewards, done, {}

    def reset(self):
        return self._get_state()

    def render(self):
        field = np.zeros(GridWorldSoccer.FIELD_DIMENSIONS)
        field[self.player1[0], self.player1[1]] = 1
        field[self.player2[0], self.player2[1]] = 2
        rendering = f"Field: \n{field}\nPlayer {self.has_ball} has the ball."
        return rendering

    def _get_state(self):
        state = np.zeros(5)
        state[0] = self.player1[0]
        state[1] = self.player1[1]
        state[2] = self.player2[0]
        state[3] = self.player2[1]
        state[4] = self.has_ball
        return {1: state, 2: state}

    def _update_state(self, actions):
        """ Update the game state.

        Either player 1 or player 2's action is randomly chosen to be processed first.

        :param actions: Actions.
        """
        new_coords_player1 = self._get_new_coords(self.player1, actions[1])
        new_coords_player2 = self._get_new_coords(self.player2, actions[2])
        first_player = np.random.choice([1, 2])

        # If player1's actions get processed first.
        if first_player == 1:
            # If player 1 moves onto player 2's position.
            if (new_coords_player1[0] == self.player2[0]) and (new_coords_player1[1] == self.player2[1]):
                # Player 2 steals player 1's ball.
                self.player1 = self.player1
                self.player2 = self.player2
                self.has_ball = 2

            else:
                if self.has_ball == 1:
                    self.player1 = new_coords_player1
                    # Player 2 can move if player 1 hasn't already moved into that position.
                    if (new_coords_player1[0] != new_coords_player2[0]) or (new_coords_player1[1] != new_coords_player2[1]):
                        self.player2 = new_coords_player2
                    else:
                        self.player2 = self.player2
                    self.has_ball = 1

                else:
                    if (new_coords_player1[0] == new_coords_player2[0]) and (new_coords_player1[1] == new_coords_player2[1]):
                        self.player1 = new_coords_player1
                        self.player2 = self.player2
                        self.has_ball = 1
                    else:
                        self.player1 = new_coords_player1
                        self.player2 = new_coords_player2
                        self.has_ball = 2

        # If player2's actions get processed first.
        elif first_player == 2:
            # If player 2 moves onto player 1's position.
            if (new_coords_player2[0] == self.player1[0]) and (new_coords_player2[1] == self.player1[1]):
                # Player 1 steals player 2's ball.
                self.player1 = self.player1
                self.player2 = self.player2
                self.has_ball = 1

            else:
                if self.has_ball == 2:
                    self.player2 = new_coords_player2
                    # Player 1 can move if player 2 hasn't already moved into that position.
                    if (new_coords_player1[0] != new_coords_player2[0]) or (new_coords_player1[1] != new_coords_player2[1]):
                        self.player1 = new_coords_player1
                    else:
                        self.player1 = self.player1
                    self.has_ball = 2

                else:
                    if (new_coords_player1[0] == new_coords_player2[0]) and (new_coords_player1[1] == new_coords_player2[1]):
                        self.player1 = self.player1
                        self.player2 = new_coords_player2
                        self.has_ball = 2
                    else:
                        self.player1 = new_coords_player1
                        self.player2 = new_coords_player2
                        self.has_ball = 1
        else:
            raise ValueError(f"Unknown player {first_player}.")

        assert (self.player1[0] != self.player2[0]) or (self.player1[1] != self.player2[1]), "Players cannot occupy same position."

    def _get_reward(self):
        """ Get the reward for player 1.

        This is a zero-sum game, so player 2's reward is the negation of
        player 1's reward.

        :return: Reward.
        """
        player = self.player1 if self.has_ball == 1 else self.player2
        _, c = player

        if c == 0:
            return -100.0
        elif c == GridWorldSoccer.FIELD_DIMENSIONS[1] - 1:
            return 100.0
        else:
            return 0.0

    @staticmethod
    def _get_new_coords(coords, action):
        """ Get the player's coordinates after taking an action.

        :param coords:
        :param action:
        :return:
        """
        r, c = coords

        if action == 0:                 # Up.
            r = max(0, r - 1)
        elif action == 1:               # Down.
            r = min(GridWorldSoccer.FIELD_DIMENSIONS[0] - 1, r + 1)
        elif action == 2:               # Left.
            c = max(0, c - 1)
        elif action == 3:               # Right.
            c = min(GridWorldSoccer.FIELD_DIMENSIONS[1] - 1, c + 1)

        return r, c
