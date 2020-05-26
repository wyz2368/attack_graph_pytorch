""" Script players. """
import numpy as np


class Player2v0():
    """ Player 2 policy.

    This agent will run left (to the goal) if it has the ball; otherwise, it will act randomly.
    """

    def __call__(self, observation, *args, **kwargs):
        has_ball = observation[4]

        if has_ball == 2:
            return 2
        else:
            return np.random.choice(5)


class Player2v1():
    """ Player 2 policy.

    This agent will act randomly.
    """

    def __call__(self, observation, *args, **kwargs):
        return np.random.choice(5)


class Player2v2():
    """ Player 2 policy.

    Semi-intelligent player:
     - If have ball, move towards goal.
        - If can be intercepted:
            - Move rows.
        - If can't be intercepted:
            - Move towards goal.
     - If don't have ball:
        - If can intercept:
            - If in same row, stay.
            - Else: move rows.
        - If can't intercept:
            - Run towards my goal.

    """

    def __call__(self, observation, *args, **kwargs):
        p1_r, p1_c = observation[0], observation[1]
        p2_r, p2_c = observation[2], observation[3]
        has_ball = observation[4]

        if has_ball == 2:
            # Can be intercepted.
            if p1_c < p2_c:
                # Change rows.
                return 0 if p2_c else 1
            else:
                # Move towards goal.
                return 2
        else:
            # If can intercept.
            if p2_c > p1_c:
                # If in the same row, stay put.
                if p1_r == p2_r:
                    return 4
                # Otherwise move rows.
                else:
                    return 0 if p2_c else 1
            else:
                # If we can't currently intercept the ball, move towards goal.
                return 3


class Player2v3():
    """ Player 2 policy.

    Semi-intelligent non-determinisitc agent.
    """

    def __call__(self, observation, *args, **kwargs):
        p1_r, p1_c = observation[0], observation[1]
        p2_r, p2_c = observation[2], observation[3]
        has_ball = observation[4]

        if has_ball == 2:
            # Can be intercepted.
            if p1_c < p2_c:
                # 60% of the time change rows; otherwise run towards goal.
                change_rows = np.random.choice([True, False], p=[0.6, 0.4])
                if change_rows:
                    # Change rows.
                    return 0 if p2_c else 1
                else:
                    return 2
            else:
                # Move towards goal.
                return 2
        else:
            # If can intercept.
            if p2_c > p1_c:
                # If in the same row, move closer.
                if p1_r == p2_r:
                    return 3
                # Otherwise move rows.
                else:
                    return 0 if p2_c else 1
            else:
                # If we can't currently intercept the ball, move towards goal.
                return 3


class Player2v4():
    """ Player 2 policy. """

    def __call__(self, observation, *args, **kwargs):
        p1_r, p1_c = observation[0], observation[1]
        p2_r, p2_c = observation[2], observation[3]
        has_ball = observation[4]

        if has_ball == 2:
            # Can be intercepted.
            if p1_c < p2_c:
                # Change rows.
                return 0 if p2_c else 1
            else:
                # Move towards goal.
                return 2
        else:
            # If can intercept.
            if p2_c > p1_c:
                # If in the same row, stay put.
                if p1_r == p2_r:
                    return 4
                # Otherwise move rows.
                else:
                    return 0 if p2_c else 1
            elif p2_c == p1_c:
                # If they're next to each other, try and run into them.
                if p2_r > p1_r:
                    return 1
                else:
                    return 0
            else:
                # If we can't currently intercept the ball, move towards goal.
                return 3
