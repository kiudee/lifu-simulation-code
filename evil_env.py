#!/usr/bin/env python
# vim: et tabstop=8 shiftwidth=4 softtabstop=4

import numpy as np
from math import sqrt, log, expm1


class EvilEnvironment():

    """The evil environment tries to use the probability distribution
    of the learner against himself.

    """

    def __init__(self, arms, rounds):
        self._arms = arms
        self._dist = np.ones(arms) / arms
        sim_learning_rate = min(1.0,
                                sqrt((arms * log(arms)) /
                                     (expm1(1.0) * rounds)))
        self._alpha = 3 * sim_learning_rate

    def get_rewards(self, dist):
        rewards = np.zeros(self._arms)
        if self._dist[0] < self._alpha:
            rewards[0] = 1.0
        else:
            rewards[1] = 1.0
        return rewards

    def pass_guess(self, arm):
        pass

    def pass_dist(self, dist):
        """Recieve the probability distribution of the learner."""
        self._dist = dist

    def get_name(self):
        return "Evil"
