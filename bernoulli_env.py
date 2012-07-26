#!/usr/bin/env python
# vim: et tabstop=8 shiftwidth=4 softtabstop=4

import numpy as np
from scipy.stats import bernoulli


class BernoulliEnvironment():

    def __init__(self, arms, rounds):
        pass

    def get_rewards(self, dist):
        rewards = [bernoulli.rvs(0.6, size=1)[0]]
        for i in range(len(dist) - 1):
            rewards.append(bernoulli.rvs(0.5, size=1)[0])
        return rewards

    def pass_guess(self, arm):
        pass

    def pass_dist(self, dist):
    	pass

    def get_name(self):
    	return "Bernoulli"


