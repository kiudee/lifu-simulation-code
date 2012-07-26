#!/usr/bin/env python
# vim: et tabstop=8 shiftwidth=4 softtabstop=4

import numpy as np
import bigfloat
from math import sqrt, log, expm1, exp
from util import choice_with_weight


class Exp3Forecaster():
    """Forecaster using the Exp3-Algorithm."""
    def __init__(self, options, rounds, highprob=False, lr=None):
        if lr == None:
            self._learning_rate = min(1.0, sqrt((options * log(options)) / (expm1(1) * rounds)))
        else:
            self._learning_rate = lr
        self._weights = np.ones(options, dtype=np.dtype("f16"))
        self.prob = self._weights / options
        self.gains = np.zeros(options)

    def next_arm(self):
        return choice_with_weight(self.prob)

    def reward(self, arm, value):
        self.gains[arm] += value
        adjusted_gain = value / self.prob[arm]
        options = len(self.prob)
        self._weights[arm] *= bigfloat.exp(self._learning_rate * adjusted_gain / options, bigfloat.precision(128))
        sum_weights = np.sum(self._weights)
        self.prob = np.array([w / sum_weights for w in self._weights])

    def get_name(self):
        return "Exp3"
if __name__ == "__main__":
    from scipy.stats import bernoulli
    def smallsim(lr):
        if lr > 0.9 or lr < 1e-10:
            return 1e10
        print(lr)
        rounds = 1000
        reps = 10
        rewards = {0: bernoulli.rvs(0.5, size=rounds),
                   1: bernoulli.rvs(0.6, size=rounds)}
        cumregret = 0.0
        for rep in range(reps):
            exp3 = Exp3Forecaster(2, rounds, lr=lr)
            cumrewards = np.zeros(2)
            lrncumreward = 0.0
            for t in range(rounds):
                elem = exp3.next_arm()
                cumrewards[0] += rewards[0][t]
                cumrewards[1] += rewards[1][t]
                rew = rewards[elem][t]
                exp3.reward(elem, rew)
                lrncumreward += rew
                cumregret += np.max(cumrewards) - lrncumreward
            del exp3
        cumregret /= reps
        return cumregret
    from scipy.optimize import anneal
    start = min(1.0, sqrt((2 * log(2)) / (expm1(1) * 10000)))
    anneal(smallsim, start, full_output=True, lower=1e-10, upper=0.9)

    # rounds = 10000
    # exp3 = Exp3Forecaster(2, rounds)
    #rewards = {0 : truncnorm.rvs(0, 1, size=rounds),
               #1 : truncnorm.rvs(0.1, 1, size=rounds),
               #2 : truncnorm.rvs(0.2, 0.9, size=rounds),
               #3 : truncnorm.rvs(0.3, 0.5, size=rounds),
               #4 : truncnorm.rvs(0, 0.9, size=rounds)
              #}
    #rewards = {0: truncnorm.rvs(0.9, 1.0, size=rounds),
               #1: truncnorm.rvs(0.85, 1.0, size=rounds),
               #2: truncnorm.rvs(0.8, 1.0, size=rounds),
               #3: truncnorm.rvs(0.75, 1.0, size=rounds),
               #4: truncnorm.rvs(0.7, 1.0, size=rounds)
              #}
    # rewards = {0: bernoulli.rvs(0.8, size=rounds),
    #            1: bernoulli.rvs(0.9, size=rounds),
    #            2: bernoulli.rvs(1.0, size=rounds),
    #            3: bernoulli.rvs(1.0, size=rounds),
    #            4: bernoulli.rvs(1.0, size=rounds)
    #           }
    # for t in range(rounds):
    #     elem = exp3.next_arm()
    #     exp3.reward(elem, rewards[elem][t])
    #     #print("[{}] - Elem [{}] - Dist: {}, Sum: {}".format(t, elem, inf.prob, sum(inf.prob)))
    # print("Dist: {}".format(exp3.prob))
    #     #pp.pprint(inf.prob)
