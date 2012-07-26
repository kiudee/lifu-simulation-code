#!/usr/bin/env python
# vim: et tabstop=8 shiftwidth=4 softtabstop=4

import numpy as np
from math import sqrt, log
from util import choice_with_weight
from scipy.optimize import newton

class INForecaster():
    """Forecaster using the PolyInf-Algorithm."""
    def __init__(self, options, rounds, highprob=False):
        self.prob = np.ones(options) / options
        self.gains = np.zeros(options)
        if highprob:
            # Against oblivious:
            self._highprob = True
            self._eta = 2.0 * sqrt(rounds)
            self._gamma = min(1.0 / 2.0, 3.0 * sqrt(float(options) / rounds))
            self._beta = 1.0 / sqrt(2.0 * options * rounds)
            self._delta = 0.10
        else:
            self._highprob = False
            self._gamma = min(1.0 / 2.0, sqrt((3.0 * options) / rounds))
            self._eta = sqrt(5 * rounds)
        self._q = 2.0

    def _potential(self, x):
        return pow(self._eta / -x, self._q) + self._gamma / len(self.prob)

    def _sum_with_constant(self, constant):
        return sum((self._potential(x - constant) for x in self.gains)) - 1.0

    def _calc_constant(self):
        c = newton(self._sum_with_constant, max(self.gains) + 1)
        return c

    def next_arm(self):
        return choice_with_weight(self.prob)

    def reward(self, arm, value):
        if self._highprob:
            self.gains[arm] += (-1.0 / self._beta) * log(1.0 - (self._beta * value) / self.prob[arm])
        else:
            self.gains[arm] += value / self.prob[arm]
        c = self._calc_constant()
        self.prob = np.array([self._potential(gain - c) for gain in self.gains])

    def get_name(self):
        return "INForecaster"

if __name__ == "__main__":
    from scipy.stats import norm, truncnorm, bernoulli
    rounds = 10000
    inf = INForecaster(2, rounds, highprob=False)
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
    rewards = {0: bernoulli.rvs(0.8, size=rounds),
               1: bernoulli.rvs(0.9, size=rounds),
               2: bernoulli.rvs(1.0, size=rounds),
               3: bernoulli.rvs(1.0, size=rounds),
               4: bernoulli.rvs(1.0, size=rounds)
              }
    import pprint as pp
    for t in range(rounds):
        elem = inf.next_arm()
        inf.reward(elem, rewards[elem][t])
        #print("[{}] - Elem [{}] - Dist: {}, Sum: {}".format(t, elem, inf.prob, sum(inf.prob)))
    print("Dist: {}".format(inf.prob))
        #pp.pprint(inf.prob)

