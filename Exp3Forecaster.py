#!/usr/bin/env python
# vim: et tabstop=8 shiftwidth=4 softtabstop=4

import numpy as np
from math import sqrt, log, expm1, exp
from util import choice_with_weight


class Exp3Forecaster():
    """Forecaster using the Exp3-Algorithm."""
    def __init__(self, options, rounds, highprob=False, lr=None):
        if lr == None:
            self._learning_rate = min(1.0, sqrt((options * log(options)) / (expm1(1) * rounds)))
        else:
            self._learning_rate = lr
        self._weights = np.ones(options, dtype=float)
        self.prob = self._weights / options
        self.gains = np.zeros(options)

    def next_arm(self):
        return choice_with_weight(self.prob)

    def reward(self, arm, value):
        self.gains[arm] += value
        adjusted_gain = value / self.prob[arm]
        options = len(self.prob)
        self._weights[arm] *= exp(self._learning_rate * adjusted_gain / options)
        sum_weights = np.sum(self._weights)
        self.prob = np.array([w / sum_weights for w in self._weights])

    def get_name(self):
        return "Exp3"
