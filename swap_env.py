#!/usr/bin/env python
# vim: et tabstop=8 shiftwidth=4 softtabstop=4

import numpy as np
from math import sqrt, log, expm1, floor


class SwapEnvironment():


    def __init__(self, arms, rounds):
        self._arms = arms
        self._rounds = rounds
        self._round = 0
        self._swap_every = int(floor(log(rounds)))

    def get_rewards(self, dist):
        rewards = np.zeros(self._arms)
        if self._round % self._swap_every == 0:
            swap = True
        else:
            swap = False
        rewards[0] = 0 if swap else 1
        rewards[1] = 1 if swap else 0
        self._round += 1;
        return rewards

    def pass_guess(self, arm):
        pass

    def pass_dist(self, dist):
        pass

    def get_name(self):
        return "Swap"
