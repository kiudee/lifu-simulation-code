#!/usr/bin/env python
# vim: et tabstop=8 shiftwidth=4 softtabstop=4

import random

def choice_with_weight(dist):
    rnd = random.random() * sum(dist)
    for i, w in enumerate(dist):
        rnd -= w
        if rnd < 0:
            return i
