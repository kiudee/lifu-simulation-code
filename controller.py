#!/usr/bin/env python
# vim: et tabstop=8 shiftwidth=4 softtabstop=4

import numpy as np


class Controller():
    """This is the controller for the simulation.

    These are the tasks it has to accomplish:
     * Interact with learner (ask for next arm, pass reward)
     * Interact with environment:
        - pass it guess and probability distribution of learner
        - recieve reward vector
     * Track gains for each option to calculate cumulative regret for learner
     *
    """

    def __init__(self, arms, rounds, environment, repetitions):
        """The simulation controller gets passed all it needs to setup the
        environment.
        Note that the learner is later given to the run command.
        """
        # Initialize environment:
        #  * Initialize array of best cumulative arm for each round
        self.arms = arms
        self.rounds = rounds
        self.environment = environment
        self._envname = self.environment(self.arms, self.rounds).get_name()
        self.repetitions = repetitions

    def run(self, learner, pbar):
        runs = []

        for rep in range(self.repetitions):
            lrn = learner(self.arms, self.rounds)
            env = self.environment(self.arms, self.rounds)
            name = lrn.get_name()
            this_run = []
            cum_rewards = np.zeros(self.arms)
            lrn_cum_reward = 0.0
            for t in range(self.rounds):
                pbar.update(pbar.currval + 1)
                arm = lrn.next_arm()
                rewards = env.get_rewards(lrn.prob)
                lrn_reward = rewards[arm]
                env.pass_guess(arm)
                env.pass_dist(lrn.prob)
                lrn.reward(arm, lrn_reward)

                # Calculate regret
                lrn_cum_reward += lrn_reward
                cum_rewards += rewards
                regret = np.max(cum_rewards) - lrn_cum_reward
                this_run.append(regret)
            del lrn
            runs.append(this_run)

        return (name, np.array(runs))

    def get_filename(self):
        return "{}r_{}Rep_{}arms_{}env".format(self.rounds, self.repetitions,
                                               self.arms,
                                               self._envname)
