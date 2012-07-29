#!/usr/bin/env python
# vim: et tabstop=8 shiftwidth=4 softtabstop=4 cc=79

import os
import io
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.stats import trimboth
import progressbar as pb
from simple_env import SimpleEnvironment
from bernoulli_env import BernoulliEnvironment
from evil_env import EvilEnvironment
from swap_env import SwapEnvironment
from INForecaster import INForecaster
from Exp3Forecaster import Exp3Forecaster
from controller import Controller


colors = ["#1f77b4",
          "#ff7f0e",
          "#2ca02c",
          "#d62728"]


class Plotter():

    def __init__(self, controller, algorithms):
        self._algos = algorithms
        self._sim = controller
        matplotlib.rcParams.update({"font.size": 22})

    def run(self):
        progress_length = len(self._algos) * \
                          self._sim.rounds * \
                          self._sim.repetitions
        pbar = pb.ProgressBar(widgets=[pb.AnimatedMarker(), " ",
                                       pb.Percentage(), pb.Bar(), " ",
                                       pb.Timer()],
                              maxval=progress_length).start()

        data_container = dict()
        for algo in self._algos:
            name, data = self._sim.run(algo, pbar)
            data_container[name] = data
        pbar.finish()
        filename = self._get_filename(self._sim.get_filename())
        self._make_boxplot(data_container, filename)
        self._make_plot(data_container, filename)
        self._write_to_disk(data_container, filename)

    def _make_boxplot(self, dc, filename):
        all_data = []
        all_labels = []
        max_val = 0
        min_val = 1e10
        for name, data in dc.items():
            all_labels.append(name)
            sums = data.T[-1]
            max_val = max(np.max(sums), max_val)
            min_val = min(np.min(sums), min_val)
            all_data.append(sums)
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(111)
        plt.subplots_adjust(left=0.2, right=.9, top=.9, bottom=0.2)

        # Boxplot with settings:
        bp = plt.boxplot(all_data, widths=0.3)
        plt.setp(bp["boxes"], color="black", linewidth=2)
        plt.setp(bp["whiskers"], color="black", linewidth=2)
        plt.setp(bp["fliers"], color="red", marker="+", ms=12, linewidth=3,
                 markeredgewidth=2)

        ax1.yaxis.grid(True, linestyle="-", which="major", color="lightgrey",
                       alpha=0.5)
        ax1.set_axisbelow(True)
        ax1.set_xlabel("Verfahren")
        ax1.set_ylabel("Kumulativer Regret")
        ax1.set_yscale("log")


        # Boxes with colors:
        for i in range(len(dc.items())):
            box = bp["boxes"][i]
            boxCoords = [(box.get_xdata()[j], box.get_ydata()[j])
                         for j in range(5)]
            boxPolygon = Polygon(boxCoords, facecolor=colors[i])
            ax1.add_patch(boxPolygon)
            med = bp["medians"][i]
            medianX = []
            medianY = []
            for j in range(2):
                medianX.append(med.get_xdata()[j])
                medianY.append(med.get_ydata()[j])
                plt.plot(medianX, medianY, "k")

        around = round((max_val - min_val)*0.05)
        ax1.set_ylim(min_val-around, max_val+around)
        xticknames = plt.setp(ax1, xticklabels=all_labels)
        plt.setp(xticknames)
        null = fig.savefig(filename + "_box.pdf", dpi=400, bbox_inches="tight")

    def _make_plot(self, dc, filename):
        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_subplot(111)
        plt.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.135)
        col_index = 0
        for name, data in dc.items():
            mean = np.array([np.mean(trimboth(np.sort(x), 0.25)) for x in data.T])
            plt.plot(mean, color=colors[col_index], label=name, linewidth=2)
            col_index += 1
        ax1.set_xlabel("Runde")
        ax1.set_ylabel("Kumulativer Regret")
        plt.legend(loc=0) # 0=best, 4=lower right
        null = fig.savefig(filename + "_plot.pdf", dpi=400,
                           bbox_inches="tight")

    def _get_filename(self, filename):
        tmpname = "sim/" + filename
        uniq = 0
        while (os.path.isfile(tmpname + ".txt")):
            tmpname = "sim/" + filename + str(uniq)
            uniq += 1
        return tmpname

    def _write_to_disk(self, dc, filename):
        with io.open(filename + ".txt", "wb") as datafile:
            pickle.dump(dc, datafile)


if __name__ == "__main__":
    # Parameters:
    rounds = 10000
    repetitions = 200
    # Possible Environments:
    #  * BernoulliEnvironment
    #  * SwapEnvironment (static Flip)
    #  * EvilEnvironment (adaptive Flip)
    environment = BernoulliEnvironment

    algos = [Exp3Forecaster, INForecaster]
    sim = Controller(arms=2, rounds=rounds, environment=environment,
                     repetitions=repetitions)
    plotter = Plotter(sim, algos)
    plotter.run()


