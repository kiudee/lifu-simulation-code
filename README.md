# Required software
To be able to run the simulation you need these components.
Note that this is the configuration the experiments were run on.
So it may be possible to run the simulation on older versions.
* Python 3.2.3
* NumPy 1.6.2
* SciPy 0.10.1
* Matplotlib 1.1.1

# Running a simulation
The main file responsible for running a simulation is called `plotter.py`.

Look for the line `# Parameters:`. Here you can set the following parameters:

* `rounds`: The number of rounds/trials in each simulation
* `repetitions`: How many times do you want to repeat the simulation?
* `environment`: Here choose one of these environment/opponent models:
  * `BernoulliEnvironment` is an environment with 2 bernoulli-distributed arms (p1 = 0.5, p2 = 0.6)
  * `SwapEnvironment` implements the static _Flip_-strategy (oblivious enemy)
  * `EvilEnvironment` implements the adaptive _Flip_-strategy (general enemy)

Running the simulation using `python plotter.py` will create 3 files in the subdirectory `./sim/`:

1. A pickled data file containing the whole simulation
2. A pdf showing a plot of the average cumulative regret over time
3. A pdf showing a boxplot of the cumulative regret