Here are the tools for automated holoocean experiments. Parameters of the experiment are defined in a json file, with trial parameters overriding shared parameters, which override default parameters (default parameters are defined in experiment_runner.py).

It is currently set up for collection of factor graphs, but can be easily modified for other types of data collection.

When designing your own experiments you will likely need to modify trial_runner.py to accept new parameters.

In the example folder there is an example of how to run an experiment and visualize the factor graph it collects.