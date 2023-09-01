import json
from HoloOceanUtils.automated_experiments.trial_runner import run_trial, _run_trials_parallel
import numpy as np

# these parameters are guaranteed to be in the parameters dictionary, but may be overwritten by the experiment config
default_parameters = {
    "capture_length": 1.0,
    "num_captures": 10,
    "num_landmarks": 0,
    "auv_range_sigma": 0.1,
    "landmark_range_sigma": 0.1,
    "landmark_bounds": [-100, 100, -100, 100, 0, 0],
    "should_render_viewport": True,
    "ocean_current_multiplier": 0.0,
    "ocean_current_offset": [0.0, 0.0, 0.0],
    "ocean_current_field_index": 0,
    "diver_command_random_seed": None,
    "pyfg_save_location": "~/",
    "pyfg_save_name": "trial"
}

def json_to_dict(file_name):
    experiment_config = open(file_name)
    data = json.load(experiment_config)
    return data

def run_experiment(file_name):
    config = json_to_dict(file_name)
    for i, trial in enumerate(config["trials"]):
        # get default parameters
        parameters = default_parameters.copy()

        # get shared parameters
        # shared parameters override default parameters
        for property in config["shared_parameters"]:
            parameters[property] = config["shared_parameters"][property]

        # get trial parameters
        # trial parameters override default and shared parameters
        for property in trial:
            parameters[property] = trial[property]

        if 'scenario' not in parameters:
            raise Exception("trial ", i, ": scenario must be defined in experiment config")

        run_trial(i, parameters)

# not working as intended
# acoustic beacon sensors are sharing memory between trials when they shouldn't be
def _run_experiment_parallel(file_name):
    config = json_to_dict(file_name)
    each_trial_parameters = []
    for i, trial in enumerate(config["trials"]):
        # get default parameters
        parameters = default_parameters.copy()

        # get shared parameters
        # shared parameters override default parameters
        for property in config["shared_parameters"]:
            parameters[property] = config["shared_parameters"][property]

        # get trial parameters
        # trial parameters override default and shared parameters
        for property in trial:
            parameters[property] = trial[property]

        if 'scenario' not in parameters:
            raise Exception("trial ", i, ": scenario must be defined in experiment config")

        each_trial_parameters.append(parameters)
    _run_trials_parallel(each_trial_parameters)

if __name__ == "__main__":
    run_experiment("pyfg_files/experiment_3/experiment_3.json")