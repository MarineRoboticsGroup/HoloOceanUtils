import time
import holoocean
from HoloOceanUtils.factor_graph.factor_actor import FactorGraphCollector
import numpy as np
import HoloOceanUtils.automated_experiments.controller as controller

class TrialRunner:
    def __init__(self, parameters, index):
        if isinstance(parameters['scenario'], str):
            self.env = holoocean.make(parameters['scenario'])
        elif isinstance(parameters['scenario'], dict):
            self.env = holoocean.make(scenario_cfg=parameters['scenario'])
        else:
            raise Exception("scenario must be a string or dict")

        self.env.should_render_viewport(parameters["should_render_viewport"])

        self.env.set_ocean_current_multiplier(parameters["ocean_current_multiplier"])

        self.env.set_ocean_current_offset(parameters["ocean_current_offset"])

        self.env.set_ocean_current_field_index(parameters["ocean_current_field_index"])

        self.env.reset()

        self.dt = 1.0 / float(self.env._ticks_per_sec)
        self.now = 0.0

        self.factor_graph_collector = FactorGraphCollector(self.env, parameters, index)

        self.parameters = parameters
        self.index = index

        if parameters["diver_command_random_seed"] == None:
            self.diver_command_random_generator = np.random
        else:
            self.diver_command_random_generator = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(parameters["diver_command_random_seed"])))

        self.diver_command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def tick_trial(self, counter, start_time):
        state = self.env.tick()

        # add to factor graph
        if counter % int(float(self.env._ticks_per_sec) * self.parameters["capture_length"]) == 0:
            # self.env.send_acoustic_message(1, 0, "MSG_RESPX", 'howdy!')
            self.factor_graph_collector.update_graph(state, self.now)

        # update actors
        self.factor_graph_collector.update_actors(state, self.dt)

        for actor in self.factor_graph_collector.actors:
            if actor.name == "A":
                self.env.act("A", np.append(controller.auv_command(self.parameters, state, actor.prev_pos_error, actor.sum_pos_error, self.dt, start_time, self.now),[0.0, 0.0, 0.0]))

        if counter % self.env._ticks_per_sec == 0:
            self.diver_command = controller.choose_random_commands(10, self.diver_command_random_generator)

        self.env.act("B", self.diver_command)

        self.now += self.dt

        if self.factor_graph_collector.done_collecting:
            return True
        else:
            return False


def run_trial(index, parameters):
    start_time = time.time()
    counter = 0

    trial_runner = TrialRunner(parameters, index)

    while True:
        if trial_runner.tick_trial(counter, start_time):
            break
        counter += 1

# not working as intended
# acoustic beacon sensors are sharing memory between trials when they shouldn't be
def _run_trials_parallel(each_trial_parameters):
    start_time = time.time()
    counter = 0

    trial_runners = []
    for index, parameters in enumerate(each_trial_parameters):
        trial_runners.append(TrialRunner(parameters, index))

    while True:
        all_done = True
        for trial_runner in trial_runners:
            if trial_runner.tick_trial(counter, start_time) == False:
                all_done = False
        if all_done:
            break

        counter += 1