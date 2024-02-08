import time
import holoocean
import numpy as np
import HoloOceanUtils.automated_experiments.controller as controller
from HoloOceanUtils.factor_graph.factor_actor import FactorGraphCollector
from HoloOceanUtils.factor_graph.gtsam_utils import in_situ_gtsam
#from HoloOceanUtils.factor_graph.sam_io import write_results_to_tum
from ra_slam.utils.solver_utils import LM_SOLVER, ISAM2_SOLVER
from py_factor_graph.utils.solver_utils import SolverResults
from py_factor_graph.io.pyfg_text import save_to_pyfg_text

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

        # counter to update factor graph
        self.counter = 0

        # Legacy random diver seed
        # if parameters["diver_command_random_seed"] == None:
        #      self.diver_command_random_generator = np.random
        # else:
        #      self.diver_command_random_generator = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(parameters["diver_command_random_seed"])))

        # self.diver_command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def tick_trial(self, counter, start_time):
        # Initiate modem message every 50 ticks
        if counter %50 == 0:
            # AUV (0) recieves divers (1) bearing
            # AUV will have azimuth to diver, elevation to diver, range and its depth
            self.env.send_acoustic_message(1, 0, "MSG_RESPX", counter)

        # Advance trial counter    
        state = self.env.tick()
        self.counter += 1

        # Add measurements to factor graph
        if (self.counter >= int(float(self.env._ticks_per_sec) * self.parameters["capture_length"])) and \
             "AcousticBeaconSensor" in state['A']:
            self.factor_graph_collector.update_graph(state, self.now)
            self.counter = 0

        # Add actor states to factor graph
        self.factor_graph_collector.update_actors(state, self.dt)
# TODO: Improve in-situ feedback from gtsam:
        # will need to collect all current poses and covariance ellipsoids at those poses
        # Optimize existing pyfg
        # Waits for 6 measurements before attempting to solve, then solves after every msg
        #current_pose_est = in_situ_gtsam(self.factor_graph_collector.pyfg,LM_SOLVER,"all",counter,6,1,10)
        #actor_B_pose = current_pose_est
        # Test function for GTSAM inputs:
        # This isn't right (needs to change every 300 counts)
        # if counter >= 1800 and counter %300 == 0:
        #     loc_error = actor_B_pose-state["B"]["LocationSensor"]
        #     print("Localization Error: ", f"{loc_error}")
        # pose_updates = [[0, 0, -20]]
        # if counter >= 1800 and counter %300 == 0:
        #     actor_B_pose = state["B"]["LocationSensor"]
        #     pose_updates.append(actor_B_pose)
        # else:
        actor_B_pose = []
        # gtsam_pose = pose_updates[-1]
        # #print("gtsam_pose: ", gtsam_pose)

        # Update actor control commands
        for actor in self.factor_graph_collector.actors:
            if actor.name == "A":
                self.env.act("A", np.append(controller.auv_command(self.parameters, state, actor.prev_pos_error, actor.sum_pos_error, self.dt, counter, start_time, self.now),[0.0, 0.0, 0.0]))
            if actor.name == "B":
                self.env.act("B", np.append(controller.diver_command_open(self.parameters, state, actor.prev_pos_error, actor.sum_pos_error, self.dt, counter, start_time, self.now, actor_B_pose),[0.0, 0.0, 0.0]))      
             
        # Required to implement random diver motion (with constant random seed)
        # if counter % self.env._ticks_per_sec == 0:
        #     self.diver_command = controller.choose_random_commands(10, self.diver_command_random_generator)
        # self.env.act("B", self.diver_command)
            
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