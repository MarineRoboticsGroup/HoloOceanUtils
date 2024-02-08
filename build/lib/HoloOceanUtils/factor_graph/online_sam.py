import time
import holoocean
import numpy as np
import math as math
import itertools

from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.solver_utils import SolverResults, save_to_tum
from ra_slam.utils.solver_utils import LM_SOLVER, ISAM2_SOLVER
from ra_slam.solve_mle_gtsam import solve_mle_gtsam
from ra_slam.utils.gtsam_utils import GtsamSolverParams

# TODO: These functions are impletmented in trial_runner.tick_trial such that they iterate at some
# frequency relative to the overall scenario tick rate.

# TODO: Read the existing factor graph and provide optimized result
def fg_solver(
    fg: FactorGraphData, method: str, selected_cost: str)-> SolverResults:
    """Optimize the current factor graph
    Args:
        fg (FactorGraphData): The intermediate factor graph provieded by the FactorGraphCollector
        method (str): Use to select batch or iterative optimization
            Options: "LM_SOLVER" (batch) or "ISAM2_SOLVER" (iterative)
        selected_cost (str): The factors to use for optimization
            Options: "all", "odom", "bearing-odom", "range-odom"
    Returns: 
        gtsam_results (SolverResults): Optimized results (see solver_utils for structure)
    """
    
    # Solver parameters and initialization
    solver_params = GtsamSolverParams(
        init_technique="gt",
        landmark_init="gt",
        custom_init_file=None,
        init_translation_perturbation=None,
        init_rotation_perturbation=None,
        start_at_gt=True,
    )
    # Solver function
    gtsam_results = solve_mle_gtsam(
        fg,
        solver_params,
        solver=method,
        selected_cost=selected_cost
    )
    #print("gtsam update available")
    # Save the solved results to .tum
    tum_fpath = "/home/morrisjp/Documents/git/HoloOceanUtils/HoloOceanUtils/automated_experiments/example/experiment_1/gtsam_test.tum"
    save_to_tum(gtsam_results,tum_fpath)
    return gtsam_results

# TODO: Parse the gtsam result into position updates for agents
def extract_current_pose(
    gtsam_results: SolverResults):
    # Unsure what type this output should be
    """Extracts the pose of a given agent from the gtsam_results
    Args:
        gtsam_results (SolverResults): The solver results data structure
    Returns:
        solved_poses (list): Contains the most recent pose of all agents in the pyfg
    """
    solved_poses = []
    # iterate process for each agent in pyfg
    for pose_chain in gtsam_results.pose_chain_names:
        # pose_chain is a list of the pose keys
        # gtsam_results.poses is a dict
        #pose_chain_letter = pose_chain[0][0] # confirms that this is iterating through both chains
        #print("Sample pose: ", pose_chain_letter)
        
        # get the last (most recent) pose key in the pose chain
        last_pose_key = pose_chain[-1] # operates correctly
        #print("Sample pose: ", last_pose_key)
        
        # get pose associated with the pose key
        last_pose = gtsam_results.poses[last_pose_key] # correctly retrieves the poses
        #print("Current Pose: ", last_pose)
        
        # add this last pose to a collector list
        solved_poses.append(last_pose)
        #print("Current poses: ", solved_poses)
    # solved_poses should be a list of two elements: the most recent poses of agent A and B
    return solved_poses


# TODO: Update controller to use the navigation updates
    # Need to pass the pose to the controller so that it updates the segment desired position
    # The pose will only reset/update when a new msg comes in
    # really only need to strip out the translation (the first 3) but nice to have more
    # needs to be resilient to not having a gtsam update, since the commands update at 300x the rate
    # could start by providing a vector(dt) that is based on the destination, then overwrite the destination
    
    # The current implementation doesn't really work (passes pose as zero unless there's an update)

# TODO: Things to update in the json
# - Solver method
# - AUV and diver control schemes (not the mode, what are they referencing)
# - gtsam parameters