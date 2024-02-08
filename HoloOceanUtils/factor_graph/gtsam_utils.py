from os import makedirs
from os.path import isfile, dirname, isdir
from typing import Dict, Optional, Tuple, List
import time
import holoocean
import numpy as np
import math as math
import itertools

from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.solver_utils import SolverResults, save_to_tum
from py_factor_graph.utils.logging_utils import logger
#from py_factor_graph.calibrations.range_measurement_calibration import get_linearly_calibrated_measurements

from ra_slam.utils.solver_utils import LM_SOLVER, ISAM2_SOLVER
from ra_slam.solve_mle_gtsam import solve_mle_gtsam
from ra_slam.utils.gtsam_utils import GtsamSolverParams

# These functions are impletmented in trial_runner.tick_trial such that they
# iterate at some frequency relative to the overall scenario tick rate.
# The functions can be called at different rates, provided that they are
# called at a lower frequency than the fg_solver.

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
    # Calibrate range measurements:
    # uncalibrated_measurements = fg.range_measurements
    # calibrated_measurements = get_linearly_calibrated_measurements(uncalibrated_measurements)
    # fg.range_measurements = calibrated_measurements
    # Solver parameters and initialization
    solver_params = GtsamSolverParams(
        init_technique="compose",
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
    return gtsam_results

def extract_current_pose(
    gtsam_results: SolverResults):
    """Extracts the pose of the agents from the gtsam_results. Must parse the 
    solved_poses list separately to get pose for desired agent.

    Args:
        gtsam_results (SolverResults): The solver results data structure

    Returns:
        solved_poses (list): Contains the most recent pose of all agents in the pyfg
    """
    solved_poses = []
    for pose_chain in gtsam_results.pose_chain_names:
        last_pose_key = pose_chain[-1]
        last_pose = gtsam_results.poses[last_pose_key]
        solved_poses.append(last_pose)
    return solved_poses

def extract_translation(
        solved_poses,):
    solved_trans = []
    for pose in solved_poses:
        tx = pose[0][3]
        ty = pose[1][3]
        tz = pose[2][3]
        t = tx, ty, tz
        solved_trans.append(t)
    return solved_trans

def save_intermediate_gtsam_to_tum(
        gtsam_results: SolverResults,
        filepath: str,
        strip_extension: bool = False, 
        verbose: bool = False) -> List[str]:
    """Collects the last pose of each intermediate gtsam solved results and saves to .tum.
    
    Args: 
        gtsam_results (Solver_Results): The solver results data structure
        filepath (str): The .tum filepath
        strip_extension (bool, optional): Whether to strip the file extension
        and replace with ".tum". This should be set to true if the file
        extension is not already ".tum". Defaults to False.

    Returns:
        List(str): The output tum files  
        """
    assert (
        gtsam_results.pose_chain_names is not None
    ), "Pose_chain_names must be provided for multi robot trajectories"
    acceptable_pose_chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".replace("L", "")
    # TODO: Add support for exporting without pose_chain_names

    save_files = []
    for pose_chain in gtsam_results.pose_chain_names:
        if len(pose_chain) == 0:
            continue
        pose_chain_letter = pose_chain[0][0]  # Get first letter of first pose in chain
        assert (
            pose_chain_letter in acceptable_pose_chain_letters
        ), "Pose chain letter must be uppercase letter and not L"

        # Removes extension from filepath to add tum extension
        if strip_extension:
            filepath = filepath.split(".")[0] + ".tum"

        assert filepath.endswith(".tum"), "File extension must be .tum"
        modified_path = filepath.replace(".tum", f"_{pose_chain_letter}.tum")

        # if file already exists we won't write over it (typically)
        if verbose and isfile(modified_path) and "/tmp/" not in modified_path:
            logger.warning(f"{modified_path} already exists, overwriting")

        if not isdir(dirname(modified_path)):
            makedirs(dirname(modified_path))
        
        # If the intermediate .tum already exists, add a new line:
        if isfile(modified_path):
            current_pose_key = pose_chain[-1]
            pose_times = gtsam_results.pose_times[current_pose_key]
            translations = gtsam_results.translations[current_pose_key]
            quats = gtsam_results.rotations_quat[current_pose_key]
            with open(modified_path, "a+") as f:
                trans_solve = translations
                if len(trans_solve) == 2:
                    tx, ty = trans_solve
                    tz = 0.0
                elif len(trans_solve) == 3:
                    tx, ty, tz = trans_solve
                else:
                    raise ValueError(
                        f"Solved for translation of wrong dimension {len(trans_solve)}"
                    )
                quat_solve = quats
                qx, qy, qz, qw = quat_solve
                i = pose_times
                f.write(
                    f"{i:6f} {tx:.5f} {ty:.5f} {tz:.5f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n"
                )
                # f.write(f"{i} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
        # If we haven't written an intermediate .tum, write it:
        else:
            pose_times = gtsam_results.pose_times
            translations = gtsam_results.translations
            quats = gtsam_results.rotations_quat
            with open(modified_path, "w") as f:
                for pose_key in pose_chain:
                    trans_solve = translations[pose_key]
                    if len(trans_solve) == 2:
                        tx, ty = trans_solve
                        tz = 0.0
                    elif len(trans_solve) == 3:
                        tx, ty, tz = trans_solve
                    else:
                        raise ValueError(
                            f"Solved for translation of wrong dimension {len(trans_solve)}"
                        )
                    quat_solve = quats[pose_key]
                    qx, qy, qz, qw = quat_solve
                    i = pose_times[pose_key]
                    f.write(
                        f"{i:6f} {tx:.5f} {ty:.5f} {tz:.5f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n"
                    )
                    # f.write(f"{i} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
            if verbose and "/tmp/" not in modified_path:
                logger.info(f"Wrote: {modified_path}")
            save_files.append(modified_path)
    return save_files

def in_situ_gtsam(
    fg: FactorGraphData,
    method: str,
    selected_cost: str,
    counter: int,
    minimum_measurements: int,
    pose_estimate_freq: int,
    tum_save_freq: int,
):
    """Called in trial runner to provide in-situ optimization of available measurements
    and provide position estimates to the diver.
    Optimizes factor graph, then extracts the current pose and covariance ellipsoid."""

    start_counts = 300 * minimum_measurements
    pose_estimate_rate = 300 * pose_estimate_freq
    tum_save_rate = 300 * tum_save_freq

    # Optimize factor graph at specified rate
    agent_B_est = []
    if counter >= start_counts and counter %pose_estimate_rate == 0:
        gtsam_results = fg_solver(fg, method, selected_cost)
        current_poses = extract_current_pose(gtsam_results)
        current_translations = extract_translation(current_poses)
        agent_A_est = current_translations[0]
        agent_B_est = current_translations[1]
        #print("B Pose", current_poses[1])
        #print("B Loc Est: ", agent_B_est)
    # Save intermediate optimization to .tum at specified rate
    if counter >= start_counts and counter %tum_save_rate == 0:
        filepath = "/home/morrisjp/Documents/git/HoloOceanUtils/HoloOceanUtils/automated_experiments/example/experiment_1/gtsam_int.tum"
        save_intermediate_gtsam_to_tum(gtsam_results,filepath)
    return agent_B_est

# TODO: Build a data structure that collects the agent B estimates, agent B location data, and covariance ellipses
    # so we can analyze where the traj falls in the ellipses?

# TODO: Consider using some variety of custom initialization
    # improve depth error based results
    # or use a depth prior?

# TODO: Update vehicle controllers to use the navigation updates from gtsam
    # specifically consider basing AUV circling behavior off the int_gtsam