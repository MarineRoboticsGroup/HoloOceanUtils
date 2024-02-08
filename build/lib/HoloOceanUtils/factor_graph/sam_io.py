import os
from os.path import join, exists
import pickle
import copy
from typing import List, Dict, Optional, Tuple
from signal import default_int_handler
from attrs import define, field
from functools import partial

import time
import holoocean
import numpy as np
import math as math
import itertools
import pandas as pd
import logging
import coloredlogs
import matplotlib.pyplot as plt
from matplotlib import ticker

from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.solver_utils import SolverResults, save_to_tum
from py_factor_graph.io.pyfg_text import read_from_pyfg_text

from ra_slam.utils.solver_utils import LM_SOLVER, ISAM2_SOLVER
from ra_slam.solve_mle_gtsam import solve_mle_gtsam
from ra_slam.utils.gtsam_utils import GtsamSolverParams

from evo.tools.file_interface import FileInterfaceException, read_tum_trajectory_file
from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D
from evo.core.geometry import umeyama_alignment
from evo.tools import plot

from HoloOceanUtils.factor_graph.online_sam import fg_solver


# TODO: Improve io of simulator results to provide good comparison between ground truth, IMU, and GTSAM
    # Developed from CORA evaluate_utils.py
    # Revise plot function to allow for plotting the agents as two different colors
    # TODO: Save stats as a text file

VALID_ROBOT_CHARS = [chr(ord("A") + i) for i in range(26)]
VALID_ROBOT_CHARS.remove("L")

ALIGNED_RESULTS_FNAME = "aligned_results.pickle"
APE_ERROR_STATS_FNAME = "ape_error_stats.pickle"

# Trajectory Types
GT_STR = "gt_traj"
GTSAM = "gtsam"
GTSAM_LEADING_STRS = [GTSAM]
ALL_LEADING_STRS = [GT_STR] + GTSAM_LEADING_STRS

### File Path Construction ###
def _get_joined_tum_fpath(results_dir: str, leading_str: str) -> str:
    """Get the joined tum file path"""
    assert leading_str in ALL_LEADING_STRS
    fname = f"joined_{leading_str}.tum"
    fpath = join(results_dir, fname)
    return fpath

def _get_tum_fpaths(results_dir: str, leading_str: str, num_robots: int) -> List[str]:
    """Get the different .tum fpaths in the results directory corresponding to the leading_str

    Args:
        results_dir (str): the directory containing the CORA tum files
        leading_str (str): the leading string for the tum file
        num_robots (int): the number of robots

    Returns:
        List[str]: the list of CORA tum files
    """
    assert leading_str in ALL_LEADING_STRS
    trailing_chars = VALID_ROBOT_CHARS[:num_robots]
    tum_paths = [join(results_dir, f"{leading_str}_{c}.tum") for c in trailing_chars]
    return tum_paths
### Logger ###
logger = logging.getLogger(__name__)
field_styles = {
    "filename": {"color": "green"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
}
coloredlogs.install(
    logger=logger,
    level="INFO",
    fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
    field_styles=field_styles,
)
### Data Classes ###
GT_TRAJ_LABEL = "Ground Truth"
GTSAM_TRAJ_LABEL = "GTSAM"

PREFERRED_LABEL_ORDERING = [
    GT_TRAJ_LABEL,
    GTSAM_TRAJ_LABEL,
]

TRAJ_LABEL_TO_COLOR = {
    GT_TRAJ_LABEL: "gray",
    GTSAM_TRAJ_LABEL: "green",
}

@define
class ResultsPoseTrajCollection:
    """The results of a single trial"""

    gt_traj: PoseTrajectory3D = field()
    gtsam: PoseTrajectory3D = field()

    def __str__(self) -> str:
        return f"ResultsPoseTrajCollection with trajectories: {self.all_traj_names}"
    
    @property
    def num_comp_trajs(self) -> int:
        return len(self.comparison_trajs)

    @property
    def ordered_comp_name_color_triplets(
        self,
    ) -> Tuple[List[PoseTrajectory3D], List[str], List[str]]:
        comp_trajs = self.comparison_trajs
        comp_names = self.comparison_traj_names
        comp_colors = self.comparison_traj_colors
        assert len(comp_trajs) == len(comp_names) == len(comp_colors)

        triplets = list(zip(comp_trajs, comp_names, comp_colors))

        # sort by preferred name ordering
        def _get_name_ordering(name: str) -> int:
            return PREFERRED_LABEL_ORDERING.index(name)

        triplets.sort(key=lambda x: _get_name_ordering(x[1]))

        sorted_comp_trajs, sorted_comp_names, sorted_comp_colors = zip(*triplets)
        assert (
            len(sorted_comp_trajs) == len(sorted_comp_names) == len(sorted_comp_colors)
        )
        assert all([isinstance(traj, PoseTrajectory3D) for traj in sorted_comp_trajs])
        assert all([name in PREFERRED_LABEL_ORDERING for name in sorted_comp_names])
        assert all(
            [color in TRAJ_LABEL_TO_COLOR.values() for color in sorted_comp_colors]
        )
        return sorted_comp_trajs, sorted_comp_names, sorted_comp_colors
    
    @property
    def multi_robot_trajs_available(self) -> bool:
        return (
            self.gtsam is not None
        )

    @property
    def comparison_trajs(self) -> List[PoseTrajectory3D]:
        comparison_trajs = [
            self.gtsam,
        ]
        if self.multi_robot_trajs_available:
            comparison_trajs.append(self.gtsam)

        return comparison_trajs

    @property
    def comparison_traj_names(self) -> List[str]:
        names = [
            GTSAM_TRAJ_LABEL,
        ]
        if self.multi_robot_trajs_available:
            names.append(GTSAM_TRAJ_LABEL)
        else:
            # remove the GT- prefix from the name
            names = [name.replace("GT-", "") for name in names]

        return names
    
    @property
    def comparison_traj_colors(self) -> List[str]:
        return [TRAJ_LABEL_TO_COLOR[name] for name in self.comparison_traj_names]
    
    @property
    def comparison_traj_color_map(self) -> Dict[str, str]:
        return {
            name: color
            for name, color in zip(
                self.comparison_traj_names, self.comparison_traj_colors
            )
        }
    
    @property
    def all_trajs(self) -> List[PoseTrajectory3D]:
        comp_trajs = self.comparison_trajs
        all_trajs = [self.gt_traj] + comp_trajs
        return all_trajs

    @property
    def all_traj_names(self) -> List[str]:
        comp_traj_names = self.comparison_traj_names
        all_traj_names = [GT_TRAJ_LABEL] + comp_traj_names
        return all_traj_names

    @property
    def all_traj_colors(self) -> List[str]:
        return [TRAJ_LABEL_TO_COLOR[name] for name in self.all_traj_names]

    @property
    def all_traj_color_map(self) -> Dict[str, str]:
        return {
            name: color
            for name, color in zip(self.all_traj_names, self.all_traj_colors)
        }
    
def _check_traj_error_df(
    expected_indices: List[str], expected_cols: List[str], df: pd.DataFrame
):
    """Check that the traj error df is valid"""
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == len(expected_indices)
    assert df.shape[1] == len(
        expected_cols
    ), f"Expected {expected_cols} but got {df.columns}"
    assert set(df.index) == set(
        expected_indices
    ), f"Expected {expected_indices} but got {df.index}"
    assert set(df.columns) == set(expected_cols)  

@define
class TrajErrorDfs:
    """pandas DataFrames representing the trajectory error statistics for a
    single experiment. All errors are w.r.t. aligned trajectories

    Sample DF
    ---------
    #               rmse      mean    median       std       min       max          sse
    # Odometry  0.969083  0.854065  0.896745  0.457924  0.045657  2.026719  1647.219439
    # GTSAM     0.911305  0.777825  0.661960  0.474831  0.050085  1.906614  1456.655153

    """

    expected_indices: List[str] = field()
    expected_cols: List[str] = field()
    rot_error_df: pd.DataFrame = field()
    trans_error_df: pd.DataFrame = field()
    pose_error_df: pd.DataFrame = field()
    cmap: Dict[str, str] = field()
    trajs_are_indices: bool = field()  # traj names are either indices or columns

    # after initialization validate that the indices are correct
    def __attrs_post_init__(self):
        _check_traj_error_df(
            self.expected_indices, self.expected_cols, self.rot_error_df
        )
        _check_traj_error_df(
            self.expected_indices, self.expected_cols, self.trans_error_df
        )
        _check_traj_error_df(
            self.expected_indices, self.expected_cols, self.pose_error_df
        )
        # check number of keys in cmap is the same as the number of indices
        if self.trajs_are_indices:
            expected_keys = set(self.expected_indices)
        else:
            expected_keys = set(self.expected_cols)
        assert (
            set(self.cmap.keys()) == expected_keys
        ), f"Expected {expected_keys} but got {set(self.cmap.keys())}"

    def __str__(self):
        val = ""
        val += f"Rotation Error:\n{self.rot_error_df}\n"
        val += f"Translation Error:\n{self.trans_error_df}\n"
        return val
    
    def plot_ape_errors(self, rot_ax: plt.Axes, tran_ax: plt.Axes) -> None:
        _plot_rot(self.rot_error_df, rot_ax, cmap=self.cmap)
        _plot_trans(self.trans_error_df, tran_ax, cmap=self.cmap)

### File Utils ###
        
def get_pyfg_file_name_in_dir(target_dir: str) -> str:
    file_names = os.listdir(target_dir)
    candidate_pyfg_files = [f for f in file_names if f.endswith(".pyfg")]
    no_pyfg_files_found = len(candidate_pyfg_files) == 0
    if no_pyfg_files_found:
        raise FileNotFoundError(
            f"Could not find PyFG file in {target_dir} - existing files: {file_names}"
        )
    elif len(candidate_pyfg_files) > 1:
        raise FileNotFoundError(
            f"Multiple candidate PyFG files in {target_dir} - existing files: {file_names}"
        )

    return candidate_pyfg_files[0]

def check_dir_ready_for_evaluation(target_dir: str) -> None:
    """Raises errors if the right files cannot be found to run analysis

    Args:
        target_dir (str): the directory containing the right files

    Raises:
        FileNotFoundError: couldn't find factor_graph.pickle
        FileNotFoundError: couldn't find the cora tum files
    """
    pyfg_file_name = get_pyfg_file_name_in_dir(target_dir)
    pyfg_file = join(target_dir, pyfg_file_name)
    logger.info(f"Found PyFG file: {pyfg_file}")
    pyfg = read_from_pyfg_text(pyfg_file)
    num_robots = pyfg.num_robots

### Main Functions ###
def get_aligned_traj_results_in_dir(
    results_dir: str, use_cached_results: bool, selected_cost: str = "all"
) -> ResultsPoseTrajCollection:
    check_dir_ready_for_evaluation(results_dir)
    aligned_results_pickle_path = join(results_dir, ALIGNED_RESULTS_FNAME)
    if exists(aligned_results_pickle_path) and use_cached_results:
        logger.debug(
            f"Found cached aligned results, loading from {aligned_results_pickle_path}"
        )
        aligned_results = pickle.load(open(aligned_results_pickle_path, "rb"))
        return aligned_results
    else:
        logger.info(f"Aligning results in {results_dir}")
        pyfg_file_name = get_pyfg_file_name_in_dir(results_dir)
        fg_path = join(results_dir, pyfg_file_name)
        pyfg = read_from_pyfg_text(fg_path)
        num_robots = pyfg.num_robots

        def _all_files_exist(fpaths: List[str]) -> bool:
            return all([exists(f) for f in fpaths])

        def _should_write_new_tum_files(
            leading_str: str, using_cached_results: bool, num_robots: int
        ) -> bool:
            if using_cached_results:
                cached_files_found = _all_files_exist(
                    _get_tum_fpaths(results_dir, leading_str, num_robots)
                )
                return not cached_files_found
            else:
                return True
            
        # groundtruth tum files
        if _should_write_new_tum_files(GT_STR, use_cached_results, num_robots):
            pyfg.write_pose_gt_to_tum(results_dir)

        # gtsam tum files
    # TODO: Verify that this correctly sources the already made GTSAM .tum.
        # Make the if statement reference a function in online_sam.py
        # So it solves the results using the same params
        gtsam_experiments = copy.deepcopy(GTSAM_LEADING_STRS)
        for gtsam_leading_str in gtsam_experiments:
            if _should_write_new_tum_files(
                gtsam_leading_str, use_cached_results, num_robots
            ):
                logger.info(f"Need to write new {gtsam_leading_str} tum files")
                gtsam_result = fg_solver(pyfg, LM_SOLVER, "range_odom")
                tum_fpath = "/home/morrisjp/Documents/git/HoloOceanUtils/HoloOceanUtils/automated_experiments/example/experiment_1/gtsam.tum"

                save_to_tum(gtsam_result, tum_fpath)
            else:
                logger.debug(f"Found cached {gtsam_leading_str} tum files")
        def _should_write_new_joined_tum(
            leading_str: str,
            using_cached_results: bool,
        ) -> bool:
            assert leading_str in ALL_LEADING_STRS
            if using_cached_results:
                cached_file_found = exists(
                    _get_joined_tum_fpath(results_dir, leading_str)
                )
                return not cached_file_found
            else:
                logger.debug(f"Writing new joined tum files for {leading_str}")
                return True
                    
        # if need to write new joined tum files, do so
        def _join_tum_files(tum_file_paths: List[str], save_path: str) -> None:
            """Joins multiple TUM files into one file

            Args:
                tum_file_paths (List[str]): list of paths to TUM files
                save_path (str): path to save the joined TUM file
            """
            with open(save_path, "w") as f:
                for tum_file_path in tum_file_paths:
                    with open(tum_file_path, "r") as f_tum:
                        for line in f_tum:
                            f.write(line)
            logger.debug(f"Saved joined TUM file to: {save_path}")

        all_trajs_to_show = [GT_STR] + gtsam_experiments
        for leading_str in all_trajs_to_show:
            if _should_write_new_joined_tum(leading_str, use_cached_results):
                _join_tum_files(
                    _get_tum_fpaths(results_dir, leading_str, num_robots),
                    _get_joined_tum_fpath(results_dir, leading_str),
                )

        # get the trajectories from joined tum files
        def _get_traj_from_file(results_dir: str, leading_str: str) -> PoseTrajectory3D:
            joined_tum_path = _get_joined_tum_fpath(results_dir, leading_str)
            logger.debug(
                f"Constructing {leading_str} trajectory from {joined_tum_path}"
            )

            try:
                traj = read_tum_trajectory_file(joined_tum_path)
            except FileInterfaceException as e:
                logger.error(f"Error reading {joined_tum_path}: {e}")
                raise e
            return traj
        joined_trajs: Dict[str, PoseTrajectory3D] = {
            leading_str: _get_traj_from_file(results_dir, leading_str)
            for leading_str in all_trajs_to_show
        }
        gt_traj = joined_trajs[GT_STR]

        # align the trajectories to the ground truth
        def _get_aligned_traj(ref_traj: PoseTrajectory3D, traj: PoseTrajectory3D):
            traj_aligned = copy.deepcopy(traj)
            align_rot, align_trans, _ = umeyama_alignment(
                traj_aligned.positions_xyz.T, ref_traj.positions_xyz.T
            )

            transform = np.eye(4)
            transform[:3, :3] = align_rot
            transform[:3, -1] = align_trans
            traj_aligned.transform(transform)
            return traj_aligned

        def _align_traj_to_gt(traj: PoseTrajectory3D) -> PoseTrajectory3D:
            return _get_aligned_traj(gt_traj, traj)

        aligned_joined_trajs = {
            leading_str: _align_traj_to_gt(traj)
            for leading_str, traj in joined_trajs.items()
            if leading_str != GT_STR
        }

        # group the results
        aligned_results = ResultsPoseTrajCollection(
            gt_traj = gt_traj,
            gtsam = aligned_joined_trajs[GTSAM],
        )
        pickle.dump(aligned_results, open(aligned_results_pickle_path, "wb"))

    return aligned_results

def make_evo_traj_plots(
    aligned_results: ResultsPoseTrajCollection,
    results_dir: str,
    show_plots: bool,
    valid_plot_views: List[plot.PlotMode] = [plot.PlotMode.xyz],
    overlay_river_image: bool = False,
):
    """Make plots comparing the ground truth, cora, and gtsam trajectories

    Args:
        aligned_results (ResultsPoseTrajCollection): the aligned trajectories
        results_dir (str): the directory to save the plots
        show_plots (bool, optional): whether to show the plots. Defaults to False.
    """
    for plot_mode in valid_plot_views:
        # make figure the full screen size
        fig = plt.figure(figsize=(20, 10))

        def _improve_plot_config():
            # increase linewidth and font size
            plt.rcParams["lines.linewidth"] = 4
            plt.rcParams["axes.labelsize"] = 30
            plt.rcParams["xtick.labelsize"] = 30
            plt.rcParams["ytick.labelsize"] = 30
            plt.rcParams["legend.fontsize"] = 20
            plt.rcParams["legend.loc"] = "upper left"

            # some configs just for the xyz plot
            if plot_mode == plot.PlotMode.xyz:
                plt.rcParams["axes.labelpad"] = 20.0
            elif plot_mode == plot.PlotMode.xz:
                plt.rcParams["axes.labelpad"] = 10.0
                plt.rcParams["figure.subplot.hspace"] = 0.05
                plt.rcParams["grid.linewidth"] = 2

            # set grid color: https://matplotlib.org/stable/gallery/color/named_colors.html
            plt.rcParams["grid.color"] = "gray"
            if overlay_river_image:
                plt.rcParams["grid.color"] = "lightgray"
                plt.rcParams["figure.subplot.hspace"] = 0.01

            plt.rcParams["axes.edgecolor"] = "gray"

        _improve_plot_config()

        # make 1xn grid of axes
        #num_plots = aligned_results.num_comp_trajs
        num_plots = 1 # this allows for overiding the comparison plot
        axes_idxs = list(range(num_plots))
        if num_plots <= 4:
            num_rows = 1
            num_cols = num_plots
        else:
            num_rows = 2
            num_cols = num_plots // 2

        subplot_args = [int(f"{num_rows}{num_cols}{i+1}") for i in axes_idxs]
        axes = [
            plot.prepare_axis(fig, plot_mode, subplot_arg=subplot_arg)
            for subplot_arg in subplot_args
        ]

        def _plot_traj(
            ax: plt.Axes, traj: PoseTrajectory3D, name: str, color: str, style: str
        ):
            # find places where timestamps are not monotonically increasing, this
            # is where a new trajectory starts
            timestamps = traj.timestamps
            xyz = traj.positions_xyz
            non_monotonic_timestamp_idxs = np.where(np.diff(timestamps) < 0)[0]
            if len(non_monotonic_timestamp_idxs) > 0:
                logger.debug(
                    f"Found {len(non_monotonic_timestamp_idxs)} non-monotonic timestamps"
                )

            separated_traj_idxs = np.split(
                np.arange(len(timestamps)), non_monotonic_timestamp_idxs + 1
            )
            separate_trajs = [xyz[idxs] for idxs in separated_traj_idxs]

            x_idx, y_idx, z_idx = plot.plot_mode_to_idx(plot_mode)

            # plot all of them but only add the label to the first one
            for traj_cnt, separate_traj in enumerate(separate_trajs):
                x = separate_traj[:, x_idx]
                y = separate_traj[:, y_idx]
                if plot_mode == plot.PlotMode.xyz:
                    z = separate_traj[:, z_idx]
                    ax.plot(
                        x,
                        y,
                        z,
                        style,
                        color=color,
                        label=name if traj_cnt == 0 else None,
                    )
                    plot.set_aspect_equal(ax)
                else:
                    ax.plot(
                        x, y, style, color=color, label=name if traj_cnt == 0 else None
                    )

        gt_traj = aligned_results.gt_traj
        gt_traj_name = GT_TRAJ_LABEL
        gt_traj_color = TRAJ_LABEL_TO_COLOR[gt_traj_name]
        (
            ordered_comparison_trajs,
            ordered_comparison_names,
            ordered_comparison_colors,
        ) = aligned_results.ordered_comp_name_color_triplets
        for idx, traj, name, color in zip(
            axes_idxs,
            ordered_comparison_trajs,
            ordered_comparison_names,
            ordered_comparison_colors,
        ):
            ax = axes[idx]
            _plot_traj(ax, gt_traj, gt_traj_name, gt_traj_color, "--")
            _plot_traj(ax, traj, name, color, "-")
            ax.legend(frameon=True)

            ax.set_facecolor("white")

        if overlay_river_image:
            from .paths import DATA_DIR

            river_image_path = join(DATA_DIR, "river_img.png")
            river_image_path = join(DATA_DIR, "river_2.png")
            river_img = plt.imread(river_image_path)
            # trim the top 10 pixels
            river_img = river_img[10:, :, :]

            xmin = -250
            xmax = 50
            ymin = -110
            ymax = 50

            # show the image with no padding
            for idx in axes_idxs:
                ax = axes[idx]
                ax.imshow(river_img, extent=[xmin, xmax, ymin, ymax], alpha=1, zorder=0)

                # move the lines to the front (but with room for the legend)
                for line in ax.lines:
                    line.set_zorder(10)

                # make sure the legend is on top
                ax.get_legend().set_zorder(20)

            # reduce padding or buffer between plots
            plt.subplots_adjust(left=0.10, right=0.95, wspace=0.05)

        # make all axes the same size
        ax_xmin, ax_xmax = axes[0].get_xlim()
        ax_ymin, ax_ymax = axes[0].get_ylim()
        for ax in axes:
            new_ax_xmin, new_ax_xmax = ax.get_xlim()
            new_ax_ymin, new_ax_ymax = ax.get_ylim()
            ax_xmin = min(ax_xmin, new_ax_xmin)
            ax_xmax = max(ax_xmax, new_ax_xmax)
            ax_ymin = min(ax_ymin, new_ax_ymin)
            ax_ymax = max(ax_ymax, new_ax_ymax)

        # add a bit of padding to the top for the legend
        # (except for when we overlay the river image)
        if not overlay_river_image:
            ax_ymax += 0.1 * (ax_ymax - ax_ymin)
            for ax in axes:
                ax.set_xlim(ax_xmin, ax_xmax)
                ax.set_ylim(ax_ymin, ax_ymax)

        def _get_tick_increment_from_range(minval, maxval) -> int:
            val_range = maxval - minval
            # choose an increment such that there will be 3 or 4 ticks on the axis
            tick_options = [1, 2, 4, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
            for tick_option in tick_options:
                if val_range / tick_option <= 4:
                    return tick_option

            return tick_options[-1]
        
        x_tick_increment = _get_tick_increment_from_range(ax_xmin, ax_xmax)
        y_tick_increment = _get_tick_increment_from_range(ax_ymin, ax_ymax)

        # shift the axes values such that they start at zero and are non-negative
        def shift_formatter(ax_min, tick_increment, x, pos):
            # shift such that the least value is 0 and each tick is rounded down to
            # nearest multiple of tick_increment
            shifted_val = x - ax_min
            rounded_val = np.floor(shifted_val / tick_increment) * tick_increment
            return f"{rounded_val:.0f}"

        shift_formatter_y = ticker.FuncFormatter(
            partial(shift_formatter, ax_ymin, y_tick_increment)
        )
        shift_formatter_x = ticker.FuncFormatter(
            partial(shift_formatter, ax_xmin, x_tick_increment)
        )
        for idx, ax in enumerate(axes):
            cur_row = idx // num_cols
            cur_col = idx % num_cols

            ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_increment))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_increment))
            ax.xaxis.set_major_formatter(shift_formatter_x)
            ax.yaxis.set_major_formatter(shift_formatter_y)

            # only have y labels and ticks on the left column for 2D plots
            if cur_col != 0 and plot_mode != plot.PlotMode.xyz:
                ax.set_yticklabels([])
                ax.set_ylabel("")
                ax.tick_params(axis="y", which="both", length=0)

            # only have x labels and ticks on the bottom row for 2D plots
            if cur_row != num_rows - 1 and plot_mode != plot.PlotMode.xyz:
                ax.set_xticklabels([])
                ax.set_xlabel("")
                ax.tick_params(axis="x", which="both", length=0)

            plot.set_aspect_equal(ax)

            # make sure grid is on
            ax.grid(True, which="both", axis="both")

        traj_plot_path = join(results_dir, f"traj_plot_{plot_mode.name}.png")
        plt.savefig(traj_plot_path, transparent=True, dpi=fig.dpi)
        plt.savefig(
            traj_plot_path.replace(".png", ".svg"),
            format="svg",
            transparent=True,
            dpi=fig.dpi,
        )
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        logger.info(f"Saved trajectories to: {traj_plot_path}")

def clean_df_collection(dfs: TrajErrorDfs) -> TrajErrorDfs:
    stats_to_keep = ["rmse", "max"]
    stat_renaming = {"rmse": "RMSE", "max": "Max"}
    current_indices = dfs.rot_error_df.index.tolist()

    def _clean_df(df) -> pd.DataFrame:
        stats_to_drop = [stat for stat in df.columns if stat not in stats_to_keep]
        new_df = df.swapaxes("index", "columns")
        new_df.drop(stats_to_drop, axis=0, inplace=True)
        new_df.rename(index=stat_renaming, inplace=True)
        return new_df

    expected_indices = [stat_renaming[stat] for stat in stats_to_keep]
    cleaned_tran_df = _clean_df(dfs.trans_error_df)
    cleaned_rot_df = _clean_df(dfs.rot_error_df)
    cleaned_pose_df = _clean_df(dfs.pose_error_df)
    return TrajErrorDfs(
        expected_indices=expected_indices,
        expected_cols=current_indices,
        rot_error_df=cleaned_rot_df,
        trans_error_df=cleaned_tran_df,
        pose_error_df=cleaned_pose_df,
        cmap=dfs.cmap,
        trajs_are_indices=False,
    )

def _set_yticks_only_ints(ax: plt.Axes):
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

def _plot_trans(df_trans: pd.DataFrame, target_ax: plt.Axes, cmap: Dict[str, str]):
    df_trans.plot.bar(ax=target_ax, ylabel="Translation Error (meters)", color=cmap)
    ylim_top_min = 3.0
    if target_ax.get_ylim()[1] < ylim_top_min:
        target_ax.set_ylim(0, ylim_top_min)
    target_ax.set_xticklabels(target_ax.get_xticklabels(), rotation=0)
    target_ax.set_facecolor("white")
    target_ax.grid(visible=False, which="both", axis="x")
    # target_ax.grid(False)
    # _set_yaxis_to_log_if_order_magnitude_range(target_ax)

def _plot_rot(df_rot: pd.DataFrame, target_ax: plt.Axes, cmap: Dict[str, str]):
    df_rot.plot.bar(ax=target_ax, ylabel="Rotation Error (degrees)", color=cmap)
    # ylim_top_min = 110
    # if target_ax.get_ylim()[1] < ylim_top_min:
    #     target_ax.set_ylim(0, ylim_top_min)
    # ylim_top = target_ax.get_ylim()[1]
    # target_ax.set_ylim(0, ylim_top*(1.4))
    _set_yticks_only_ints(target_ax)
    target_ax.legend(frameon=True, loc="upper left")

    target_ax.set_xticklabels(target_ax.get_xticklabels(), rotation=0)
    target_ax.set_facecolor("white")
    target_ax.grid(visible=False, which="both", axis="x")

    # target_ax.grid(False)
    # _set_yaxis_to_log_if_order_magnitude_range(target_ax)

def make_plots_from_error_dfs(
    error_dfs: TrajErrorDfs,
    comparison_traj_color_map: Dict[str, str],
    save_dir: str,
    show_plots: bool = False,
):
    #               rmse      mean    median       std       min       max          sse
    # Odometry  0.969083  0.854065  0.896745  0.457924  0.045657  2.026719  1647.219439
    # GTSAM     0.911305  0.777825  0.661960  0.474831  0.050085  1.906614  1456.655153

    df_trans = error_dfs.trans_error_df
    df_rot = error_dfs.rot_error_df
    df_pose = error_dfs.pose_error_df

    all_stats = df_trans.columns
    df_trans = df_trans.swapaxes("index", "columns")
    df_rot = df_rot.swapaxes("index", "columns")
    df_pose = df_pose.swapaxes("index", "columns")

    # drop all indices but rmse and max
    stats_to_keep = ["rmse", "max"]
    stats_to_drop = [stat for stat in all_stats if stat not in stats_to_keep]
    df_trans.drop(stats_to_drop, axis=0, inplace=True)
    df_rot.drop(stats_to_drop, axis=0, inplace=True)
    df_pose.drop(stats_to_drop, axis=0, inplace=True)

    # rename "rmse" to "RMSE"
    df_trans.rename(index={"rmse": "RMSE"}, inplace=True)
    df_rot.rename(index={"rmse": "RMSE"}, inplace=True)
    df_pose.rename(index={"rmse": "RMSE"}, inplace=True)

    # rename "max" to "Max"
    df_trans.rename(index={"max": "Max"}, inplace=True)
    df_rot.rename(index={"max": "Max"}, inplace=True)
    df_pose.rename(index={"max": "Max"}, inplace=True)

    # set the color map
    color_map = comparison_traj_color_map

    plt.rcParams["grid.color"] = "gray"
    plt.rcParams["axes.edgecolor"] = "gray"
    plt.rcParams["legend.loc"] = "upper left"

    saved_figs = []

    def _save_fig(fpath: str):
        plt.savefig(fpath, transparent=True)
        plt.savefig(fpath.replace(".png", ".svg"), format="svg", transparent=True)
        saved_figs.append(fpath)

    def _set_yaxis_to_log_if_order_magnitude_range(ax: plt.Axes):
        bar_vals = [rect.get_height() for rect in ax.patches]
        bar_val_range = max(bar_vals) / min(bar_vals)
        if bar_val_range > 100:
            ax.set_yscale("log")
            # turn on minor grid lines
            ax.grid(which="minor", axis="y")

    def _plot_pose(target_ax: plt.Axes, cmap: Dict[str, str]):
        df_pose.plot.bar(ax=target_ax, ylabel="Average Pose Error", color=cmap)
        target_ax.set_xticklabels(target_ax.get_xticklabels(), rotation=0)
        # target_ax.grid(False)

    # make bar plots
    default_figsize = (16, 8)

    # make two bar plot with translation and rotation
    def _two_bar_plot():
        fig, axs = plt.subplots(1, 2, figsize=default_figsize)
        _plot_trans(df_trans, axs[0], color_map)
        _plot_rot(df_rot, axs[1], color_map)
        fig.subplots_adjust(wspace=0.3)
        two_bar_plot_path = join(save_dir, "two_bar_plot.png")
        _save_fig(two_bar_plot_path)

    _two_bar_plot()
    if show_plots:
        plt.show()
    else:
        plt.close("all")

    for f in saved_figs:
        logger.info(f"Saved: {f}")

def make_evo_ape_plots_from_trajs(
    aligned_results: ResultsPoseTrajCollection,
    results_dir: str,
    use_cached_error_dfs: bool = False,
    show_plots: bool = False,
):
    collected_error_dfs = get_ape_error_stats_from_aligned_trajs(
        aligned_results, results_dir, use_cached_error_dfs
    )
    make_plots_from_error_dfs(
        collected_error_dfs,
        aligned_results.comparison_traj_color_map,
        results_dir,
        show_plots,
    )
    return collected_error_dfs

def get_ape_error_stats_from_aligned_trajs(
    aligned_results: ResultsPoseTrajCollection,
    results_dir: str,
    use_cached_results: bool,
) -> TrajErrorDfs:
    ape_error_stats_path = join(results_dir, APE_ERROR_STATS_FNAME)
    if use_cached_results and exists(ape_error_stats_path):
        logger.debug(
            f"Found cached APE error stats, loading from {ape_error_stats_path}"
        )
        collected_error_dfs = pickle.load(open(ape_error_stats_path, "rb"))
        return collected_error_dfs

    gt_traj = aligned_results.gt_traj
    (
        comparison_trajs,
        comparison_traj_names,
        _,
    ) = aligned_results.ordered_comp_name_color_triplets

    # get translation error
    translation_errors: List[Dict[str, float]] = []
    for traj in comparison_trajs:
        translation_metric = metrics.APE(metrics.PoseRelation.translation_part)
        translation_metric.process_data((gt_traj, traj))
        translation_stats = translation_metric.get_all_statistics()
        translation_errors.append(translation_stats)

    # get rotation error
    rotation_errors: List[Dict[str, float]] = []
    for traj in comparison_trajs:
        rotation_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
        rotation_metric.process_data((gt_traj, traj))
        rotation_stats = rotation_metric.get_all_statistics()
        rotation_errors.append(rotation_stats)

    # get APE stats
    pose_errors: List[Dict[str, float]] = []
    for traj in comparison_trajs:
        pose_metric = metrics.APE(metrics.PoseRelation.full_transformation)
        pose_metric.process_data((gt_traj, traj))
        pose_stats = pose_metric.get_all_statistics()
        pose_errors.append(pose_stats)

    df_trans = pd.DataFrame(
        translation_errors,
        index=comparison_traj_names,
    )
    df_rot = pd.DataFrame(
        rotation_errors,
        index=comparison_traj_names,
    )
    df_pose = pd.DataFrame(
        pose_errors,
        index=comparison_traj_names,
    )

    all_error_stats = ["rmse", "mean", "median", "std", "min", "max", "sse"]
    collected_error_dfs = TrajErrorDfs(
        expected_indices=comparison_traj_names,
        expected_cols=all_error_stats,
        rot_error_df=df_rot,
        trans_error_df=df_trans,
        pose_error_df=df_pose,
        cmap=aligned_results.comparison_traj_color_map,
        trajs_are_indices=True,
    )

    pickle.dump(collected_error_dfs, open(ape_error_stats_path, "wb"))
    return collected_error_dfs

# Code to Execute:
def scenario_plots(results_dir, selected_cost):
    aligned_results = get_aligned_traj_results_in_dir(results_dir,False,selected_cost)
    make_evo_traj_plots(aligned_results,results_dir,True,[plot.PlotMode.xyz],False)
    make_evo_ape_plots_from_trajs(aligned_results, results_dir, True, True)
    get_ape_error_stats_from_aligned_trajs(aligned_results, results_dir, True)
