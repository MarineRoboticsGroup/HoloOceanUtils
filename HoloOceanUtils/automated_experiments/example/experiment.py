from HoloOceanUtils.automated_experiments.experiment_runner import run_experiment
from py_factor_graph.io.pyfg_text import read_from_pyfg_text


if __name__ == '__main__':
    run_experiment("HoloOceanUtils/automated_experiments/example/experiment_1/experiment_1.json")
    pyfg = read_from_pyfg_text("HoloOceanUtils/automated_experiments/example/experiment_1/test_0.pyfg")
    pyfg.animate_odometry_3d(show_gt=True, draw_range_lines=True, num_timesteps_keep_ranges = 1)
    pyfg = read_from_pyfg_text("HoloOceanUtils/automated_experiments/example/experiment_1/test_1.pyfg")
    pyfg.animate_odometry_3d(show_gt=True, draw_range_lines=True, num_timesteps_keep_ranges = 1)
    pyfg = read_from_pyfg_text("HoloOceanUtils/automated_experiments/example/experiment_1/test_2.pyfg")
    pyfg.animate_odometry_3d(show_gt=True, draw_range_lines=True, num_timesteps_keep_ranges = 1)
