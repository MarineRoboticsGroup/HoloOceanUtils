from py_factor_graph.io.pyfg_text import read_from_pyfg_text

if __name__ == "__main__":
    # print('hi')
    # points = [(1.0, 0.0, 2.0)]
    # print(*points)
    # xs, ys, zs = zip(*points)
    # pyfg = read_from_pyfg_text("diver_auv_perfect_with_current.pyfg")
    # pyfg.animate_odometry_3d(show_gt=True, draw_range_lines=True)
    pyfg = read_from_pyfg_text("pyfg_files/experiment_3/trial_0.pyfg")
    pyfg.animate_odometry_3d(show_gt=True, pause_interval=1.0, draw_range_lines=True, num_timesteps_keep_ranges = 30)
    pyfg = read_from_pyfg_text("pyfg_files/experiment_3/trial_2.pyfg")
    pyfg.animate_odometry_3d(show_gt=True, pause_interval=1.0, draw_range_lines=True, num_timesteps_keep_ranges = 30)
    # pyfg2 = read_from_pyfg_text("diver_auv_IMU_with_current_2.pyfg")
    # pyfg2.animate_odometry_3d(show_gt=True, pause_interval=0.001, draw_range_lines=True)

