from params_proto import PrefixProto
import numpy as np
import math


class Cfg(PrefixProto, cli=False):
    class data(PrefixProto, cli=False):
        collect_data = False
    
    class sim(PrefixProto, cli=False):
        use_real_robot = False
        num_envs = 1
        use_gpu = True
        show_gui = True
        sim_device = 'cuda:0' if use_gpu else 'cpu'
        use_gripper = True

    def update_parms():
        if Cfg.sim.use_real_robot:
            Cfg.sim.num_envs = 1
            Cfg.sim.use_gpu = False
            Cfg.sim.show_gui = False
            Cfg.sim.sim_device = "cpu"
        else:
            from isaacgym import gymapi  # gymapi should be imported before torch
            Cfg.motor_controller.reset_time = 1.5  # less time to reset in simulation

    class logging(PrefixProto, cli=False):
        log_info = True
        log_info = False
        log_interval = 100

    class motor_controller(PrefixProto, cli=False):
        dt = 0.0025
        reset_time = 3.0
        power_protect_level = 10

        # kp, kd: [hip, thigh, calf, manipulator_joint_1, manipulator_joint_2, manipulator_joint_3]
        class real(PrefixProto, cli=False):
            kp = np.array([200, 200, 200, 50, 5, 0.05])
            kd = kp * 0.01 - np.array([0., 0., 0., 50, 5, 0.05]) * 0.008

            # for locomotion
            kp_stance_loco = np.array([30, 30, 30, 50, 5, 0.05])
            kd_stance_loco = np.array([1, 1, 1, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002
            kp_swing_loco = np.array([30, 30, 30, 50, 5, 0.05])
            kd_swing_loco = np.array([1, 1, 1, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002

            # for manipulation
            kp_stance_mani = np.array([60, 60, 60, 50, 5, 0.05])
            kd_stance_mani = np.array([2, 2, 2, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002
            kp_swing_mani = np.array([60, 60, 60, 50, 5, 0.05])
            kd_swing_mani = np.array([2, 2, 2, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002

            # for bi-manipulation
            kp_bimanual_switch = np.array([30., 30., 30., 30., 30., 30., 80., 80., 80., 80., 80., 80.])
            kd_bimanual_switch = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
            kp_bimanual_command = np.array([30., 30., 30., 30., 30., 30., 100., 100., 100., 100., 100., 100.])
            kd_bimanual_command = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])

        class sim(PrefixProto, cli=False):
            kp = np.array([200, 200, 200, 50, 5, 0.05])
            kd = kp * 0.01 - np.array([0., 0., 0., 50, 5, 0.05]) * 0.008

            # for locomotion
            kp_stance_loco = np.array([30, 30, 30, 50, 5, 0.05])
            kd_stance_loco = np.array([1, 1, 1, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002
            kp_swing_loco = np.array([30, 30, 30, 50, 5, 0.05])
            kd_swing_loco = np.array([1, 1, 1, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002

            # for manipulation
            kp_stance_mani = np.array([60, 60, 60, 50, 5, 0.05])
            kd_stance_mani = np.array([2, 2, 2, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002
            kp_swing_mani = np.array([60, 60, 60, 50, 5, 0.05])
            kd_swing_mani = np.array([2, 2, 2, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002

            # for bi-manipulation
            kp_bimanual_switch = np.array([30., 30., 30., 30., 30., 30., 80., 80., 80., 80., 80., 80.])
            kd_bimanual_switch = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
            kp_bimanual_command = np.array([30., 30., 30., 30., 30., 30., 100., 120., 100., 100., 120., 100.])
            kd_bimanual_command = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 3.0, 2.5, 2.5, 3.0, 2.5])


    class wbc(PrefixProto, cli=False):
        # mani_max_dq is currently not used
        # mani_max_delta_q = 0.08
        # mani_max_dq = 100.0
        # mani_max_ddq = 60

        # mani_max_delta_q_safe = 0.06
        # mani_max_dq_safe = 100.
        # mani_max_ddq_safe = 80

        # mani_max_delta_q_gripper_joint0 = 0.10
        # mani_max_dq_gripper_joint0 = 100.0
        # mani_max_ddq_gripper_joint0 = 100.0

        # mani_max_delta_q_gripper_joint0_safe = 0.08
        # mani_max_dq_gripper_joint0_safe = 100
        # mani_max_ddq_gripper_joint0_safe = 120.0

        # for scooping
        mani_max_delta_q = 0.30
        mani_max_dq = 200.0
        mani_max_ddq = 120

        mani_max_delta_q_safe = 0.30
        mani_max_dq_safe = 200.
        mani_max_ddq_safe = 160

        mani_max_delta_q_gripper_joint0 = 0.30
        mani_max_dq_gripper_joint0 = 200.0
        mani_max_ddq_gripper_joint0 = 120.0

        mani_max_delta_q_gripper_joint0_safe = 0.30
        mani_max_dq_gripper_joint0_safe = 200
        mani_max_ddq_gripper_joint0_safe = 160.0

        # for tossing
        # mani_max_delta_q = 0.60
        # mani_max_dq = 200.0
        # mani_max_ddq = 120

        # mani_max_delta_q_safe = 0.60
        # mani_max_dq_safe = 200.
        # mani_max_ddq_safe = 160

        # mani_max_delta_q_gripper_joint0 = 0.60
        # mani_max_dq_gripper_joint0 = 200.0
        # mani_max_ddq_gripper_joint0 = 120.0

        # mani_max_delta_q_gripper_joint0_safe = 0.60
        # mani_max_dq_gripper_joint0_safe = 200
        # mani_max_ddq_gripper_joint0_safe = 160.0

        loco_max_delta_q = 3.0
        loco_max_dq = 1000.0
        loco_max_ddq = 1000.0

        interpolate_decay_ratio = 0.6
        interpolate_threshold = 0.002
        interpolate_decay_ratio_safe = 0.6
        interpolate_threshold_safe = 0.002
        interpolate_schedule_step = 50

        singularity_thresh = 0.0026
        
        ground_collision_gripper_thresh = 0.011
        ground_collision_foot_thresh = 0.006
        ground_collision_knee_thresh = 0.014

        class real(PrefixProto, cli=False):
            # locomotion - base
            base_position_kp_loco = np.array([1., 1., 1.]) * 100
            base_position_kd_loco = np.array([1., 1., 1.]) * 10
            base_orientation_kp_loco = np.array([1., 1., 1.]) * 100
            base_orientation_kd_loco = np.array([1., 1., 1.]) * 10

            # manipulation - base
            base_position_kp_mani = np.array([1., 1., 1.]) * 100
            base_position_kd_mani = np.array([1., 1., 1.]) * 1
            base_orientation_kp_mani = np.array([1., 1., 1.]) * 100
            base_orientation_kd_mani = np.array([1., 1., 1.]) * 1

            # both - footeef, eef
            footeef_position_kp = np.array([1., 1., 1.]) * 100
            footeef_position_kd = np.array([1., 1., 1.]) * 10
            eef_orientation_kp = np.array([1., 1., 1.]) * 100
            eef_orientation_kd = np.array([1., 1., 1.]) * 10

        class sim(PrefixProto, cli=False):
            # locomotion - base
            base_position_kp_loco = np.array([1., 1., 1.]) * 100
            base_position_kd_loco = np.array([1., 1., 1.]) * 10
            base_orientation_kp_loco = np.array([1., 1., 1.]) * 100
            base_orientation_kd_loco = np.array([1., 1., 1.]) * 10

            # manipulation - base
            base_position_kp_mani = np.array([1., 1., 1.]) * 100
            base_position_kd_mani = np.array([1., 1., 1.]) * 1
            base_orientation_kp_mani = np.array([1., 1., 1.]) * 100
            base_orientation_kd_mani = np.array([1., 1., 1.]) * 1

            # both - footeef, eef
            footeef_position_kp = np.array([1., 1., 1.]) * 100
            footeef_position_kd = np.array([1., 1., 1.]) * 10
            eef_orientation_kp = np.array([1., 1., 1.]) * 100
            eef_orientation_kd = np.array([1., 1., 1.]) * 10


    class gripper(PrefixProto, cli=False):
        vid_pid = "0403:6014"
        baudrate = 1000000

        motor_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        reset_pos_sim = np.array([-np.pi, -0.05, 0., 0.02, -np.pi, 0.05, 0., 0.02])
        min_position = [-3.25, -2.4, -3.15, 0., -3.25, -0.6, -3.15, 0.]  # change the joint limit of the loco-manipulator's first link from 3.17 to 3.25
        max_position = [1.74, 0.6, 3.15, -0.5, 1.74, 2.4, 3.15, -0.5]
        min_velocity = [-20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0]
        max_velocity = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
        min_torque = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        max_torque = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        reset_time = 0.5
        s2r_scale = np.array([-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0])
        s2r_offset = np.array([-np.pi, 0.0, 0.0, 0.0, np.pi, 0.0, 0.0, 0.0])
        # s2r_offset = np.array([-np.pi, 0.0, np.pi/2, 0.0, np.pi, 0.0, -np.pi/2, 0.0])

        dof_idx = [0, 1, 2, 4, 5, 6]
        gripper_idx = [3, 7]

        arm_1_idx = [0, 1, 2, 3]
        arm_2_idx = [4, 5, 6, 7]

        update_gripper_freq = 200

        kI = np.zeros(len(motor_ids))
        kP = np.array([8, 6, 4, 12, 8, 6, 4, 12]) * 128
        kD = np.array([65, 40, 30, 90, 65, 40, 30, 90]) * 16

        kP = np.array([8, 6, 6, 6, 8, 6, 6, 6]) * 128
        kD = np.array([65, 40, 40, 100, 65, 40, 40, 100]) * 16

        # Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        curr_lim = 900
        gripper_delta_max = 0.2

        gripper_des_pos_sim_topic = '/gripper_des_pos_sim'
        gripper_cur_state_sim_topic = '/gripper_cur_state_sim'


    class bimanual_trajectory(PrefixProto, cli=False):
        recording_fps_mul_motor_controller_fps = 2
        move_gripper_foot_time = 1.5

        class with_gripper(PrefixProto, cli=False):
            trajectory_path = 'bimanual_trajectory/bimanual_w_gripper_data.pkl'
            stand_up_start_time = 0.1
            stand_up_end_time = 2.5
            stabilize_time = 0.5
            stand_down_start_time = 5.4
            stand_down_end_time = 7.2

        class without_gripper(PrefixProto, cli=False):
            trajectory_path = 'bimanual_trajectory/bimanual_wo_gripper_data.pkl'
            stand_up_start_time = 0.4
            stand_up_end_time = 2.2
            stabilize_time = 0.5
            stand_down_start_time = 5.6
            stand_down_end_time = 7.4


    class locomotion(PrefixProto, cli=False):
        early_touchdown_phase_threshold = 0.5
        lose_contact_phase_threshold = 0.1

        gait_params = [2, np.pi, np.pi, 0., 0.45]
        gait_params = [2, np.pi, np.pi, 0., 0.4]
        desired_pose = np.array([0., 0., 0.28, 0.0, 0., 0.])
        # desired_pose = np.array([0., 0., 0.28, 0.0, 0., 0.])
        desired_velocity = np.array([0., 0., 0.0, 0.0, 0.0, 0.])
        foot_landing_clearance_sim = 0.0
        foot_landing_clearance_real = 0.01
        foot_height = 0.06

    class bimanual_manipulation(PrefixProto, cli=False):
        body_action_dim = 6
        leg_action_dim = 6
        swing_foot_names = ['FR', 'FL']
        eef_action_dim = 3

    class manipulation(PrefixProto, cli=False):
        # Body
        body_action_dim = 6
        locoman_body_state = np.array([-0.05, 0.03, 0.02, .0, .0, .0])  # doesn't count
        initialization_time_sequence = np.array([1.5, 1.5])
        # Leg
        leg_action_dim = 3
        swing_foot_names = ['FL']
        swing_foot_names = ['FR']
        manipulate_leg_idx =  {'FR':0, 'FL':1, 'RR':2, 'RL':3}[swing_foot_names[0]] # 0:FR, 1:FL, 2:RR, 3:RL
        # swing_foot_names = ['FR']
        # manipulate_leg_idx = 0 # 0:FR, 1:FL, 2:RR, 3:RL
        # swing_foot_name = ['FR', 'FL']
        # manipulate_leg_idx = 0 # 0:FR, 1:FL, 2:RR, 3:RL
        # EEF
        eef_action_dim = 3
        # non_manipulate_eef_reference_frame = 'world'
        # non_manipulate_eef_reference_frame = 'foot'
        non_manipulate_eef_reference_frame = 'joint'

        # body state (6d), leg state in initial foot frame (3d), reset_signal(1d), contact_state(1d), swing_leg(1d), duration (1d), stay time (1d)
        # right triangle
        # loco_locoman_switch_actions = np.array([[-0.05, -0.03, 0.02, .0, .0, .0, 0., 0., 0., 1, 1, -1, 1., .5],
        #                                         ])
        # # isosceles triangle
        # loco_locoman_switch_actions = np.array([[-0.025, -0.0, 0.01, .0, .0, .0, 0., 0., 0., 1, 1, -1, 0.8, 1.1],
        #                                         [0.0, 0.0, 0.0, .0, .0, .0, 0.0, 0.0, 0.04, 1, 0, 0, 0.5, 0.01],
        #                                         [0.0, 0.0, 0.0, .0, .0, .0, 0.03, 0.08, 0.0, 1, 0, 0, 0.8, 0.01],
        #                                         [0.0, 0.0, 0.0, .0, .0, .0, 0.0, 0.0, -0.04, 1, 0, 0, 0.5, 0.01],
        #                                         [0.025, 0.0, 0.0, .0, .0, .0, 0.0, 0.0, 0., 1, 1, -1, 0.8, .1],
        #                                         ])

        grp = np.array([-np.pi, -0.05, 0., 0.02, -np.pi, 0.05, 0., 0.02])
        if manipulate_leg_idx:
            # body state (6d), leg state in initial foot frame (3d), eef_joint_pos(6d), mode(1d), reset_signal(1d), contact_state(1d), swing_leg(1d), gripper_closing(2d), duration (1d), stay time (1d)
            loco_locoman_switch_actions = np.array([[-0.05, -0.03, 0.02, .0, .0, .0, 0., 0., 0., grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 0, 1, 1, -1, grp[3], grp[7], 1.5, 0.2],
                                                    ])
            loco_manipulation_actions  =  np.array([
                                                    [0., 0., 0., .0, .0, .0, 0., 0.08, -0.11, grp[0], grp[1], grp[2], np.pi, 0.05, 0., 1, 0, 0, -1, grp[3], grp[7], 1.5, .5],
                                                    [0., 0., 0., 0., 0., 0., 0.1, 0.1, -0.15, grp[0], grp[1], grp[2], np.pi, 0.05, 0., 1, 0, 0, -1, grp[3], grp[7], 1.8, 1.],
                                                    [0., 0., 0., 0.1, 0., 0.1, 0.2, 0.03, -0.20, grp[0], grp[1], grp[2], np.pi, 0.05, 0., 1, 0, 0, -1, grp[3], grp[7], 1.8, 1.],
                                                    [0., 0., 0., 0., -0.2, 0., 0.26, 0.1, -0.1, grp[0], grp[1], grp[2], np.pi, 0.05, 0., 1, 0, 0, -1, grp[3], grp[7], 1.8, 1.],
                                                    [0., 0., 0., .0, .0, .0, 0., 0.08, -0.13, grp[0], grp[1], grp[2], np.pi, 0.05, 0., 1, 0, 0, -1, grp[3], grp[7], 1.5, .5],
                                                    [0., 0., 0., .0, .0, .0, 0., 0., 0., grp[0], grp[1], grp[2], np.pi, 0.05, 0., 1, 0, 0, -1, grp[3], grp[7], 1., 0.2],  # back to the initial foot position
                                                    ])
        else:
            loco_locoman_switch_actions = np.array([[-0.05, 0.03, 0.02, .0, .0, .0, 0., 0., 0., grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 0, 1, 1, -1, grp[3], grp[7], 1., .2],
                                                    ])
            loco_manipulation_actions  =  np.array([
                                                    [0., 0., 0., .0, .0, .0, 0., -0.08, -0.12, np.pi/2, 0., 0., grp[4], grp[5], grp[6], 1, 0, 0, -1, grp[3], grp[7], 1.5, 0.5],
                                                    [0., 0., 0., 0., 0., 0., 0.38, -0.24, -0.20, -1.5, 0., 0., grp[4], grp[5], grp[6], 2, 0, 0, -1, grp[3], grp[7], 1.8, 1.],
                                                    [0., 0., 0., -0.1, 0., 0.1, 0.38, -0.22, -0.15, 0., -1.5, 0., grp[4], grp[5], grp[6], 2, 0, 0, -1, grp[3], grp[7], 1.8, 1.],
                                                    [0., 0., 0., 0., -0.2, 0., 0.55, -0.20, -0.0, 0, -1.5, -0.1, grp[4], grp[5], grp[6], 2, 0, 0, -1, grp[3], grp[7], 1.8, 1.],
                                                    [0., 0., 0., .0, .0, .0, 0., -0.08, -0.15, np.pi, 0., 0., grp[4], grp[5], grp[6], 1, 0, 0, -1, grp[3], grp[7], 1.8, 0.5],
                                                    [0., 0., 0., .0, .0, .0, 0., 0., 0., np.pi, 0., 0., grp[4], grp[5], grp[6], 1, 0, 0, -1, grp[3], grp[7], 1., 0.5],  # back to the initial foot position
                                                    ])

        if manipulate_leg_idx:
            # body state (6d), footeef_position(3d), eef_joint_pos(6d), mode(1d), reset_signal(1d), contact_state(1d), swing_leg(1d), gripper_angles(2d), duration (1d), stay time (1d)
            loco_locoman_switch_actions = np.array([[-0.05, -0.03, 0.02, .0, .0, .0, 0., 0., 0., grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 0, 1, 1, -1, grp[3], grp[7], 1.5, 0.2],
                                                    ])
            loco_manipulation_actions  =  np.array([
                                                    [0., 0., 0., .0, .0, .0, 0., 0.08, -0.20, grp[0], grp[1], grp[2], -np.pi/8, 0., 0., 1, 0, 0, -1, grp[3], grp[7], 1.5, .5],
                                                    [0., 0., 0., 0, 0., 0., 0.38, 0.28, -0.20, grp[0], grp[1], grp[2], 1.3, -0.2, 0, 2, 0, 0, -1, grp[3], grp[7], 1.8, 1.],
                                                    [0., 0., 0., 0.1, 0., 0.1, 0.38, 0.28, -0.20, grp[0], grp[1], grp[2], 0., -1.5, 0., 2, 0, 0, -1, grp[3], grp[7], 1.8, 1.],
                                                    [0., 0., 0., 0., -0.2, 0., 0.55, 0.20, -0.0, grp[0], grp[1], grp[2], 0, -1.5, 0.1, 2, 0, 0, -1, grp[3], grp[7], 1.8, 1.],
                                                    [0., 0., 0., .0, .0, .0, 0., 0.08, -0.20, grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 1, 0, 0, -1, grp[3], grp[7], 1.5, .5],
                                                    [0., 0., 0., .0, .0, .0, 0., 0., 0., grp[0], grp[1], grp[2],grp[4], grp[5], grp[6], 1, 0, 0, -1, grp[3], grp[7], 2., 0.2],  # back to the initial foot position
                                                    ])
        else:
            loco_locoman_switch_actions = np.array([[-0.05, 0.03, 0.02, .0, .0, .0, 0., 0., 0., grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 0, 1, 1, -1, grp[3], grp[7], 1., .2],
                                                    ])
            loco_manipulation_actions  =  np.array([
                                                    [0., 0., 0., .0, .0, .0, 0., -0.08, -0.12, -np.pi/5, 0., 0., grp[4], grp[5], grp[6], 1, 0, 0, -1, grp[3], grp[7], 1.5, 0.5],
                                                    [0., 0., 0., 0., 0., 0., 0.38, -0.24, -0.20, -1.5, 0., 0., grp[4], grp[5], grp[6], 2, 0, 0, -1, grp[3], grp[7], 1.8, 1.],
                                                    [0., 0., 0., -0.1, 0., 0.1, 0.38, -0.22, -0.15, 0., -1.5, 0., grp[4], grp[5], grp[6], 2, 0, 0, -1, grp[3], grp[7], 1.8, 1.],
                                                    [0., 0., 0., 0., -0.2, 0., 0.55, -0.20, -0.0, 0, -1.5, -0.1, grp[4], grp[5], grp[6], 2, 0, 0, -1, grp[3], grp[7], 1.8, 1.],
                                                    [0., 0., 0., .0, .0, .0, 0., -0.08, -0.15, grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 1, 0, 0, -1, grp[3], grp[7], 1.8, 0.5],
                                                    [0., 0., 0., .0, .0, .0, 0., 0., 0., grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 1, 0, 0, -1, grp[3], grp[7], 1., 0.5],  # back to the initial foot position
                                                    ])



        # # # idx_0: mode(1d): stance-0, foot_manipulation-1, eef_manipulation-2, locomotion-3, loco-manipulation-4
        # # # idx_1-4: foot contact state(4d): [FR, FL, RR, RL], swing-0, stance-1
        # # # idx_5-10: torso state(6d): [x, y, z, roll, pitch, yaw]
        # # # idx_11-13: foot position or eef position(3d*4): [x, y, z]*4  # currently only use one, given based on the hip frame, but convert it on the world frame for wbc
        # # # idx_14-19: eef orientation(3d*4): [roll, pitch, yaw]  # currently only use two, only when swing legs on the manipulation mode based on the world frame, else based on the calf frame
        # # # idx_20: action-duration(1d)
        # # # idx_21: stay-duration(1d)
        # # # idx_22: reset_signal(1d)


    class loco_manipulation(PrefixProto, cli=False):
        loco_manipulation_commander = False
        locomotion_only = False
        swing_foot_names = ['FL']
        swing_foot_names = ['FR']
        manipulate_leg_idx =  {'FR':0, 'FL':1}[swing_foot_names[0]] # 0:FR, 1:FL, 2:RR, 3:RL
        desired_eef_state = np.array([100, 100, 0.12, 0., 0., 100])
        desired_eef_rpy_w = np.array([0., -np.pi, 0.]) if manipulate_leg_idx == 0 else np.array([0., -np.pi, 0.])


    class commander(PrefixProto, cli=False):
        joystick_command_topic = "/joystick_command"
        body_p_command_type = 'delta'
        # body_p_command_type = 'scale'
        body_v_command_type = 'delta'
        # body_v_command_type = 'scale'
        robot_reset_pose_topic = "/robot_reset_pose"

        eef_task_space = 'world'
        # eef_task_space = 'eef'

        human_command_topic = "/human_command"
        auto_policy_topic = "/auto_policy_command"

        reset_manipulator_when_switch = True
        # reset_manipulator_when_switch = False

        body_pv_scale = np.array([0.0005, 0.0005, 0.002, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01, 0.1])
        footeef_p_scale = 0.001
        eef_joint_pos_scale = 0.005
        eef_rpy_scale = 0.005
        locomotion_height_range = [0.15, 0.40]

        gripper_angle_scale = 1.0
        gripper_angle_range = [0.01, 1.8]

        class sim_limit(PrefixProto, cli=False):
            body_pv_limit = np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.4, 0.6, 0.6, 0, 0, 0, 1.0])

        class real_limit(PrefixProto, cli=False):
            body_pv_limit = np.array([0.06, 0.06, 0.2, 0.3, 0.3, 0.4, 0.2, 0.2, 0, 0, 0, 0.3])


    class teleoperation(PrefixProto, cli=False):

        robot_state_topic = "/robot_state"

        receive_action_topic = "/receive_action"
        
        robot_reset_topic = "/robot_reset"
        
        ready_action_topic = "/ready_action"
                
        auto_mode_topic = "/auto_mode"

        teleop_view_topic = "/teleop_view_sim"
        
        fpv_height = 960
        fpv_width = 1280
        
        # fpv_height = 420
        # fpv_width = 660
        
        shm_name = "pAlad1n3traiT"
        wrist_shm_name = "SaHlofoLinA"
        
        # the supported resoltion of the camera
        # (720, 1280)
        realsense_resolution = (480, 640)
        stereo_rgb_resolution = (720, 1280)
        rgb_resolution = (720, 1280)
        
        # the resolution of the images to be seen and recorded
        head_view_resolution = (480, 640)
        wrist_view_resolution = (480, 640)

        class human_teleoperator(PrefixProto, cli=False):
            SERVER_HOST = '189.176.158.13'
            SERVER_PORT = 12345
            mode = 3  # 0:stance, 1:right-gripper manipulation, 2: left-gripper manipulation 3: bi-manual manipulation, 4:None
            mode_updata_topic = "/mode_update"
            fsm_to_teleop_mode_mapping = [0, 1, 2, 1, 2, 4, 4, 3]

            # thresholds are used to filter out the human shaking and noise
            body_xyz_threshold = 0.003
            body_rpy_threshold = 0.003
            eef_xyz_threshold = 0.001
            eef_rpy_threshold = 0.005
            gripper_angle_threshold = 0.01
            
            # needs to keep fixed for training and rollout for now
            body_xyz_scale = 0.5
            body_rpy_scale = 1.0
            eef_xyz_scale = 1.0
            eef_rpy_scale = 1.0
            gripper_angle_scale = 1.0
            # gripper_angle_range = [0.01, 1.8]

            if mode == 1 or mode == 2:
                body_xyz_scale = 0.0
                body_rpy_scale = 1.0
            
            body_xyz_max_step = 0.001
            body_rpy_max_step = 0.0015
            eef_xyz_max_step = 0.001
            eef_rpy_max_step = 0.002
            gripper_angle_max_step = 0.03

            body_r_range = [-0.40, 0.40]
            body_p_range = [-0.25, 0.25]
            body_y_range = [-0.40, 0.40]


    class fsm_switcher(PrefixProto, cli=False):
        fsm_state_topic = "/fsm_state"

        # all the motions are based on the front_right leg
        class stance_and_manipulation(PrefixProto, cli=False):
            base_movement = np.array([-0.05, 0.03, 0.02, .0, .0, .0])  # for right
            foot_movement = np.array([0., 0., 0.04]) # orig z is 0.04
            manipulator_angles = np.array([-np.pi/8, 0., 0.])
            base_action_time = 1.0
            foot_action_time = 1.0
            eef_action_time = 2.0

            # use the reset_pos_sim from gripper
            grp = np.array([-np.pi, -0.05, 0., 0.02, -np.pi, 0.05, 0., 0.02])
            # body state (6d), footeef_position(3d), eef_joint_pos(6d), mode(1d), reset_signal(1d), contact_state(1d), swing_leg(1d), gripper_angles(2d), duration (1d), stay time (1d)
            # will be updated in the constructor
            manipulation_to_stance_actions = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 1, 0, 0, -1, grp[3], grp[7], 2.0, .1],
                                                        [0., 0., 0., .0, .0, .0, -1, -1, -1, grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 1, 1, 0, -1, grp[3], grp[7], 1.5, .1],
                                                        [-1, -1, 0., 0., 0., 0., 0., 0., 0., grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 0, 0, 1, -1, grp[3], grp[7], 1.0, .1],
                                                        ])
            stance_to_manipulation_gripper_delta = np.zeros(8)

        class stance_and_locomanipulation(PrefixProto, cli=False):
            manipulate_leg_idx = 0  # 0:FR, 1:FL, 2:RR, 3:RL
            # use the reset_pos_sim from gripper
            grp = np.array([-np.pi, -0.05, 0., 0.02, -np.pi, 0.05, 0., 0.02])
            manipualtor_angles_ready_to_hold = grp.copy()
            manipualtor_angles_ready_to_hold[manipulate_leg_idx*4] = -np.pi * 3/4
            stance_to_locomanipulation_actions = np.array([[0., 0., 0., .0, .0, .0, -1, -1, -1, grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 0, 0, 1, -1, grp[3], grp[7], 1.5, 1.0],
                                                        ])
            
            transition_time = 1.5
            stablize_time = 1.

    class fsm_resetter(PrefixProto, cli=False):
        fsm_state_topic = "/fsm_state"

        class manipulation(PrefixProto, cli=False):
            base_reset_pose = np.array([0., 0., 0., 0., 0., 0.])
            foot_reset_offset = np.array([0.03, 0., 0.05])
            manipulator_reset_angles = np.array([-np.pi/8, 0., 1.57])
            manipulator_rest_angles = np.array([-np.pi, -0.05, 0., -np.pi, 0.05, 0.])
            gripper_reset_angles = np.array([0.02])
            gripper_rest_angles = np.array([0.02, 0.02])

            base_reset_pose_range = [np.array([0., 0., 0., 0., 0., -0.03]), np.array([0., 0., 0., 0., 0.05, 0.03])]
            foot_reset_offset_range = [np.array([0.02, 0., 0.05]), np.array([0.05, 0., 0.07])]
            manipulator_reset_angles_range = [np.array([-np.pi/8 - 0.02, -0.02, 1.50]), np.array([-np.pi/8 + 0.02, 0.02, 1.58])]

        class bimanual_manipulation(PrefixProto, cli=False):
            foot_reset_pose_range = [np.array([0.14, -0.105, -0.005, 0.14, 0.105, -0.005]), np.array([0.17, -0.135, 0.015, 0.17, 0.135, 0.015])]
            manipulator_reset_angles_range = [np.array([0.13, -0.02, 1.50, 0.13, -0.02, 1.50]), np.array([0.17, 0.02, 1.58, 0.17, -0.02, 1.50])]
            gripper_reset_angles = np.array([0.02, 0.02])

    class asset(PrefixProto, cli=False):
        urdf_path = 'asset/go1/urdf/go1.urdf'


    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        # rot = [0.0, 0.0, 0.149438, 0.98877]  # x,y,z,w [quat] for testing the sim state estimator
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

    class reward(PrefixProto, cli=False):
        reward_coeff = 0.5
        max_reward = 1.0
    

    def get_sim_config(use_penetrating_contact=True):
        from isaacgym import gymapi
        from ml_collections import ConfigDict
        use_gpu = Cfg.sim.use_gpu
        # dt = Cfg.motor_controller.dt
        # dt = 0.0005
        dt = 0.0025
        # simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.use_gpu_pipeline = use_gpu
        sim_params.dt = dt
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UpAxis(gymapi.UP_AXIS_Z)
        sim_params.gravity = gymapi.Vec3(0., 0., -9.81)
        sim_params.physx.use_gpu = use_gpu
        sim_params.physx.num_subscenes = 0  #default_args.subscenes
        sim_params.physx.num_threads = 10
        sim_params.physx.solver_type = 1 # TGS
        sim_params.physx.num_position_iterations = 6 #4 improve solver convergence
        sim_params.physx.num_velocity_iterations = 1 # keep default
        if use_penetrating_contact:
            sim_params.physx.contact_offset = 0.
            sim_params.physx.rest_offset = -0.004
        else:
            sim_params.physx.contact_offset = 0.01
            sim_params.physx.rest_offset = 0.
        sim_params.physx.bounce_threshold_velocity = 0.2  #0.5 [m/s]
        sim_params.physx.max_depenetration_velocity = 100.0
        sim_params.physx.max_gpu_contact_pairs = 2**23  #2**24 needed for 8000+ envs
        sim_params.physx.default_buffer_size_multiplier = 5
        sim_params.physx.contact_collection = gymapi.ContactCollection(2)  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
        # plane parameters
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.
        # create config
        config = ConfigDict()
        config.sim_device = 'cuda:0' if use_gpu else 'cpu'
        config.physics_engine = gymapi.SIM_PHYSX
        config.sim_params = sim_params
        config.plane_params = plane_params
        config.action_repeat = round(Cfg.motor_controller.dt / sim_params.dt)
        config.dt = dt
        config.env_spacing = 2.0
        return config
    

    def get_asset_config():
        from isaacgym import gymapi
        from ml_collections import ConfigDict
        config = ConfigDict()
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = 3
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        # asset_options.flip_visual_attachments = True  # for dae
        asset_options.flip_visual_attachments = False  # for stl
        asset_options.fix_base_link = False
        # asset_options.fix_base_link = True
        asset_options.density = 0.001
        asset_options.angular_damping = 0.
        asset_options.linear_damping = 0.
        asset_options.max_angular_velocity = 1000.
        asset_options.max_linear_velocity = 1000.
        asset_options.armature = 0.
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        config.asset_options = asset_options
        config.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        return config
    
    def get_act_config():
        from ml_collections import ConfigDict
        import os
        config = ConfigDict()
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(cur_dir)        
        config.dataset_dir = os.path.join(parent_dir, 'demonstrations/real_robot')
        config.num_episodes = 60
        config.episode_len = 1000
        config.camera_names = ["main"]
        return config


