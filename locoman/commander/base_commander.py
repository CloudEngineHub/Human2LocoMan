import numpy as np
from robot.base_robot import BaseRobot
from config.config import Cfg
import rospy
from std_msgs.msg import Float32MultiArray
import time


class BaseCommander:
    def __init__(self, robot: BaseRobot, env_ids=0):
        self._robot = robot
        self._cfg: Cfg = robot._cfg
        self._dt = self._robot._dt
        self._env_ids = env_ids
        self._is_used = False
        self._is_updating_command = False
        self._is_computing_command = False

        # buffers for wbc
        self._action_mode = 0
        self._contact_state = np.ones(4, dtype=bool)
        # [6d pose * 3 (pva)]
        self._desired_body_pva = np.zeros(18)
        # [3 (pva), 3d position * 4 legs ]
        self._desired_footeef_pva_w = np.zeros((3, 12))
        # [2 eef, 3d orientation * 3 (pva)]
        self._desired_eef_pva = np.zeros((2, 9))
        self._desired_eef_pva[0, 0:3] = self._cfg.gripper.reset_pos_sim[0:3]
        self._desired_eef_pva[1, 0:3] = self._cfg.gripper.reset_pos_sim[4:7]
        self._desired_gripper_angles = np.zeros(2) * self._cfg.gripper.reset_pos_sim[3]
        self._reset_manipulator_when_switch = self._cfg.commander.reset_manipulator_when_switch

        self._body_pva_cmd = self._desired_body_pva.copy()
        self._body_scaled_pva_cmd = self._desired_body_pva.copy()
        self._last_executed_body_pva_cmd = self._desired_body_pva.copy()
        self._body_current_pv = np.zeros(12)
        self._body_pv_buffer = np.zeros(12)

        self._eef_pva_cmd = self._desired_eef_pva.copy()
        self._last_executed_eef_pva_cmd = self._desired_eef_pva.copy()
        self._eef_joint_pos_buffer = np.zeros((2, 3))
        self._eef_joint_pos_scale = np.ones((2, 3)) * self._cfg.commander.eef_joint_pos_scale
        self._eef_rpy_buffer = np.zeros((2, 3))
        self._eef_rpy_scale = np.ones((2, 3)) * self._cfg.commander.eef_rpy_scale

        self._gripper_angles_cmd = self._desired_gripper_angles.copy()
        self._gripper_angles_buffer = np.zeros(2)
        self._gripper_angle_scale = self._cfg.commander.gripper_angle_scale
        self._gripper_angle_range = self._cfg.commander.gripper_angle_range

        self._command_executed = True
        self._eef_task_space_world = True if self._cfg.commander.eef_task_space == 'world' else False

        self._body_pose_scale = self._cfg.commander.body_pv_scale[0:6].copy()
        self._body_vel_scale = self._cfg.commander.body_pv_scale[6:12].copy()
        self._body_pv_scale = self._cfg.commander.body_pv_scale.copy()
        self._body_pv_limit = self._cfg.commander.real_limit.body_pv_limit.copy() if self._cfg.sim.use_real_robot else self._cfg.commander.real_limit.body_pv_limit.copy()
        self._locomotion_height_range = self._cfg.commander.locomotion_height_range
        self._body_p_command_delta = True if self._cfg.commander.body_p_command_type == "delta" else False
        self._body_v_command_delta = True if self._cfg.commander.body_v_command_type == "delta" else False
        #14d body: xyzrpy, eef_r/l: xyzrpy, grippers: 2 angles
        self._joystick_command_sub = rospy.Subscriber(self._cfg.commander.joystick_command_topic, Float32MultiArray, self._update_joystick_command_callback, queue_size=1)

        # for human teleoperation
        self._human_time_at_first_stamp = time.time()
        self._human_time_from_first_stamp = 0.
        self._controller_time_at_first_stamp = time.time()
        self._controller_time_from_first_stamp = 0.
        self._time_for_tracking_human = 0.1
        self._reset_from_human = True
        self._cmd_from_human = None
        self._first_receive_from_human = True
        self.body_xyz_threshold = self._cfg.teleoperation.human_teleoperator.body_xyz_threshold
        self.body_rpy_threshold = self._cfg.teleoperation.human_teleoperator.body_rpy_threshold
        self.eef_xyz_threshold = self._cfg.teleoperation.human_teleoperator.eef_xyz_threshold
        self.eef_rpy_threshold = self._cfg.teleoperation.human_teleoperator.eef_rpy_threshold
        self.gripper_angle_threshold = self._cfg.teleoperation.human_teleoperator.gripper_angle_threshold
        self.body_xyz_scale = self._cfg.teleoperation.human_teleoperator.body_xyz_scale
        self.body_rpy_scale = self._cfg.teleoperation.human_teleoperator.body_rpy_scale
        self.eef_xyz_scale = self._cfg.teleoperation.human_teleoperator.eef_xyz_scale
        self.eef_rpy_scale = self._cfg.teleoperation.human_teleoperator.eef_rpy_scale
        self.gripper_angle_scale = self._cfg.teleoperation.human_teleoperator.gripper_angle_scale
        self.body_xyz_max_step = self._cfg.teleoperation.human_teleoperator.body_xyz_max_step
        self.body_rpy_max_step = self._cfg.teleoperation.human_teleoperator.body_rpy_max_step
        self.eef_xyz_max_step = self._cfg.teleoperation.human_teleoperator.eef_xyz_max_step
        self.eef_rpy_max_step = self._cfg.teleoperation.human_teleoperator.eef_rpy_max_step
        self.gripper_angle_max_step = self._cfg.teleoperation.human_teleoperator.gripper_angle_max_step
        #20d body: xyzrpy, eef_r: xyzrpy, eef_l: xyzrpy, grippers: 2 angles
        self._human_command_sub = rospy.Subscriber(self._cfg.commander.human_command_topic, Float32MultiArray, self._update_human_command_callback, queue_size=1)
        
        # for autonoumous policy rollout command
        self._reset_from_auto_policy = True
        self._cmd_from_auto_policy = None
        self._first_receive_from_auto_policy = True
        self._auto_policy_time_at_first_stamp = time.time()
        self._auto_policy_time_from_first_stamp = 0
        self._time_for_tracking_auto_policy = 0.1
        self._auto_policy_sub = rospy.Subscriber(self._cfg.commander.auto_policy_topic, Float32MultiArray, self._update_auto_policy_command_callback, queue_size=1)

    def reset(self):
        self._is_updating_command = False
        self._first_receive_from_human = True
        self._body_pva_cmd[:] = self._desired_body_pva
        self._body_scaled_pva_cmd[:] = self._desired_body_pva
        self._last_executed_body_pva_cmd[:] = self._desired_body_pva
        self._body_pv_buffer[:] = 0

        if self._reset_manipulator_when_switch:
            self._eef_pva_cmd[:] = self._desired_eef_pva
            self._gripper_angles_cmd[:] = self._desired_gripper_angles
        else:
            manipulator_joint_pos = self._robot.joint_pos[self._env_ids, self._robot._gripper_joint_idx].cpu().numpy()
            self._eef_pva_cmd[0, 0:3] = manipulator_joint_pos[0:3]
            self._eef_pva_cmd[1, 0:3] = manipulator_joint_pos[3:6]
            self._gripper_angles_cmd[:] = self._robot.gripper_angles_np[self._env_ids]
        self._last_executed_eef_pva_cmd[:] = self._eef_pva_cmd[:]
        self._eef_joint_pos_buffer[:] = 0
        self._gripper_angles_buffer[:] = 0
        self._command_executed = True
        self._robot._update_state(reset_estimator=True)


    def use_commander(self):
        self._is_used = True

    def unuse_commander(self):
        self._is_used = False

    def compute_command_for_wbc(self):
        if self._command_executed:
            self._last_executed_body_pva_cmd[:] = self._body_pva_cmd
            self._last_executed_eef_pva_cmd[:] = self._eef_pva_cmd
        else:
            # print('command not executed')
            self._body_pva_cmd[:] = self._last_executed_body_pva_cmd
            self._eef_pva_cmd[:] = self._last_executed_eef_pva_cmd

    def _update_human_command_callback(self, command_msg):
        return

    def _update_auto_policy_command_callback(self, command_msg):
        return

    def get_body_current_pv(self):
        return np.concatenate((self._robot.base_pos_w_np[self._env_ids], self._robot.base_rpy_w2b_np[self._env_ids], self._robot.base_lin_vel_w_np[self._env_ids], self._robot.base_ang_vel_w_np[self._env_ids]))

    def feedback(self, command_executed):
        self._command_executed = command_executed
















