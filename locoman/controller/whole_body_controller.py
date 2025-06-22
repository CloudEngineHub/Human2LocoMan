import numpy as np
# np.set_printoptions(edgeitems=3, infstr='inf', linewidth=200, nanstr='nan', precision=4, suppress=True, threshold=1000, formatter=None)
import pinocchio as pin
import scipy
import scipy.linalg
from robot.base_robot import BaseRobot
from utilities.orientation_utils_numpy import rot_mat_to_rpy, rpy_to_rot_mat, rpy_to_quat, quat_to_rot_mat
from utilities.rotation_interpolation import interpolate_rpy
import quadprog
from fsm.finite_state_machine import FSM_State, FSM_OperatingMode, Manipulate_Mode, fsm_command_to_fsm_state_and_manipulate_mode
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import os
import rospy
from config.config import Cfg
from std_msgs.msg import Float32MultiArray, Int32


def compute_dynamically_consistent_pseudo_inverse(J, M_inv):
    M_inv_J_t = M_inv @ (J.T)  # M^{-1} J^T
    return M_inv_J_t @ (scipy.linalg.inv(J @ M_inv_J_t))  # (M^{-1} J^T) (J M^{-1} J^T)^{-1}

class WholeBodyController:
    def __init__(self, robot: BaseRobot, env_ids):
        self._robot = robot
        self._use_gripper = self._robot._use_gripper
        self._num_joints = self._robot._num_joints
        self._env_ids = env_ids
        mesh_dir = self._robot._cfg.asset.mesh_path
        self._raw_robot_model = pin.buildModelFromUrdf(self._robot._cfg.asset.wbc_urdf_path, pin.JointModelFreeFlyer())
        self._robot_model = pin.RobotWrapper(self._raw_robot_model)
        self._geom_model = pin.buildGeomFromUrdf(pin.buildModelFromUrdf(self._robot._cfg.asset.wbc_urdf_path, pin.JointModelFreeFlyer()), 
                                                 self._robot._cfg.asset.wbc_urdf_path, 
                                                 pin.GeometryType.COLLISION,
                                                 package_dirs=[mesh_dir])
        self._geom_model.addAllCollisionPairs()
        self._torso_frame_id = self._robot_model.index('root_joint')
        self._calf_frame_ids = []
        self._foot_frame_ids = []
        for calf_name in self._robot._cfg.asset.calf_names:
            self._calf_frame_ids.append(self._robot_model.model.getFrameId(calf_name))
        for foot_name in self._robot._cfg.asset.foot_names:
            self._foot_frame_ids.append(self._robot_model.model.getFrameId(foot_name))

        # robot state
        self._q, self._dq = np.zeros(self._num_joints+7), np.zeros(self._num_joints+6)
        self._last_q = np.zeros_like(self._q)
        self._last_dq = np.zeros_like(self._dq)

        # Desired base state
        self._des_torso_rpy = np.zeros(3)
        self._des_torso_pos = np.zeros(3)
        self._des_torso_lin_vel = np.zeros(3)
        self._des_torso_ang_vel = np.zeros(3)
        self._des_torso_lin_acc = np.zeros(3)
        self._des_torso_ang_acc = np.zeros(3)
        self._safe_des_torso_rpy = np.zeros(3)
        self._safe_des_torso_pos = np.zeros(3)

        # Desired foot state
        self._des_footeef_pos = np.zeros(12)
        self._des_footeef_vel = np.zeros(12)
        self._des_footeef_acc = np.zeros(12)
        self._safe_des_footeef_pos = np.zeros(12)

        # contact state and action mode
        self._contact_state = np.ones(4, dtype=bool)
        self._action_mode = 0  # 0: stance, 1: foot-manipulation, 2: eef-manipulation, 3: locomotion, 4:loco-manipulation, 5: bi-manipulation
        self._manipulation_action_modes = [1, 2, 5]
        self._foot_manipulation_action_modes = [1]
        self._eef_manipulation_action_modes = [2, 5]
        self._locomotion_action_modes = [3, 4]

        # safety check
        self._safe_command = False
        # self._foot_clear_command = False
        self._first_command = True
        self._min_joint_pos = self._robot._motors.min_positions_np.copy()
        self._max_joint_pos = self._robot._motors.max_positions_np.copy()
        self._mani_max_delta_q = self._robot._cfg.wbc.mani_max_delta_q
        self._mani_max_dq = self._robot._cfg.wbc.mani_max_dq
        self._mani_max_ddq = self._robot._cfg.wbc.mani_max_ddq
        self._loco_max_delta_q = self._robot._cfg.wbc.loco_max_delta_q
        self._loco_max_dq = self._robot._cfg.wbc.loco_max_dq
        self._loco_max_ddq = self._robot._cfg.wbc.loco_max_ddq
        self._interpolate_decay_ratio = self._robot._cfg.wbc.interpolate_decay_ratio
        self._interpolate_threshold = self._robot._cfg.wbc.interpolate_threshold

        self._mani_max_delta_q_standard = self._robot._cfg.wbc.mani_max_delta_q
        self._mani_max_dq_standard = self._robot._cfg.wbc.mani_max_dq
        self._mani_max_ddq_standard = self._robot._cfg.wbc.mani_max_ddq
        self._interpolate_decay_ratio_standard = self._robot._cfg.wbc.interpolate_decay_ratio
        self._interpolate_threshold_standard = self._robot._cfg.wbc.interpolate_threshold

        self._mani_max_delta_q_safe = self._robot._cfg.wbc.mani_max_delta_q_safe
        self._mani_max_dq_safe = self._robot._cfg.wbc.mani_max_dq_safe
        self._mani_max_ddq_safe = self._robot._cfg.wbc.mani_max_ddq_safe
        self._interpolate_decay_ratio_safe = self._robot._cfg.wbc.interpolate_decay_ratio_safe
        self._interpolate_threshold_safe = self._robot._cfg.wbc.interpolate_threshold_safe

        if self._use_gripper:
            self._mani_max_delta_q_gripper_joint0 = self._robot._cfg.wbc.mani_max_delta_q_gripper_joint0
            self._mani_max_dq_gripper_joint0 = self._robot._cfg.wbc.mani_max_dq_gripper_joint0
            self._mani_max_ddq_gripper_joint0 = self._robot._cfg.wbc.mani_max_ddq_gripper_joint0            
            
            self._mani_max_delta_q_gripper_joint0_standard = self._robot._cfg.wbc.mani_max_delta_q_gripper_joint0
            self._mani_max_dq_gripper_joint0_standard = self._robot._cfg.wbc.mani_max_dq_gripper_joint0
            self._mani_max_ddq_gripper_joint0_standard = self._robot._cfg.wbc.mani_max_ddq_gripper_joint0            
            
            self._mani_max_delta_q_gripper_joint0_safe = self._robot._cfg.wbc.mani_max_delta_q_gripper_joint0_safe
            self._mani_max_dq_gripper_joint0_safe = self._robot._cfg.wbc.mani_max_dq_gripper_joint0_safe
            self._mani_max_ddq_gripper_joint0_safe = self._robot._cfg.wbc.mani_max_ddq_gripper_joint0_safe

        self._interpolate_idx = self._robot._cfg.wbc.interpolate_schedule_step
        self._interpolate_schedule_step = self._robot._cfg.wbc.interpolate_schedule_step

        self._singularity_thresh = self._robot._cfg.wbc.singularity_thresh
        self._mani_index = 0.0
        
        self._ground_collision_gripper_thresh = self._robot._cfg.wbc.ground_collision_gripper_thresh
        self._ground_collision_foot_thresh = self._robot._cfg.wbc.ground_collision_foot_thresh
        self._ground_collision_knee_thresh = self._robot._cfg.wbc.ground_collision_knee_thresh
        
        # Gains
        use_real_robot = self._robot._cfg.sim.use_real_robot
        self._use_real_robot = use_real_robot
        self._base_position_kp_loco = self._robot._cfg.wbc.real.base_position_kp_loco if use_real_robot else self._robot._cfg.wbc.sim.base_position_kp_loco
        self._base_position_kd_loco = self._robot._cfg.wbc.real.base_position_kd_loco if use_real_robot else self._robot._cfg.wbc.sim.base_position_kd_loco
        self._base_orientation_kp_loco = self._robot._cfg.wbc.real.base_orientation_kp_loco if use_real_robot else self._robot._cfg.wbc.sim.base_orientation_kp_loco
        self._base_orientation_kd_loco = self._robot._cfg.wbc.real.base_orientation_kd_loco if use_real_robot else self._robot._cfg.wbc.sim.base_orientation_kd_loco

        self._base_position_kp_mani = self._robot._cfg.wbc.real.base_position_kp_mani if use_real_robot else self._robot._cfg.wbc.sim.base_position_kp_mani
        self._base_position_kd_mani = self._robot._cfg.wbc.real.base_position_kd_mani if use_real_robot else self._robot._cfg.wbc.sim.base_position_kd_mani
        self._base_orientation_kp_mani = self._robot._cfg.wbc.real.base_orientation_kp_mani if use_real_robot else self._robot._cfg.wbc.sim.base_orientation_kp_mani
        self._base_orientation_kd_mani = self._robot._cfg.wbc.real.base_orientation_kd_mani if use_real_robot else self._robot._cfg.wbc.sim.base_orientation_kd_mani
        
        self._footeef_position_kp = self._robot._cfg.wbc.real.footeef_position_kp if use_real_robot else self._robot._cfg.wbc.sim.footeef_position_kp
        self._footeef_position_kd = self._robot._cfg.wbc.real.footeef_position_kd if use_real_robot else self._robot._cfg.wbc.sim.footeef_position_kd

        # for grippers
        if self._use_gripper:
            self._eef_frame_ids = []
            for eef_name in self._robot._cfg.asset.eef_names:
                self._eef_frame_ids.append(self._robot_model.model.getFrameId(eef_name))
            self._des_eef_pos = np.zeros((len(self._eef_frame_ids), 3))
            self._des_eef_vel = np.zeros((len(self._eef_frame_ids), 3))
            self._des_eef_acc = np.zeros((len(self._eef_frame_ids), 3))
            self._safe_des_eef_pos = np.zeros((len(self._eef_frame_ids), 3))
            self._des_eef_frame = {'world':0, 'foot':1, 'joint':2}[self._robot._cfg.manipulation.non_manipulate_eef_reference_frame]
            if self._des_eef_frame == 2:
                from manipulator.gripper_kinematics import GripperKinematics
                self._gripper_kinematics = GripperKinematics(self._robot._cfg)
            self._eef_orientation_kp = self._robot._cfg.wbc.real.eef_orientation_kp if use_real_robot else self._robot._cfg.wbc.sim.eef_orientation_kp
            self._eef_orientation_kd = self._robot._cfg.wbc.real.eef_orientation_kd if use_real_robot else self._robot._cfg.wbc.sim.eef_orientation_kd
            self._tasks = ['torso_pos', 'torso_rpy', 'swing_foot_eef_pos', 'swing_eef_rpy', 'stance_eef_pos']
            self._task_dims = [3, 3, 12, 3, 3]
            self._gripper_joint_idx = self._robot._cfg.asset.gripper_joint_idx
        else:
            self._tasks = ['torso_pos', 'torso_rpy', 'swing_foot_pos']
            self._task_dims = [3, 3, 12]
        self._task_dims = [3] + self._task_dims

        # buffers for WBC
        self._tasks_num = len(self._tasks)

        self._e = [np.zeros(self._task_dims[i]) for i in range(self._tasks_num+1)]
        self._dx = [np.zeros(self._task_dims[i]) for i in range(self._tasks_num+1)]
        self._ddx = [np.zeros(self._task_dims[i]) for i in range(self._tasks_num+1)]
        self._delta_q, self._dq_cmd, self._ddq_cmd, self._torque = np.zeros(self._num_joints+6), np.zeros(self._num_joints+6), np.zeros(self._num_joints+6), np.zeros(self._num_joints+6)
        self._des_q, self._last_des_q, self._last_dq_cmd, self._last_torque = np.zeros(self._num_joints), np.zeros(self._num_joints), np.zeros(self._num_joints), np.zeros(self._num_joints)
        self._last_ddq_cmd = np.zeros_like(self._ddq_cmd)

        self._N = [np.eye(self._num_joints+6) for _ in range(self._tasks_num+1)]
        self._N_i_im1 = [np.eye(self._num_joints+6) for _ in range(self._tasks_num+1)]
        self._N_dyn = [np.eye(self._num_joints+6) for _ in range(self._tasks_num+1)]
        self._N_i_im1_dyn = [np.eye(self._num_joints+6) for _ in range(self._tasks_num+1)]

        self._J = [np.zeros((self._task_dims[i], self._num_joints+6)) for i in range(self._tasks_num+1)]
        self._J_i_im1 = [np.zeros((self._task_dims[i], self._num_joints+6)) for i in range(self._tasks_num+1)]
        self._J_pre = [np.zeros((self._task_dims[i], self._num_joints+6)) for i in range(self._tasks_num+1)]
        self._J_pre_inv = [np.zeros((self._num_joints+6, self._task_dims[i])) for i in range(self._tasks_num+1)]
        self._J_i_im1_dyn = [np.zeros((self._task_dims[i], self._num_joints+6)) for i in range(self._tasks_num+1)]
        self._J_pre_dyn = [np.zeros((self._task_dims[i], self._num_joints+6)) for i in range(self._tasks_num+1)]
        self._J_pre_dyn_inv = [np.zeros((self._num_joints+6, self._task_dims[i])) for i in range(self._tasks_num+1)]
        self._dJdq = [np.zeros(self._task_dims[i]) for i in range(self._tasks_num+1)]
        self._dsl_mat = [np.zeros((self._num_joints+6, self._task_dims[i])) for i in range(self._tasks_num+1)]
        
        # only consider single-gripper manipulation for now
        # self.eef_jacobian_pos = np.zeros((3, 6))
        # self.eef_jacobian_ori = np.zeros((3, 6))
        self.eef_jacobian = np.zeros((6, 6))
        self.singularity_value = 0
        self._last_executed_command = {}
        self._safety_flag = "safe command"
        self._start_unsafe_command = False
        self._start_keep_current_state = False
        self._on_joint_limit_or_collision = False

        self._desired_gripper_angles = np.zeros(2) * self._robot._cfg.gripper.reset_pos_sim[3]
        # to align with the eef manipulate commander for now
        self._desired_eef_pva = np.zeros((2, 9))
        self._desired_eef_pva[0, 0:3] = self._robot._cfg.gripper.reset_pos_sim[0:3]
        self._desired_eef_pva[1, 0:3] = self._robot._cfg.gripper.reset_pos_sim[4:7]

        self._self_collision_count = 0
        self._check_self_collision = False

        self.fsm_publisher = rospy.Publisher(Cfg.fsm_switcher.fsm_state_topic, Int32, queue_size = 1)

    def update_robot_model(self):
        # update the pinocchio model
        self._last_q = self._q
        self._last_dq = self._dq
        self._q = np.concatenate((self._robot.base_pos_w_np[self._env_ids],
                               self._robot.base_quat_w2b_np[self._env_ids],
                               self._robot.joint_pos_np[self._env_ids]))

        # the convention of pinocchio is to use dq in the local frame !
        self._dq = np.concatenate((self._robot.base_lin_vel_b_np[self._env_ids],
                                self._robot.base_ang_vel_b_np[self._env_ids],
                                self._robot.joint_vel_np[self._env_ids]))
        self._robot_model.forwardKinematics(self._q, self._dq, np.zeros_like(self._dq))
        self._robot_model.computeJointJacobians(self._q)

    # def update(self, action_mode, des_body_pva, contact_state, des_footeef_pva_w, desired_eef_pva):
    #     self.update_robot_model()
    #     # set desired state
    #     self._action_mode = action_mode
    #     self.set_des_torso_pva(des_body_pva.copy())
    #     self.set_contact_state(contact_state.copy())
    #     self.set_desired_footeef_pva(des_footeef_pva_w.copy())
    #     if self._use_gripper:
    #         self.set_desired_eef_pva(desired_eef_pva.copy())
    #     # compute actions
    #     return self.compute_actions()

    def set_desired_static_states(self, first_unsafe=False):
        # will not change action mode and contact state
        # set torso state
        if first_unsafe:
            self._safe_des_torso_pos = copy.deepcopy(self._robot.base_pos_w_np[self._env_ids])
            self._safe_des_torso_rpy = copy.deepcopy(self._robot.base_rpy_w2b_np[self._env_ids])
        self._des_torso_pos = self._safe_des_torso_pos
        self._des_torso_rpy = self._safe_des_torso_rpy
        self._des_torso_lin_vel = np.zeros(3)
        self._des_torso_ang_vel = np.zeros(3)
        self._des_torso_lin_acc = np.zeros(3)
        self._des_torso_ang_acc = np.zeros(3)

        # set foot eef state
        # set end effector position
        # self._des_footeef_pos: [12]
        if first_unsafe:
            swing_indices = np.nonzero(np.logical_not(self._contact_state))[0]
            if len(swing_indices) > 0:
                swing_dofs = []
                for index in swing_indices:
                    swing_dofs.extend([index * 3, index * 3 + 1, index * 3 + 2])
                swing_dofs = np.array(swing_dofs)
                cur_footeef_pos = self._robot.eef_pos_w_np[self._env_ids][swing_indices].flatten() if (self._use_gripper and self._action_mode in self._eef_manipulation_action_modes) else self._robot.foot_pos_w_np[self._env_ids][swing_indices].flatten() 
                self._safe_des_footeef_pos[swing_dofs] = copy.deepcopy(cur_footeef_pos)

        self._des_footeef_pos = self._safe_des_footeef_pos
        self._des_footeef_vel = np.zeros(12)
        self._des_footeef_acc = np.zeros(12)

        if self._use_gripper:
        # set end effector orientation
        # self._des_eef_pos: [2, 3]
            if first_unsafe:
                for i in range(len(self._eef_frame_ids)):
                    self._safe_des_eef_pos[i] = copy.deepcopy(self._robot.eef_rpy_w_np[self._env_ids][i])
            self._des_eef_pos = self._safe_des_eef_pos
            self._des_eef_vel = np.zeros((len(self._eef_frame_ids), 3))
            self._des_eef_acc = np.zeros((len(self._eef_frame_ids), 3))

    def set_interpolated_desired_states_from_two_commands(self, last_command, command, alpha):
        # smaller alpha represents smaller command change
        self._action_mode = command["action_mode"]
        self.set_contact_state(command["contact_state"].copy())

        # interpolate and set desired torso 6d pose
        last_torso_pos = last_command['body_pva'][:3]
        last_torso_rpy = last_command['body_pva'][3:6]
        target_torso_pos = command['body_pva'][:3]
        target_torso_rpy = command['body_pva'][3:6]
        new_desired_torso_pos = last_torso_pos * (1 - alpha) + target_torso_pos * alpha
        new_desired_torso_rpy = interpolate_rpy(last_torso_rpy, target_torso_rpy, alpha)
        self._des_torso_pos = new_desired_torso_pos
        self._des_torso_rpy = new_desired_torso_rpy
        self._des_torso_lin_vel = np.zeros(3)
        self._des_torso_ang_vel = np.zeros(3)
        self._des_torso_lin_acc = np.zeros(3)
        self._des_torso_ang_acc = np.zeros(3)

        # interpolate and set desired end effector position
        swing_indices = np.nonzero(np.logical_not(self._contact_state))[0]
        if len(swing_indices) > 0:
            swing_dofs = []
            for index in swing_indices:
                swing_dofs.extend([index * 3, index * 3 + 1, index * 3 + 2])
            swing_dofs = np.array(swing_dofs)
            last_footeef_pos = last_command['footeef_pva'][0, :][swing_dofs]
            target_footeef_pos = command['footeef_pva'][0, :][swing_dofs]
            new_desired_footeef_pos = last_footeef_pos * (1 - alpha) + target_footeef_pos * alpha
            self._des_footeef_pos[swing_dofs] = new_desired_footeef_pos
            self._des_footeef_vel = np.zeros(12)
            self._des_footeef_acc = np.zeros(12)

        # interpolate and set desired end effector orientation
            if self._use_gripper:
                if self._action_mode in self._eef_manipulation_action_modes:
                    last_eef_pos = last_command['eef_pva'][:, :3]
                    target_eef_pos = command['eef_pva'][:, :3]
                    if self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_EEF:
                        manipulate_leg_idx = [0]
                    elif self._robot._cur_manipulate_mode == Manipulate_Mode.LEFT_EEF:
                        manipulate_leg_idx = [1]
                    else:
                        # bimanual
                        manipulate_leg_idx = [0, 1]
                    for idx in manipulate_leg_idx:
                        new_desired_eef_pos = interpolate_rpy(last_eef_pos[idx], target_eef_pos[idx], alpha)
                        self._des_eef_pos[idx] = new_desired_eef_pos
                    self._des_eef_vel = np.zeros((len(self._eef_frame_ids), 3))
                    self._des_eef_acc = np.zeros((len(self._eef_frame_ids), 3))

    def set_interpolated_desired_states_from_command(self, command, alpha):
        # smaller alpha represents smaller command change
        self._action_mode = command["action_mode"]
        self.set_contact_state(command["contact_state"].copy())

        # interpolate and set desired torso 6d pose
        cur_torso_pos = self._robot.base_pos_w_np[self._env_ids]
        cur_torso_rpy = self._robot.base_rpy_w2b_np[self._env_ids]
        target_torso_pos = command['body_pva'][:3]
        target_torso_rpy = command['body_pva'][3:6]
        new_desired_torso_pos = cur_torso_pos * (1 - alpha) + target_torso_pos * alpha
        new_desired_torso_rpy = interpolate_rpy(cur_torso_rpy, target_torso_rpy, alpha)
        self._des_torso_pos = new_desired_torso_pos
        self._des_torso_rpy = new_desired_torso_rpy
        self._des_torso_lin_vel = np.zeros(3)
        self._des_torso_ang_vel = np.zeros(3)
        self._des_torso_lin_acc = np.zeros(3)
        self._des_torso_ang_acc = np.zeros(3)
        # print('cur_torso_pos', cur_torso_pos)
        # print('target_torso_pos', target_torso_pos)
        # print('new_desired_torso_pos', self._des_torso_pos)
        # print('cur_torso_rpy', cur_torso_rpy)
        # print('target_torso_rpy', target_torso_rpy)
        # print('new_desired_torso_rpy', self._des_torso_rpy)

        # interpolate and set desired end effector position
        swing_indices = np.nonzero(np.logical_not(self._contact_state))[0]
        if len(swing_indices) > 0:
            swing_dofs = []
            for index in swing_indices:
                swing_dofs.extend([index * 3, index * 3 + 1, index * 3 + 2])
            swing_dofs = np.array(swing_dofs)
            cur_footeef_pos = self._robot.eef_pos_w_np[self._env_ids][swing_indices].flatten() if (self._use_gripper and self._action_mode in self._eef_manipulation_action_modes) else self._robot.foot_pos_w_np[self._env_ids][swing_indices].flatten() 
            target_footeef_pos = command['footeef_pva'][0, :][swing_dofs]
            new_desired_footeef_pos = cur_footeef_pos * (1 - alpha) + target_footeef_pos * alpha
            self._des_footeef_pos[swing_dofs] = new_desired_footeef_pos
            self._des_footeef_vel = np.zeros(12)
            self._des_footeef_acc = np.zeros(12)
            # print('cur_footeef_pos', cur_footeef_pos)
            # print('target_footeef_pos', target_footeef_pos)
            # print('new_desired_footeef_pos', self._des_footeef_pos[swing_dofs])
        
        # interpolate and set desired end effector orientation
            if self._use_gripper:
                if self._action_mode in self._eef_manipulation_action_modes:
                    cur_eef_pos = self._robot.eef_rpy_w_np[self._env_ids]
                    target_eef_pos = command['eef_pva'][:, :3]
                    if self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_EEF:
                        manipulate_leg_idx = [0]
                    elif self._robot._cur_manipulate_mode == Manipulate_Mode.LEFT_EEF:
                        manipulate_leg_idx = [1]
                    else:
                        # bimanual
                        manipulate_leg_idx = [0, 1]
                    for idx in manipulate_leg_idx:
                        new_desired_eef_pos = interpolate_rpy(cur_eef_pos[idx], target_eef_pos[idx], alpha)
                        self._des_eef_pos[idx] = new_desired_eef_pos
                    self._des_eef_vel = np.zeros((len(self._eef_frame_ids), 3))
                    self._des_eef_acc = np.zeros((len(self._eef_frame_ids), 3))
                    # print('cur_eef_pos', cur_eef_pos[manipulate_leg_idx])
                    # print('target_eef_pos', target_eef_pos[manipulate_leg_idx])
                    # print('new_desired_eef_pos', self._des_eef_pos[manipulate_leg_idx])

    def get_cur_executed_command(self):
        body_pva_cmd = np.zeros(18)
        body_pva_cmd[:3] = self._des_torso_pos
        body_pva_cmd[3:6] = self._des_torso_rpy
        body_pva_cmd[6:9] = self._des_torso_lin_vel
        body_pva_cmd[9:12] = self._des_torso_ang_vel
        body_pva_cmd[12:15] = self._des_torso_lin_acc
        body_pva_cmd[15:18] = self._des_torso_ang_acc

        footeef_pva_cmd = np.zeros((3, 12))
        footeef_pva_cmd[0, :] = self._des_footeef_pos
        footeef_pva_cmd[1, :] = self._des_footeef_vel
        footeef_pva_cmd[2, :] = self._des_footeef_acc

        # to align with the eef manipulate commander for now
        eef_pva_cmd = self._desired_eef_pva

        if self._use_gripper:
            if self._action_mode in self._eef_manipulation_action_modes:
                if self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_EEF:
                    manipulate_leg_idx = [0]
                elif self._robot._cur_manipulate_mode == Manipulate_Mode.LEFT_EEF:
                    manipulate_leg_idx = [1]
                else:
                    manipulate_leg_idx = [0, 1]
                for idx in manipulate_leg_idx:
                    eef_pva_cmd[idx, :3] = self._des_eef_pos[idx, :]
            
                eef_pva_cmd[:, 3:6] = self._des_eef_vel
                eef_pva_cmd[:, 6:9] = self._des_eef_acc

        return {"action_mode": self._action_mode,
                "contact_state": self._contact_state,
                "body_pva": body_pva_cmd,
                "footeef_pva": footeef_pva_cmd,
                "eef_pva": eef_pva_cmd,
                "gripper_angles": self._desired_gripper_angles
                }
    
    def set_desired_states_from_command(self, command):
        self._action_mode = command["action_mode"]
        self.set_des_torso_pva(command["body_pva"].copy())
        self.set_contact_state(command["contact_state"].copy())
        self.set_desired_footeef_pva(command["footeef_pva"].copy())
        if self._use_gripper:
            self.set_desired_eef_pva(command["eef_pva"].copy())
            self._desired_gripper_angles = command["gripper_angles"]


    # def set_des_torso_pva(self, pva):
    #     self._des_torso_pos = pva[:3].copy()
    #     self._des_torso_rpy = pva[3:6].copy()
    #     self._des_torso_lin_vel = pva[6:9].copy()
    #     self._des_torso_ang_vel = pva[9:12].copy()
    #     self._des_torso_lin_acc = pva[12:15].copy()
    #     self._des_torso_ang_acc = pva[15:18].copy()

    # def set_desired_footeef_pva(self, des_footeef_pva):
    #     self._des_footeef_pos = des_footeef_pva[0, :].copy()
    #     self._des_footeef_vel = des_footeef_pva[1, :].copy()
    #     self._des_footeef_acc = des_footeef_pva[2, :].copy()

    # def set_desired_eef_pva(self, des_eef_pva):
    #     self._des_eef_pos = des_eef_pva[:, :3].copy()
    #     self._des_eef_vel = des_eef_pva[:, 3:6].copy()
    #     self._des_eef_acc = des_eef_pva[:, 6:9].copy()

    #     transform_frame_eef_idx = []
    #     if self._des_eef_frame == 2:
    #         for i in range(len(self._eef_frame_ids)):
    #             if self._contact_state[i] or self._action_mode not in self._eef_manipulation_action_modes:
    #                 if self._robot._nex_fsm_state==FSM_State.LOCOMANIPULATION and self._robot._fsm_operation_mode==FSM_OperatingMode.TRANSITION and i==self._robot._cfg.loco_manipulation.manipulate_leg_idx:
    #                     continue
    #                 elif self._robot._cur_fsm_state==FSM_State.LOCOMANIPULATION and i==self._robot._cfg.loco_manipulation.manipulate_leg_idx and not self._robot._cfg.loco_manipulation.locomotion_only:
    #                     continue
    #                 else:
    #                     transform_frame_eef_idx.append(i)
    #     if len(transform_frame_eef_idx) > 0:
    #         des_eef_rot_foot = self._gripper_kinematics.forward_kinematics(des_eef_pva[transform_frame_eef_idx, 0:3], transform_frame_eef_idx)
    #         for i, eef_idx in enumerate(transform_frame_eef_idx):
    #             foot_rot_world = self._robot_model.framePlacement(None, self._foot_frame_ids[eef_idx], update_kinematics=False).rotation
    #             self._des_eef_pos[eef_idx] = rot_mat_to_rpy(foot_rot_world.dot(des_eef_rot_foot[i]))
    #             self._des_eef_vel[eef_idx] = np.zeros(3)
    #             self._des_eef_acc[eef_idx] = np.zeros(3)

    # def set_contact_state(self, contact_state):
    #     self._contact_state = contact_state

    def _apply_low_pass_filter_on_des_q_for_real_robot_gripper(self, alpha=0.2):
        if self._use_real_robot and self._use_gripper:
            self._des_q[self._gripper_joint_idx] = self._des_q[self._gripper_joint_idx] * alpha + self._last_des_q[self._gripper_joint_idx] * (1 - alpha)

    def _apply_low_pass_filter_and_clip_on_ddq(self, alpha=0.2, joint_clip_range=[-15, 15], torso_clip_range=[-10, 10]):
        self._ddq_cmd[:] = self._ddq_cmd[:] * alpha + self._last_ddq_cmd[:] * (1 - alpha)
        self._ddq_cmd[:-self._num_joints] = np.clip(self._ddq_cmd[:-self._num_joints], torso_clip_range[0], torso_clip_range[1])
        self._ddq_cmd[-self._num_joints:] = np.clip(self._ddq_cmd[-self._num_joints:], joint_clip_range[0], joint_clip_range[1]) 
        self._last_ddq_cmd[:] = self._ddq_cmd[:]

    def _apply_low_pass_filter(self, on_derivative=False, on_torque=True, alpha=0.2):
        # self._robot._update_state()
        if on_derivative:
            derivative_error = self._dq_cmd[-self._num_joints:] - self._dq[-self._num_joints:]
            last_derivative_error = self._last_dq_cmd - self._last_dq[-self._num_joints:]
            filtered_detivative_error = derivative_error * alpha + last_derivative_error * (1 - alpha)
            filtered_dq_cmd = filtered_detivative_error + self._dq[-self._num_joints:]
            self._dq_cmd[-self._num_joints:] = filtered_dq_cmd
        if on_torque:
            self._torque[-self._num_joints:] = self._torque[-self._num_joints:] * alpha + self._last_torque * (1 - alpha)

    def _interpolate_scheduler(self, reset=False, zero=False, beta=4):
        if self._action_mode in self._manipulation_action_modes:
            if reset:
                if zero:
                    self._interpolate_idx = 0
                else:
                    self._interpolate_idx = self._interpolate_schedule_step
            ratio = self._interpolate_idx / self._interpolate_schedule_step
            # ratio = np.exp(-beta * (1 - ratio))

            self._mani_max_delta_q = self._mani_max_delta_q_standard * ratio + self._mani_max_delta_q_safe * (1 - ratio)
            self._mani_max_dq = self._mani_max_dq_standard * ratio + self._mani_max_dq_safe * (1 - ratio)
            self._mani_max_ddq = self._mani_max_ddq_standard * ratio + self._mani_max_ddq_safe * (1 - ratio)

            if self._use_gripper:
                self._mani_max_delta_q_gripper_joint0 = self._mani_max_delta_q_gripper_joint0_standard * ratio + self._mani_max_delta_q_gripper_joint0_safe * (1 - ratio)
                self._mani_max_dq_gripper_joint0 = self._mani_max_dq_gripper_joint0_standard * ratio + self._mani_max_dq_gripper_joint0_safe * (1 - ratio)
                self._mani_max_ddq_gripper_joint0 = self._mani_max_ddq_gripper_joint0_standard * ratio + self._mani_max_ddq_gripper_joint0_safe * (1 - ratio)

            self._interpolate_decay_ratio = self._interpolate_decay_ratio_standard * ratio + self._interpolate_decay_ratio_safe * (1 - ratio)
            self._interpolate_threshold = self._interpolate_threshold_standard * ratio + self._interpolate_threshold_safe * (1 - ratio)

            if self._interpolate_idx < self._interpolate_schedule_step:
                self._interpolate_idx += 1

    def compute_action_from_command(self, command):
        if self._first_command:
            self._first_command = False
        # set desired state
        self.set_desired_states_from_command(command)
        # first compute desired actions from the command and check if safe
        self.pre_compute_actions()
        # print('self._last_executed_command', self._last_executed_command)
        # if self._last_executed_command != {}:
        #     print('last action mode for wbc', self._last_executed_command['action_mode'])
        ## safe command ##
        if self._safe_command:
            # self._start_unsafe_command = False
            # self._start_keep_current_state = False
            self._solve_joint_torques()
            # print('torques', self._torque[-self._num_joints:])
            self._last_des_q[:], self._last_dq_cmd[:], self._last_torque[:] = self._des_q[-self._num_joints:], self._dq_cmd[-self._num_joints:], self._torque[-self._num_joints:]
            self.log_wbc_info()
            self._last_executed_command = copy.deepcopy(command)
            return True, self._des_q, self._dq_cmd[-self._num_joints:], self._torque[-self._num_joints:]
        # only consider manipulation for now
        else:
            ## if unsafe due to joint limits or collision or close to singularity ##
            if self._safety_flag == 'ground collision' or self._safety_flag == 'self collision' or self._safety_flag == 'joint limits' or self._safety_flag == 'singularity':
                # want stably keep at the boundary, and keep the right ddq cmd if possible (prevent oscillation)
                self.set_desired_states_from_command(self._last_executed_command)
                # if not self._on_joint_limit_or_collision:
                #     self.set_interpolated_desired_states_from_command(command, 0.0)
                #     self._on_joint_limit_or_collision = True
                # else:
                #     self.set_desired_states_from_command(self._last_executed_command)
                self._compute_task_hierarchy()
                self._interpolate_scheduler(reset=True, zero=False)
                self._on_joint_limit_or_collision = True
                # print('interpolate idx', self._interpolate_idx)
                # print('interpolate thresh', self._interpolate_threshold)
                # print('mani max delta q', self._mani_max_delta_q)
                # print('mani max ddq', self._mani_max_ddq)
                # joint limits or collision or singularity still might be detected
                # apply last should be safe enough if not fast or stop fast
                if not (self._beyond_max_delta_q() or self._beyond_max_dq() or self._beyond_max_ddq()):
                    # self._start_unsafe_command = False 
                    # self._start_keep_current_state = False
                    self._solve_joint_torques()
                    print('apply last command')
                    self._last_des_q[:], self._last_dq_cmd[:], self._last_torque[:] = self._des_q[-self._num_joints:], self._dq_cmd[-self._num_joints:], self._torque[-self._num_joints:]
                    self.log_wbc_info()
                    # self._last_executed_command = self.get_cur_executed_command()
                    return False, self._des_q, self._dq_cmd[-self._num_joints:], self._torque[-self._num_joints:]

            ## if move too fast or the acceleration is large (when reach a joint limit and needs to stop urgently), try to interpolate between the last command and the current command ##
            if self._on_joint_limit_or_collision:
                self._interpolate_scheduler(reset=True, zero=True)
                self._on_joint_limit_or_collision = False
            else:
                self._interpolate_scheduler(reset=False)
            decay_ratio = self._interpolate_decay_ratio
            # print('interpolate idx', self._interpolate_idx)
            # print('interpolate thresh', self._interpolate_threshold)
            # print('mani max delta q', self._mani_max_delta_q)
            # print('mani max ddq', self._mani_max_ddq)
            threshold = self._interpolate_threshold
            alpha = decay_ratio
            while alpha > threshold:
                # interpolations start from last command
                self.set_interpolated_desired_states_from_two_commands(self._last_executed_command, command, alpha)
                self._compute_task_hierarchy()
                self._check_safety()
                if self._safe_command:
                    break
                alpha = alpha * decay_ratio
            if self._safe_command:         
                # self._start_unsafe_command = False  
                # self._start_keep_current_state = False
                self._solve_joint_torques()
                # if self._use_gripper and self._use_real_robot:
                #     if self._interpolate_idx > 0 and self._interpolate_idx < self._interpolate_schedule_step:
                #         self._apply_low_pass_filter_on_des_q_for_real_robot_gripper(alpha=0.1)
                print('interpolate commands')
                self._last_des_q[:], self._last_dq_cmd[:], self._last_torque[:] = self._des_q[-self._num_joints:], self._dq_cmd[-self._num_joints:], self._torque[-self._num_joints:]
                self.log_wbc_info()
                self._last_executed_command = self.get_cur_executed_command()
                return False, self._des_q, self._dq_cmd[-self._num_joints:], self._torque[-self._num_joints:]
            else:
                # want stably keep pose, and keep the right ddq cmd if possible (prevent oscillation)
                self.set_desired_states_from_command(self._last_executed_command)
                self._compute_task_hierarchy()
     
                # self._start_unsafe_command = False  
                # self._start_keep_current_state = False
                self._check_safety()
                self._apply_low_pass_filter_and_clip_on_ddq(joint_clip_range=[-self._mani_max_ddq * 2, self._mani_max_ddq * 2], alpha=1.0)
                self._solve_joint_torques()
                print('try to keep the state from last command')
                print('delta q', self._delta_q[-self._num_joints:])
                print('delta q max', np.max(abs(self._delta_q[-self._num_joints:])), 'idx', np.argmax(abs(self._delta_q[-self._num_joints:])))
                print('ddq', self._ddq_cmd[-self._num_joints:])
                print('ddq max', np.max(abs(self._ddq_cmd[-self._num_joints:])), 'idx', np.argmax(abs(self._ddq_cmd[-self._num_joints:])))
                print('mani index', self._mani_index)
                self._last_des_q[:], self._last_dq_cmd[:], self._last_torque[:] = self._des_q[-self._num_joints:], self._dq_cmd[-self._num_joints:], self._torque[-self._num_joints:]
                self.log_wbc_info()
                # self._last_executed_command = self.get_cur_executed_command()
                return False, self._des_q, self._dq_cmd[-self._num_joints:], self._torque[-self._num_joints:]
    
    def pre_compute_actions(self):
        self._compute_task_hierarchy()
        if self._robot._fsm_operation_mode == FSM_OperatingMode.NORMAL:
            self._if_check_self_collisions()
            self._safety_flag = self._check_safety()
        else:
            self._safe_command = True

    def _compute_stance_foot_position_jacobian(self):
        foot_jacobians = []
        for idx, foot_frame_id in enumerate(self._foot_frame_ids):
            if self._contact_state[idx]:
                foot_jacobians.append(
                    self._robot_model.getFrameJacobian(foot_frame_id, rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3])
        return np.concatenate(foot_jacobians, axis=0)

    def _compute_swing_foot_position_jacobian(self):
        foot_jacobians = []
        for idx, foot_frame_id in enumerate(self._foot_frame_ids):
            # LOCAL_WORLD_ALIGNED: while the origin of this frame moves with the moving part, its orientation remains fixed in alignment with the global reference frame.
            if not self._contact_state[idx]:
                foot_jacobians.append(
                    self._robot_model.getFrameJacobian(foot_frame_id, rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3])
        foot_jacobians = np.concatenate(foot_jacobians, axis=0)
        foot_jacobians[:, :6] = 0  # Not affecting base!
        return foot_jacobians

    def _compute_eef_jacobian(self, eef_idx, type='position'):
        idx_begin = 0 if type == 'position' else 3
        idx_end = 3 if type == 'position' else 6

        legged_arm_jacobians = []
        for idx in eef_idx:
            legged_arm_jacobians.append(
                    self._robot_model.getFrameJacobian(self._eef_frame_ids[idx], rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[idx_begin:idx_end])
        legged_arm_jacobians = np.concatenate(legged_arm_jacobians, axis=0)
        legged_arm_jacobians[:, :6] = 0  # Not affecting base!

        if self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_EEF:
            if eef_idx[0] == 0:
                if type == 'position':
                    self.eef_jacobian[:3, :] = legged_arm_jacobians[:, 6:12]
                else:
                    self.eef_jacobian[3:, :] = legged_arm_jacobians[:, 6:12]
        elif self._robot._cur_manipulate_mode == Manipulate_Mode.LEFT_EEF:
                if type == 'position':
                    self.eef_jacobian[:3, :] = legged_arm_jacobians[:, 12:18]
                else:
                    self.eef_jacobian[3:, :] = legged_arm_jacobians[:, 12:18]
        else:
            pass

        # # print('loco-manipulator jacobian', legged_arm_jacobians)

        return legged_arm_jacobians
    
    def _check_singularity(self):
        if self._action_mode == 2 and self._robot._fsm_operation_mode == FSM_OperatingMode.NORMAL:
            des_q = np.concatenate((self._des_torso_pos,
                                rpy_to_quat(self._des_torso_rpy).toNumpy().reshape(4,),
                                self._q[-self._num_joints:] + self._delta_q[-self._num_joints:]))
            self._robot_model.computeJointJacobians(des_q)

            wbc_steps = self._tasks_num
            swing_indices = np.nonzero(np.logical_not(self._contact_state))[0]
            if len(swing_indices) > 0:
                self._compute_eef_jacobian(eef_idx=swing_indices, type='position') if (self._use_gripper and self._action_mode in self._eef_manipulation_action_modes) else self._compute_swing_foot_position_jacobian()
            else:
                wbc_steps = self._tasks_num - 1

            if self._use_gripper:
            # eef orientation tasks
                for i in range(wbc_steps-self._tasks_num, wbc_steps-self._tasks_num+2):
                    self._J[i+4] = self._compute_eef_jacobian(eef_idx=[i], type='orientation')

            self._robot_model.computeJointJacobians(self._q)

            # if self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_FOOT or self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_EEF:
            #     j = self._robot._compute_foot_jacobian(self._des_q, True)[0][0]
            # elif self._robot._cur_manipulate_mode == Manipulate_Mode.LEFT_FOOT or self._robot._cur_manipulate_mode == Manipulate_Mode.LEFT_EEF:
            #     j = self._robot._compute_foot_jacobian(self._des_q, True)[0][1]
            # else:
            #     # bimanual not considered for now
            #     j = self._robot._compute_foot_jacobian(self._des_q, True)[0][:2]
            ##############################################
            j = self.eef_jacobian
            ##############################################
            # det_j = np.linalg.det(j)
            self._mani_index = np.sqrt(np.linalg.det(np.dot(j, j.T)))
            # print('self._mani_index', self._mani_index)
            if self._mani_index <= self._singularity_thresh:
                print('close to singularity')
                return True
            # cond_j = np.linalg.cond(j)
            # _, s, _ = np.linalg.svd(j)
            
            # print('abs(det_j):', np.abs(det_j))
            # print('mani index', mani_index)
            # print('1 / cond_j:', 1 / cond_j)
            # print('s:', s)
            # if np.abs(det_j) < 0.0015:
            #     print(f'singularity is detected: jacobian determinant {np.abs(det_j)} < 0.0015')
            #     return True
            # if any(s < 0.08):
            #     print(f'singularity is detected: smallest singular value {np.minimum(s)} < 0.08')
            #     return True
            # elif np.abs(det_j) < 0.009:
            #     print(f'singularity is detected: jacobian determinant {np.abs(det_j)} < 0.009')
            #     return True
            # elif 1 / cond_j < 0.16:
            #     print(f'singularity is detected: 1 / condition number {1/cond_j} < 0.16')
            #     return True
        
        return False

    def _compute_orientation_error_from_two_world_rpy(self, des_rpy, actual_rpy):
        des_rot = rpy_to_rot_mat(des_rpy)
        actual_rot = rpy_to_rot_mat(actual_rpy)
        return rot_mat_to_rpy(des_rot.dot(actual_rot.T))

    def _compute_orientation_error_from_world_rpy_and_rot_mat(self, des_rpy, actual_rot):
        des_rot = rpy_to_rot_mat(des_rpy)
        return rot_mat_to_rpy(des_rot.dot(actual_rot.T))
        
    def _compute_task_hierarchy(self):
        # the items to compute
        self._delta_q[:] = 0
        self._dq_cmd[:] = 0
        self._ddq_cmd[:] = 0
        wbc_steps = self._tasks_num

        # Base Jacobian
        torso_jacobian = self._robot_model.getFrameJacobian(self._torso_frame_id, rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)  # 6 x 24, [:, 6:]=0
        torso_position_jacobian = torso_jacobian[:3]
        torso_orientation_jacobian = torso_jacobian[3:6]

        # Mass matrix
        mass_matrix = self._robot_model.mass(self._q)  # 24 x 24, [:6, :] & [6:, :6] & [6:, the columns for the same leg] != 0
        mass_matrix_inv = scipy.linalg.inv(mass_matrix)  # 24 x 24, all the elements are non-zero
        # compute the contact leg jacobian
        num_legs_on_ground = np.sum(self._contact_state)
        if num_legs_on_ground > 0:
            stance_foot_jacobian = self._compute_stance_foot_position_jacobian()  # 3c x 24, only the columns of the torso and the leg's joints are non-zero
            self._J[0] = stance_foot_jacobian
            jcdqd = []
            for stance_index in np.nonzero(self._contact_state)[0]:
                frame_acc = self._robot_model.frameClassicalAcceleration(None, None, None, self._foot_frame_ids[stance_index], update_kinematics=False).vector[:3]
                jcdqd.extend(frame_acc)
            jcdqd = np.array(jcdqd)  # 3c
            self._ddq_cmd[:] = compute_dynamically_consistent_pseudo_inverse(stance_foot_jacobian, mass_matrix_inv).dot(-jcdqd)  # M^{-1} J^T (J M^{-1} J^T)^{-1} (-J_c \ddot{q}_d) # 24
            # The entire expression calculates what is known as the projection matrix into the null space of the stance foot Jacobian.
            # Physically, this represents a transformation that identifies motions (in the joint space) that do not result in any change in the position or orientation (pose) of the robot's stance foot.
            # In other words, it finds the internal motions of the robot that are "invisible" to the stance foot's pose, allowing the robot to adjust its posture or balance without affecting the foot's contact with the ground.
            self._N[0] = np.eye(self._num_joints+6) - scipy.linalg.pinv(stance_foot_jacobian, rcond=1e-3).dot(stance_foot_jacobian)  # 24 x 24
            self._N_dyn[0] = np.eye(self._num_joints+6) - compute_dynamically_consistent_pseudo_inverse(stance_foot_jacobian, mass_matrix_inv).dot(stance_foot_jacobian)  # 24 x 24
        else:
            self._N[0], self._N_dyn[0] = np.eye(self._num_joints+6), np.eye(self._num_joints+6)

        # compute error, jacobian, dJdq
        self._e[1] = self._des_torso_pos - self._robot.base_pos_w_np[self._env_ids]
        self._dx[1] = self._des_torso_lin_vel
        base_position_kp = self._base_position_kp_loco if self._action_mode in self._locomotion_action_modes else self._base_position_kp_mani
        base_position_kd = self._base_position_kd_loco if self._action_mode in self._locomotion_action_modes else self._base_position_kd_mani
        self._ddx[1] = self._des_torso_lin_acc + base_position_kp * self._e[1] + base_position_kd * (self._dx[1] - self._robot.base_lin_vel_w_np[self._env_ids])
        self._J[1] = torso_position_jacobian
        self._dJdq[1] = self._robot_model.frameClassicalAcceleration(None, None, None, self._torso_frame_id, update_kinematics=False).vector[:3]

        self._e[2] = self._compute_orientation_error_from_two_world_rpy(self._des_torso_rpy, self._robot.base_rpy_w2b_np[self._env_ids])
        self._dx[2] = self._des_torso_ang_vel
        base_orientation_kp = self._base_orientation_kp_loco if self._action_mode in self._locomotion_action_modes else self._base_orientation_kp_mani
        base_orientation_kd = self._base_orientation_kd_loco if self._action_mode in self._locomotion_action_modes else self._base_orientation_kd_mani
        self._ddx[2] = self._des_torso_ang_acc + base_orientation_kp * self._e[2] + base_orientation_kd * (self._dx[2] - self._robot.base_ang_vel_w_np[self._env_ids])
        self._J[2] = torso_orientation_jacobian
        self._dJdq[2] = self._robot_model.frameClassicalAcceleration(None, None, None, self._torso_frame_id, update_kinematics=False).vector[3:6]

        swing_indices = np.nonzero(np.logical_not(self._contact_state))[0]
        if len(swing_indices) > 0:
            swing_dofs = []
            for index in swing_indices:
                swing_dofs.extend([index * 3, index * 3 + 1, index * 3 + 2])
            swing_dofs = np.array(swing_dofs)

            cur_footeef_pos = self._robot.eef_pos_w_np[self._env_ids][swing_indices].flatten() if (self._use_gripper and self._action_mode in self._eef_manipulation_action_modes) else self._robot.foot_pos_w_np[self._env_ids][swing_indices].flatten() 
            swing_footeef_frames = [self._eef_frame_ids[swing_index] for swing_index in swing_indices] if (self._use_gripper and self._action_mode in self._eef_manipulation_action_modes) else [self._foot_frame_ids[swing_index] for swing_index in swing_indices]
            cur_footeef_vel = []
            j3dqd = []
            for swing_footeef_frame in swing_footeef_frames:
                cur_footeef_vel.append(self._robot_model.frameVelocity(None, None, swing_footeef_frame, update_kinematics=False, reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector[:3])
                j3dqd.extend(self._robot_model.frameClassicalAcceleration(None, None, None, swing_footeef_frame, update_kinematics=False).vector[:3])
            cur_footeef_vel = np.concatenate(cur_footeef_vel)
            footeef_position_kp = np.concatenate([self._footeef_position_kp] * len(swing_indices), axis=0)
            footeef_position_kd = np.concatenate([self._footeef_position_kd] * len(swing_indices), axis=0)

            self._e[3] = self._des_footeef_pos[swing_dofs] - cur_footeef_pos
            self._dx[3] = self._des_footeef_vel[swing_dofs]
            self._ddx[3] = self._des_footeef_acc[swing_dofs] + footeef_position_kp * self._e[3] - footeef_position_kd * (self._dx[3] - cur_footeef_vel)
            # print('cur_footeef_vel', cur_footeef_vel)
            # print('footeef speed', np.linalg.norm(cur_footeef_vel))
            self._J[3] = self._compute_eef_jacobian(eef_idx=swing_indices, type='position') if (self._use_gripper and self._action_mode in self._eef_manipulation_action_modes) else self._compute_swing_foot_position_jacobian()
            self._dJdq[3] = np.array(j3dqd)
        else:
            wbc_steps = self._tasks_num - 1

        if self._use_gripper:
        # eef orientation tasks
            # print('des_eef_pos', self._des_eef_pos)
            for i in range(wbc_steps-self._tasks_num, wbc_steps-self._tasks_num+2):
                # print('i', i)
                # print('des_eef_pos[i]', self._des_eef_pos[i])
                self._e[i+4] = self._compute_orientation_error_from_world_rpy_and_rot_mat(self._des_eef_pos[i], self._robot.eef_rot_w_np[self._env_ids][i])
                self._dx[i+4] = self._des_eef_vel[i]
                eef_velocity = self._robot_model.frameVelocity(None, None, self._eef_frame_ids[i], update_kinematics=False, reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector[:]
                self._ddx[i+4] = self._des_eef_acc[i] + self._eef_orientation_kp * self._e[i+4] - self._eef_orientation_kd * (self._dx[i+4] - eef_velocity[3:6])
                self._J[i+4] = self._compute_eef_jacobian(eef_idx=[i], type='orientation')
                self._dJdq[i+4] = np.array(self._robot_model.frameClassicalAcceleration(None, None, None, self._eef_frame_ids[i], update_kinematics=False).vector[3:6])

        for i in range(1, wbc_steps+1):
            if i>1:
                self._N[i-1] = self._N[i-2] @ self._N_i_im1[i-1]
                self._N_dyn[i-1] = self._N_dyn[i-2] @ self._N_i_im1_dyn[i-1]
            self._J_pre[i] = self._J[i] @ self._N[i-1]
            self._J_pre_dyn[i] = self._J[i] @ self._N_dyn[i-1]
            self._J_i_im1[i] = self._J[i] @ (np.eye(self._num_joints+6) - scipy.linalg.pinv(self._J[i-1]) @ self._J[i-1])
            self._N_i_im1[i] = np.eye(self._num_joints+6) - scipy.linalg.pinv(self._J_i_im1[i]) @ self._J_i_im1[i]
            self._J_i_im1_dyn[i] = self._J[i] @ (np.eye(self._num_joints+6) - compute_dynamically_consistent_pseudo_inverse(self._J[i-1], mass_matrix_inv) @ self._J[i-1])
            self._N_i_im1_dyn[i] = np.eye(self._num_joints+6) - compute_dynamically_consistent_pseudo_inverse(self._J_i_im1_dyn[i], mass_matrix_inv) @ self._J_i_im1_dyn[i]

            self._J_pre_inv[i] = scipy.linalg.pinv(self._J_pre[i])
            self._J_pre_dyn_inv[i] = compute_dynamically_consistent_pseudo_inverse(self._J_pre_dyn[i], mass_matrix_inv)
            
            self._dsl_mat[i] = self._J_pre[i].T @ scipy.linalg.inv(self._J_pre[i] @ self._J_pre[i].T + 1e-2 * np.eye(self._J_pre[i].shape[0]))  # 24 x 3c
            
            # compute command
            self._delta_q[:] = self._delta_q + self._J_pre_inv[i] @ (self._e[i] - self._J[i] @ self._delta_q)
            self._dq_cmd[:] = self._dq_cmd + self._J_pre_inv[i] @ (self._dx[i] - self._J[i] @ self._dq_cmd)
            self._ddq_cmd[:] = self._ddq_cmd + self._J_pre_dyn_inv[i] @ (self._ddx[i] - self._dJdq[i] - self._J[i] @ self._ddq_cmd)
        # print('delta q', self._delta_q[-self._num_joints:])
        # print('gripper 0 delta_q', self._delta_q[-self._num_joints:][3:6])
        # print('gripper 1 delta_q', self._delta_q[-self._num_joints:][9:12])
        # print('max 1', np.max(abs(self._delta_q[-self._num_joints:])))
        # print('max 2', np.max(abs(np.concatenate((self._delta_q[-self._num_joints:-self._num_joints+9], self._delta_q[-self._num_joints+10:])))))

        # print('dd q', self._ddq_cmd[-self._num_joints:])
        # print('gripper 0 dd_q', self._ddq_cmd[-self._num_joints:][3:6])
        # print('gripper 1 dd_q', self._ddq_cmd[-self._num_joints:][9:12])
        # print('ddq max', np.max(abs(self._ddq_cmd[-self._num_joints:])), 'idx', np.argmax(abs(self._ddq_cmd[-self._num_joints:])))
        # print('max 2', np.max(abs(np.concatenate((self._ddq_cmd[-self._num_joints:-self._num_joints+9], self._ddq_cmd[-self._num_joints+10:])))))
            # print('delta_q', np.max(abs(self._delta_q[-self._num_joints:])))
            # clip delta q
            # self._delta_q[-self._num_joints:] = np.clip(self._delta_q[-self._num_joints:], -0.08, 0.08)
            
            # DSL
            # self._delta_q[:] = self._delta_q + self._dsl_mat[i] @ (self._e[i] - self._J[i] @ self._delta_q)
            # self._dq_cmd[:] = self._dq_cmd + self._dsl_mat[i] @ (self._dx[i] - self._J[i] @ self._dq_cmd)
            
            
        self._des_q[:] = self._q[-self._num_joints:] + self._delta_q[-self._num_joints:]
        # print('gripper 0 q', self._q[-self._num_joints:][3:6])
        # print('gripper 1 q', self._q[-self._num_joints:][9:12])
        # print('gripper 0 des_q', self._des_q[-self._num_joints:][3:6])
        # print('gripper 1 des_q', self._des_q[-self._num_joints:][9:12])

    def point_to_plane_distance(self, P0, A, B, C):
        # PO: [n, 3] or [3]
        # A, B, C: [3]
        # Create vectors AB and AC
        AB = B - A
        AC = C - A
        
        # Compute the normal vector to the plane (n)
        n = np.cross(AB, AC)
        # Normalize the normal vector
        n_norm = np.linalg.norm(n)
        # If the normal vector's norm is zero, the points A, B, and C are collinear
        if n_norm == 0:
            raise ValueError("The points A, B, and C are collinear")
        # Compute the distance from point P0 to the plane
        distance = np.dot(P0 - A, n) / n_norm
        
        return distance

    def _check_ground_collisions(self):
        # check foot or gripper collisions with the ground during manipulation mode
        if self._robot._cur_fsm_state==FSM_State.MANIPULATION and self._robot._fsm_operation_mode == FSM_OperatingMode.NORMAL:
            # the world frame changes after reset, get the ground height through three feet that are in contact with the ground
            if self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_FOOT or self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_EEF:
                # threshold = np.mean(self._robot.foot_pos_w_np[:, 1:, 2])
                # des_eef_height = self._des_footeef_pos[2]

                des_eef_pos = self._des_footeef_pos[:3]
                ground_ref_plane = self._robot.foot_pos_w_np[:, 1:, :]
            else:
                # threshold = (np.mean(self._robot.foot_pos_w_np[:, 0, 2]) + np.mean(self._robot.foot_pos_w_np[:, 2, 2]) + np.mean(self._robot.foot_pos_w_np[:, 3, 2])) / 3
                # des_eef_height = self._des_footeef_pos[5]

                des_eef_pos = self._des_footeef_pos[3:6]
                ground_ref_plane = np.stack((self._robot.foot_pos_w_np[:, 0, :], self._robot.foot_pos_w_np[:, 2, :], self._robot.foot_pos_w_np[:, 3, :]), axis=1)
            des_dis_eef_to_ref = self.point_to_plane_distance(des_eef_pos, ground_ref_plane[0, 0, :], ground_ref_plane[0, 2, :], ground_ref_plane[0, 1, :])
            # print('threshold', threshold)
            # print('foot pos', np.mean(self._robot.foot_pos_w_np[:, -2:, 2]))
            threshold_gripper = self._ground_collision_gripper_thresh
            threshold_foot = self._ground_collision_foot_thresh
            threshold_knee = self._ground_collision_knee_thresh
            # print('thresh_gripper', threshold_gripper)
            # print('thresh_foot', threshold_foot)
            # print('thresh_knee', threshold_knee)
            if self._action_mode == 2:
                # gripper manipulation
                # print('desired gripper height', des_dis_eef_to_ref)
                if des_dis_eef_to_ref < threshold_gripper:
                    print('gripper collide with the ground')
                    return True
                # when use gripper as eef, we want to make sure the foot will not collide with the ground
                # des_foot_pos_w: [1, num_legs, 3]
                des_foot_pos_w, des_knee_pos_w = self._robot.get_desired_foot_knee_pos_world(self._des_q, self._des_torso_pos, self._des_torso_rpy, True)
                if self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_EEF:
                    des_moving_foot_pos = des_foot_pos_w[0, 0, :]
                else:
                    des_moving_foot_pos = des_foot_pos_w[0, 1, :]
                # print('des_moving_foot_pos', des_moving_foot_pos)
                des_dis_foot_to_ref = self.point_to_plane_distance(des_moving_foot_pos, ground_ref_plane[0, 0, :], ground_ref_plane[0, 2, :], ground_ref_plane[0, 1, :])
                # print('desired foot height', des_dis_foot_to_ref)
                if des_dis_foot_to_ref < threshold_foot:
                    print('foot collide with the ground')
                    return True
            elif self._action_mode == 1:
                # foot manipulation
                # print('desired foot height', des_dis_eef_to_ref)
                if des_dis_eef_to_ref < threshold_foot:
                    print('foot collide with the ground')
                    return True
                des_foot_pos_w, des_knee_pos_w = self._robot.get_desired_foot_knee_pos_world(self._des_q, self._des_torso_pos, self._des_torso_rpy, True)
            
            # both action modes for manipulation should check if the knees will collide the ground
            # if self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_FOOT or self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_EEF:
            #     des_moving_knee_pos = des_knee_pos_w[0, 0, :]
            # else:
            #     des_moving_knee_pos = des_knee_pos_w[0, 1, :]
            des_dis_knee_to_ref = self.point_to_plane_distance(des_knee_pos_w[0, :, :], ground_ref_plane[0, 0, :], ground_ref_plane[0, 2, :], ground_ref_plane[0, 1, :])
            # print('desired knee height', des_dis_knee_to_ref)
            if any(des_dis_knee_to_ref < threshold_knee):
                print('knee collide with the ground')
                return True
        
        return False
    
    def _check_self_collisions(self):
        # collsiion detection during with the simplified collision model
        if self._check_self_collision:
            des_q = np.concatenate((self._des_torso_pos,
                                rpy_to_quat(self._des_torso_rpy).toNumpy().reshape(4,),
                                self._q[-self._num_joints:] + self._delta_q[-self._num_joints:]))
            data = self._raw_robot_model.createData()
            geom_data = pin.GeometryData(self._geom_model)
            pin.computeCollisions(self._raw_robot_model, data, self._geom_model, geom_data, des_q, False)
            for k in range(len(self._geom_model.collisionPairs)):
                cr = geom_data.collisionResults[k]
                cp = self._geom_model.collisionPairs[k]
                if cr.isCollision() and cp.first+1 != cp.second:
                    print(
                        "detected self collision pair:",
                        cp.first,
                        ",",
                        cp.second
                    )
                    return True
        return False
    
    def _check_joint_limits(self):
        if self._action_mode in self._manipulation_action_modes:
            no_joint_limits = np.sum((self._des_q >= self._min_joint_pos) & (self._des_q <= self._max_joint_pos)) == self._des_q.shape[0]
            if not no_joint_limits:
                print('\n-----------Beyond Joint Limits-----------')
                # print(self._des_q >= self._min_joint_pos)
                # print(self._des_q <= self._max_joint_pos)
                beyond_joint_idx = np.where((self._des_q < self._min_joint_pos) | (self._des_q > self._max_joint_pos))[0]
                print('beyond_joint_idx:', beyond_joint_idx)
                print('min_joint_pos:', self._min_joint_pos[beyond_joint_idx])
                print('max_joint_pos:', self._max_joint_pos[beyond_joint_idx])
                # print("cur_q:", self._q[beyond_joint_idx+7])
                # print("del_q:", self._delta_q[beyond_joint_idx+6])
                print('des_q:', self._des_q[beyond_joint_idx])
                # print("---------------------------------")
                # print("DES FOOTEEF Pos:", self._des_footeef_pos)
                # print("CUR FOOTEEF Pos: {}".format(self._robot.eef_pos_w_np[self._env_ids].flatten() if self._robot._use_gripper and self._action_mode in self._manipulation_action_modes else self._robot.foot_pos_w_np[self._env_ids].flatten()))
                # if self._use_gripper:
                #     print("---------------------------------")
                #     print("DES EEF RPY:", self._des_eef_pos)
                #     print("CUR EEF RPY: {}".format(self._robot.eef_rpy_w_np[self._env_ids] if self._robot._use_gripper else 0.0))
                # self.fsm_publisher.publish(0)
                return True
        return False
    
    def _beyond_max_delta_q(self):
        if self._action_mode == 3:
            # locomotion
            if np.sum(abs(self._delta_q[-self._num_joints:]) > self._loco_max_delta_q) > 0:
                print('-----------Beyond Max delta_q-----------')
                beyond_delta_q_idx = np.where((abs(self._delta_q[-self._num_joints:]) > self._loco_max_delta_q))[0]
                print('beyond max delta q idx', beyond_delta_q_idx)
                print('delta_q:', self._delta_q[-self._num_joints:])
                return True
        elif self._action_mode in self._manipulation_action_modes:     
            # manipulation   
            # print('delta q!!!', abs(self._delta_q[-self._num_joints:]))
            mani_max_delta_q = np.ones(self._num_joints) * self._mani_max_delta_q
            if self._use_gripper:
                mani_max_delta_q[self._gripper_joint_idx[0]] = self._mani_max_delta_q_gripper_joint0
                mani_max_delta_q[self._gripper_joint_idx[3]] = self._mani_max_delta_q_gripper_joint0
                mani_max_delta_q[self._gripper_joint_idx[2]] = self._mani_max_delta_q_gripper_joint0
                mani_max_delta_q[self._gripper_joint_idx[5]] = self._mani_max_delta_q_gripper_joint0
                mani_max_delta_q[self._gripper_joint_idx[1]] = self._mani_max_delta_q_gripper_joint0
                mani_max_delta_q[self._gripper_joint_idx[4]] = self._mani_max_delta_q_gripper_joint0
            # print('mani_max_delta_q', mani_max_delta_q)
            if np.sum(abs(self._delta_q[-self._num_joints:]) > mani_max_delta_q) > 0:
                print('-----------Beyond Max delta_q-----------')
                beyond_max_delta_q_idx = np.where((abs(self._delta_q[-self._num_joints:]) > mani_max_delta_q))[0]
                print('beyond max delta q idx', beyond_max_delta_q_idx)
                print('delta_q:', self._delta_q[-self._num_joints:])
                return True
        return False
    
    def _beyond_max_dq(self):
        if self._action_mode == 3:
            # locomotion
            if np.sum(abs(self._dq_cmd[-self._num_joints:]) > self._loco_max_dq):
                print('-----------Beyond Max dq-----------')
                beyond_max_dq_idx = np.where((abs(self._dq_cmd[-self._num_joints:]) > self._loco_max_dq))[0]
                print('beyond max dq idx', beyond_max_dq_idx)
                print('dq_cmd:', self._dq_cmd[-self._num_joints:])
                return True
        elif self._action_mode in self._manipulation_action_modes:    
            # manipulation    
            mani_max_dq = np.ones(self._num_joints) * self._mani_max_dq
            if self._use_gripper:
                mani_max_dq[self._gripper_joint_idx[0]] = self._mani_max_dq_gripper_joint0
                mani_max_dq[self._gripper_joint_idx[3]] = self._mani_max_dq_gripper_joint0
                mani_max_dq[self._gripper_joint_idx[2]] = self._mani_max_dq_gripper_joint0
                mani_max_dq[self._gripper_joint_idx[5]] = self._mani_max_dq_gripper_joint0
                mani_max_dq[self._gripper_joint_idx[1]] = self._mani_max_dq_gripper_joint0
                mani_max_dq[self._gripper_joint_idx[4]] = self._mani_max_dq_gripper_joint0
            # print('mani_max_dq', mani_max_dq)
            if np.sum(abs(self._dq_cmd[-self._num_joints:]) > mani_max_dq)> 0:
                print('-----------Beyond Max dq-----------')
                beyond_max_dq_idx = np.where((abs(self._dq_cmd[-self._num_joints:]) > mani_max_dq))[0]
                print('beyond max dq idx', beyond_max_dq_idx)
                print('dq_cmd:', self._dq_cmd[-self._num_joints:])
                return True
        return False
    
    def _beyond_max_ddq(self):
        if self._action_mode == 3:
            # locomotion
            if np.sum(abs(self._ddq_cmd[-self._num_joints:]) > self._loco_max_ddq):
                print('-----------Beyond Max ddq-----------')
                beyond_max_ddq_idx = np.where((abs(self._ddq_cmd[-self._num_joints:]) > self._loco_max_ddq))[0]
                print('beyond max ddq idx', beyond_max_ddq_idx)
                print('ddq_cmd:', self._ddq_cmd[-self._num_joints:])
                return True
        elif self._action_mode in self._manipulation_action_modes:    
            # manipulation    
            mani_max_ddq = np.ones(self._num_joints) * self._mani_max_ddq
            if self._use_gripper:
                mani_max_ddq[self._gripper_joint_idx[0]] = self._mani_max_ddq_gripper_joint0
                mani_max_ddq[self._gripper_joint_idx[3]] = self._mani_max_ddq_gripper_joint0
                mani_max_ddq[self._gripper_joint_idx[2]] = self._mani_max_ddq_gripper_joint0
                mani_max_ddq[self._gripper_joint_idx[5]] = self._mani_max_ddq_gripper_joint0
                mani_max_ddq[self._gripper_joint_idx[1]] = self._mani_max_ddq_gripper_joint0
                mani_max_ddq[self._gripper_joint_idx[4]] = self._mani_max_ddq_gripper_joint0
            # print('mani_max_ddq', mani_max_ddq)
            if np.sum(abs(self._ddq_cmd[-self._num_joints:]) > mani_max_ddq)> 0:
                print('-----------Beyond Max ddq-----------')
                beyond_max_ddq_idx = np.where((abs(self._ddq_cmd[-self._num_joints:]) > mani_max_ddq))[0]
                print('beyond max ddq idx', beyond_max_ddq_idx)
                print('ddq_cmd:', self._ddq_cmd[-self._num_joints:])
                return True
        return False
    
    def _if_check_self_collisions(self, interval=100):
        self._self_collision_count += 1
        if self._self_collision_count == interval:
            self._self_collision_count = 0
            self._check_self_collision = True
        else:
            self._check_self_collision = False

    def _check_safety(self):
        self._safe_command = True
        # self._foot_clear_command = True
        # make sure to use the command from the first frame
        # check delta q, dq, ddq
        if self._check_ground_collisions():
            self._safe_command = False
            return "ground collision"
        if self._check_self_collisions():
            self._safe_command = False
            return "self collision"
        if self._check_singularity():
            self._safe_command = False
            return "singularity"
        if self._check_joint_limits():
            self._safe_command = False
            return "joint limits"
        if self._beyond_max_delta_q():
            self._safe_command = False
            return "delta q"
        if self._beyond_max_dq():
            self._safe_command = False
            return "dq"
        if self._beyond_max_ddq():
            self._safe_command = False
            return "ddq"
        return "safe command"
            
    def _solve_joint_torques(self):
        # Clip joint commands
        # print('ddq', self._ddq_cmd[6:12])
        # print('ddq max', np.max(abs(self._ddq_cmd[6:12])))
        # if abs(np.max(self._ddq_cmd[6:12])) > 20:
        #     print('ddq max > 20!!!!!!!!!!!!!!!!')
        #     print(abs(self._ddq_cmd[6:12]))
        # print('ddq max 0', np.max(abs(self._ddq_cmd[:6])))
        # print(' self._ddq_cmd[:6]',  self._ddq_cmd[:6])
        self._ddq_cmd[:6] = np.clip(self._ddq_cmd[:6], [-5, -5, -10, -10, -10, -10],
                                    [5, 5, 30, 10, 10, 10])
        self._ddq_cmd[6:] = np.clip(self._ddq_cmd[6:],
                                    np.ones(self._num_joints) * -10,
                                    np.ones(self._num_joints) * 10)
        # print('ddq max 1', np.max(abs(self._ddq_cmd[:6])))
        # compute motor torques
        mass_matrix = self._robot_model.mass(self._q)  # 24 x 24
        foot_jacobian = self._compute_stance_foot_position_jacobian()  # 3*num_legs_on_ground x 24
        coriolis_gravity = self._robot_model.nle(self._q, self._dq)  # 24, coriolis-gravity

        num_legs_on_ground = np.sum(self._contact_state)
        dim_variables = 6 + 3 * num_legs_on_ground

        # Objective: 1/2 x^T G x - a^T x
        Wq = np.diag([20.0, 20.0, 5.0, 1.0, 1.0, 0.2])
        Wf = 1e-5
        G = np.zeros((dim_variables, dim_variables))
        G[:6, :6] = Wq
        G[6:, 6:] = np.eye(3 * num_legs_on_ground) * Wf
        a = np.zeros((dim_variables, ))
        a[:6] = self._ddq_cmd[:6].T.dot(Wq)
        # reference_grf = np.zeros(3 * num_legs_on_ground)
        a[6:] = np.zeros(3 * num_legs_on_ground)

        # Equality constraint (robot dynamics): CE * x = ce
        A = mass_matrix[:6, :6]  # 6 x 6
        jc_t = foot_jacobian.T[:6]  # 6 x 3*num_legs_on_ground #pylint: disable=unsubscriptable-object
        nle = coriolis_gravity[:6]  # 6
        CE = np.zeros((6, dim_variables))
        CE[:, :6] = A
        CE[:, 6:] = -jc_t
        ce = -nle

        # Inequality constraint (friction cone): CI * x >= 0
        friction_coef = 0.6
        friction_constraints_per_leg = np.array([[0., 0., 1.],
                                                 [1, 0, friction_coef],
                                                 [-1, 0, friction_coef],
                                                 [0, 1, friction_coef],
                                                 [0, -1, friction_coef]])

        CI = np.zeros((5 * num_legs_on_ground, dim_variables))
        for idx in range(num_legs_on_ground):
          CI[idx * 5:idx * 5 + 5,
             6 + idx * 3:9 + idx * 3] = friction_constraints_per_leg

        # Call quadprog to solve QP
        C = np.concatenate((CE, -CE, CI), axis=0).T
        b = np.concatenate(
            (ce - 1e-4, -ce - 1e-4, np.zeros(5 * num_legs_on_ground)))

        sol = quadprog.solve_qp(G, a, C, b)
        ddq = self._ddq_cmd.copy()
        ddq[:6] = sol[0][:6]
        fr = sol[0][6:]

        # print('updated delta ddq:', ddq[:6] - self._ddq_cmd[:6])
        self._torque[:] = mass_matrix.dot(ddq) + coriolis_gravity - foot_jacobian.T.dot(fr)  # use the updated ddq
        # print('torque', self._torque[:6])
        # print('max torque', np.max(abs(self._torque[6:12])))
        # if np.max(abs(self._torque[6:12])) > 1:
        #     print('max torque > 1!!!!!!!!!!!!!')
        #     print(self._torque[6:12])

        # print(self._torque[-self._num_joints:])
        # motor_torques = motor_torques[-self._num_joints:]
        # print("-------------------Solve Joint Torques------------")
        # print(f"des_ddq: {des_ddq}")
        # print(f"solved_ddq: {scipy.linalg.pinv(A).dot(jc_t.dot(fr) - nle)[:6]}")
        # print(f"GRF: {fr.reshape((-1, 3))}")
        # print(f"Motor torques: {motor_torques[6:]}")
        # ans = input("Any Key...")
        # if ans in ["Y", "y", "Yes", "yes"]:
        #     import pdb
        #     pdb.set_trace()

    def log_wbc_info(self):
        # if self._robot._log_info_now and np.rad2deg(self._robot.base_rpy_w2b_np[self._env_ids])[2]<-160:
        if self._robot._log_info_now:
        # if max(abs(des_delta_q)) > 0.4:
        # if True:
            print("\n----------------- WBC -------------------------")
            # print('---------------------------------')
            # print('torso_pos:', self._robot_model.framePlacement(None, self._torso_frame_id, update_kinematics=False).translation)
            # print('torso_rot_mat: ', self._robot_model.framePlacement(None, self._torso_frame_id, update_kinematics=False).rotation)
            # print('gripper_state:', self._robot.joint_pos_np[self._env_ids, [3, 4, 5, 9, 10, 11]])
            # print('robot_q:', self._q)
            # print('robot_dq:', self._dq)
            # print('robot_rot_mat:\n', self._robot.base_rot_mat_w2b_np[self._env_ids])
            # print('robot_rot_mat:\n', quat_to_rot_mat(self._q[3:7]))
            print('---------------------------------')
            print("DES Pos: {: .4f}, {: .4f}, {: .4f}".format(*self._des_torso_pos))
            print("CUR Pos: {: .4f}, {: .4f}, {: .4f}".format(*self._robot.base_pos_w_np[self._env_ids]))
            print("DES RPY: {: .4f}, {: .4f}, {: .4f}".format(*np.rad2deg(self._des_torso_rpy)))
            print("CUR RPY: {: .4f}, {: .4f}, {: .4f}".format(*np.rad2deg(self._robot.base_rpy_w2b_np[self._env_ids])))
            # print("DES Lin Vel: {: .4f}, {: .4f}, {: .4f}".format(*self._des_torso_lin_vel))
            # print("CUR Lin Vel: {: .4f}, {: .4f}, {: .4f}".format(*self._robot.base_lin_vel_w_np[self._env_ids]))
            # print("DES Ang Vel: {: .4f}, {: .4f}, {: .4f}".format(*np.rad2deg(self._des_torso_ang_vel)))
            # print("CUR Ang Vel: {: .4f}, {: .4f}, {: .4f}".format(*np.rad2deg(self._robot.base_ang_vel_w_np[self._env_ids])))
            # print("DES Ang Acc: {: .4f}, {: .4f}, {: .4f}".format(*np.rad2deg(self._des_torso_ang_acc)))
            # print("DES Lin Acc: {: .4f}, {: .4f}, {: .4f}".format(*self._des_torso_lin_acc))
            print("---------------------------------")
            print("DES FOOTEEF Pos:", self._des_footeef_pos)
            print("CUR FOOTEEF Pos: {}".format(self._robot.eef_pos_w_np[self._env_ids].flatten() if self._robot._use_gripper and self._action_mode in self._eef_manipulation_action_modes else self._robot.foot_pos_w_np[self._env_ids].flatten()))
            print("---------------------------------")
            # print("DES EEF RPY:", self._des_eef_pos[1])
            # print("DES EEF Rot: \n", rpy_to_rot_mat(self._des_eef_pos[1]))
            # print("CUR EEF RPY: {}".format(self._robot.eef_rpy_w_np[self._env_ids][1] if self._robot._use_gripper else self._robot.foot_rpy_w_np[self._env_ids][1]))
            # print("CUR EEF Rot: \n", self._robot.eef_rot_w_np[self._env_ids][1])
            print("DES EEF RPY:", self._des_eef_pos)
            print("CUR EEF RPY: {}".format(self._robot.eef_rpy_w_np[self._env_ids] if self._robot._use_gripper else 0.0))
            # print("DES EEF Vel:", self._des_eef_vel)
            # print("DES EEF Acc:", self._des_eef_acc)
            print("---------------------------------")
            print('action_mode:', self._action_mode)
            print('contact_state:', self._contact_state)
            print("des_del_q:", self._delta_q[-self._num_joints:])
            print("des_q:", self._q[-self._num_joints:] + self._delta_q[-self._num_joints:])
            print("des_dq:", self._dq_cmd[-self._num_joints:])
            print("des_ddq:", self._ddq_cmd[-self._num_joints:])
            print("motor_torques:", self._torque[-self._num_joints:])

    def set_des_torso_pva(self, pva):
        self._des_torso_pos = pva[:3].copy()
        self._des_torso_rpy = pva[3:6].copy()
        self._des_torso_lin_vel = pva[6:9].copy()
        self._des_torso_ang_vel = pva[9:12].copy()
        self._des_torso_lin_acc = pva[12:15].copy()
        self._des_torso_ang_acc = pva[15:18].copy()

    def set_desired_footeef_pva(self, des_footeef_pva):
        self._des_footeef_pos = des_footeef_pva[0, :].copy()
        self._des_footeef_vel = des_footeef_pva[1, :].copy()
        self._des_footeef_acc = des_footeef_pva[2, :].copy()
        # print('self._des_footeef_pos', self._des_footeef_pos)

    def set_desired_eef_pva(self, des_eef_pva):
        self._des_eef_pos = des_eef_pva[:, :3].copy()
        self._des_eef_vel = des_eef_pva[:, 3:6].copy()
        self._des_eef_acc = des_eef_pva[:, 6:9].copy()

        # in some modes like stance, foot manipulation or transitions, the input eef orientation commands here are the joint angles (3dofs) of the loco-manipulator
        # so in these modes, the eef commands need to be converted to the unified one (rpy in the world frame) for the whole-body controller
        # during single gripper manipulation mode, the inactive gripper will be set with joint angles
        transform_frame_eef_idx = []
        if self._des_eef_frame == 2:
            for i in range(len(self._eef_frame_ids)):
                # print('self._contact_state[i]', self._contact_state[i])
                # print('self._action_mode', self._action_mode)
                if self._contact_state[i] or self._action_mode not in self._eef_manipulation_action_modes:
                    if self._robot._nex_fsm_state==FSM_State.LOCOMANIPULATION and self._robot._fsm_operation_mode==FSM_OperatingMode.TRANSITION and i==self._robot._cfg.loco_manipulation.manipulate_leg_idx:
                        continue
                    elif self._robot._cur_fsm_state==FSM_State.LOCOMANIPULATION and i==self._robot._cfg.loco_manipulation.manipulate_leg_idx and not self._robot._cfg.loco_manipulation.locomotion_only:
                        continue
                    else:
                        transform_frame_eef_idx.append(i)
        # print('self._contact_state', self._contact_state)
        # print('transform_frame_eef_idx', transform_frame_eef_idx)
        # print('des eef pos 0', self._des_eef_pos)
        if len(transform_frame_eef_idx) > 0:
            des_eef_rot_foot = self._gripper_kinematics.forward_kinematics(des_eef_pva[transform_frame_eef_idx, 0:3], transform_frame_eef_idx)
            for i, eef_idx in enumerate(transform_frame_eef_idx):
                foot_rot_world = self._robot_model.framePlacement(None, self._foot_frame_ids[eef_idx], update_kinematics=False).rotation
                self._des_eef_pos[eef_idx] = rot_mat_to_rpy(foot_rot_world.dot(des_eef_rot_foot[i]))
                self._des_eef_vel[eef_idx] = np.zeros(3)
                self._des_eef_acc[eef_idx] = np.zeros(3)
        # print('des eef pos 1', self._des_eef_pos)

    def set_contact_state(self, contact_state):
        self._contact_state = contact_state


    def interpolate_rpy(self, rpy_start, rpy_end, t):
        """
        Interpolate between two RPY rotations.
        
        Parameters:
        rpy_start (array-like): Start rotation as [roll, pitch, yaw]
        rpy_end (array-like): End rotation as [roll, pitch, yaw]
        t (float): Interpolation parameter, 0 <= t <= 1
        
        Returns:
        np.array: Interpolated rotation as [roll, pitch, yaw]
        """
        # Convert RPY to quaternions
        r_start = R.from_euler('xyz', rpy_start)
        r_end = R.from_euler('xyz', rpy_end)

        key_times = [0, 1]
        slerp = Slerp(key_times, R.concatenate([r_start, r_end]))
        
        # Perform slerp
        slerp_rot = slerp(t)
        
        # Convert back to RPY
        interpolated_rpy = slerp_rot.as_euler('xyz')
        
        return interpolated_rpy

    @property
    def last_executed_command(self):
        return self._last_executed_command



