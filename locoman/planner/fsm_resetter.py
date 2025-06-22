from fsm.finite_state_machine import FSM_State, Manipulate_Mode
from robot.base_robot import BaseRobot
from planner.reset_planner import ResetPlanner
from planner.bimanual_resetter import BimanualResetter
from planner.stance_resetter import StanceResetter
import numpy as np


class FSMResetter:
    def __init__(self, runner, env_ids=0):
        # define and perform the planner for reset motions WITHIN the mode of LocoMan

        self._runner = runner
        self._robot: BaseRobot = runner._robot
        self._env_ids = env_ids
        self._cfg = self._robot._cfg
        self._use_gripper = self._robot._use_gripper

    def init_trajectory(self):
        self.init_reset_pose()
        if self._robot._cur_fsm_state == FSM_State.MANIPULATION:
            self._manipulate_leg_idx = 0 if self._robot._cur_manipulate_mode in [Manipulate_Mode.RIGHT_FOOT, Manipulate_Mode.RIGHT_EEF] else 1
            self._no_manipulate_leg_idx = 0 if self._manipulate_leg_idx==1 else 1

            # two-stage trajectory, first move the gripper, and then move the leg and the torso
            # one-stage info: body state (6d), footeef_position(3d), eef_joint_pos(6d), mode(1d), reset_signal(1d), contact_state(1d), swing_leg(1d), gripper_angles(2d), duration (1d), stay time (1d)
            # mode(1d): stance-0, foot_manipulation-1, eef_manipulation-2, locomotion-3, loco-manipulation-4
            self._manipulation_reset_trajectory = np.zeros((2, 23))
            # set mode: foot manipulation mode
            self._manipulation_reset_trajectory[:, 15] = 1
            # set torso poses
            self._manipulation_reset_trajectory[0, 0:3] = self._robot.base_pos_w_np[self._env_ids]
            self._manipulation_reset_trajectory[0, 3:6] = self._robot.base_rpy_w2b_np[self._env_ids]
            self._manipulation_reset_trajectory[1, :6] = self._base_reset_pose
            # set foot position (for foot manipulation mode wbc)
            foot_pos_w = self._robot.foot_pos_w_np[self._env_ids]
            self._manipulation_reset_trajectory[0, 6:9] = foot_pos_w[self._manipulate_leg_idx]
            self._manipulation_reset_trajectory[1, 6:9] = foot_pos_w[self._no_manipulate_leg_idx] + (foot_pos_w[2+self._manipulate_leg_idx] - foot_pos_w[2+self._no_manipulate_leg_idx]) + self._foot_reset_offset
            # set manipulator angles (for foot manipulation mode wbc)
            self._manipulation_reset_trajectory[:, 9:15] = self._manipulator_rest_angles
            self._manipulation_reset_trajectory[:, 9+3*self._manipulate_leg_idx:12+3*self._manipulate_leg_idx] = self._manipulator_reset_angles
            # set reset signal
            self._manipulation_reset_trajectory[:, 16] = 0
            # set contact state
            self._manipulation_reset_trajectory[:, 17] = 0
            # set swing leg
            self._manipulation_reset_trajectory[:, 18] = -1
            # set gripper angles
            self._manipulation_reset_trajectory[:, 19:21] = self._gripper_rest_angles
            self._manipulation_reset_trajectory[:, 19+self._manipulate_leg_idx] = self._gripper_reset_angles
            # set the durations of the reset motions
            self._manipulation_reset_trajectory[0, 21] = 1.0
            self._manipulation_reset_trajectory[1, 21] = 1.0
            # set stay time (pause after the motion)
            self._manipulation_reset_trajectory[0, 22] = 0.1
            self._manipulation_reset_trajectory[1, 22] = 0.1   

        elif self._robot._cur_fsm_state == FSM_State.BIMANIPULATION:
            # one-stage info: body state (6d), footeef_position(6d), eef_joint_pos(6d), mode(1d), reset_signal(1d), contact_state(1d), swing_leg(1d), gripper_angles(2d), duration (1d), stay time (1d)
            self._manipulation_reset_trajectory = np.zeros((1, 26))
            # set mode: foot manipulation mode, where eef motor angles could be directly set
            self._manipulation_reset_trajectory[0, 18] = 1
            # set torso poses
            self._manipulation_reset_trajectory[0, 0:3] = self._robot.base_pos_w_np[self._env_ids]
            self._manipulation_reset_trajectory[0, 3:6] = self._robot.base_rpy_w2b_np[self._env_ids]
            # set foot position (for foot manipulation mode wbc)
            self._manipulation_reset_trajectory[0, 6:12] = self._foot_reset_pose
            # set manipulator angles (for foot manipulation mode wbc)
            self._manipulation_reset_trajectory[0, 12:18] = self._manipulator_reset_angles
            # set reset signal
            self._manipulation_reset_trajectory[0, 19] = 0
            # set contact state
            self._manipulation_reset_trajectory[0, 20] = 0
            # set swing leg
            self._manipulation_reset_trajectory[0, 21] = -1
            # set gripper angles
            self._manipulation_reset_trajectory[0, 22:24] = self._gripper_reset_angles
            # set the durations of the reset motions
            self._manipulation_reset_trajectory[0, 24] = 2
            # set stay time (pause after the motion)
            self._manipulation_reset_trajectory[0, 25] = 0.3

    def init_reset_pose(self):
        if self._robot._cur_fsm_state == FSM_State.MANIPULATION:
            # self._base_reset_pose = self._cfg.fsm_resetter.manipulation.base_reset_pose.copy()
            # self._foot_reset_offset = self._cfg.fsm_resetter.manipulation.foot_reset_offset.copy()
            # self._manipulator_reset_angles = self._cfg.fsm_resetter.manipulation.manipulator_reset_angles.copy()
            self._base_reset_pose = np.random.uniform(self._cfg.fsm_resetter.manipulation.base_reset_pose_range[0], 
                                                      self._cfg.fsm_resetter.manipulation.base_reset_pose_range[1])
            self._foot_reset_offset = np.random.uniform(self._cfg.fsm_resetter.manipulation.foot_reset_offset_range[0],
                                                        self._cfg.fsm_resetter.manipulation.foot_reset_offset_range[1])
            self._manipulator_reset_angles = np.random.uniform(self._cfg.fsm_resetter.manipulation.manipulator_reset_angles_range[0],
                                                               self._cfg.fsm_resetter.manipulation.manipulator_reset_angles_range[1])

            self._manipulator_rest_angles = self._cfg.fsm_resetter.manipulation.manipulator_rest_angles.copy()
            self._gripper_reset_angles = self._cfg.fsm_resetter.manipulation.gripper_reset_angles.copy()
            self._gripper_rest_angles = self._cfg.fsm_resetter.manipulation.gripper_rest_angles.copy()

        elif self._robot._cur_fsm_state == FSM_State.BIMANIPULATION:
            self._foot_reset_pose = np.random.uniform(self._cfg.fsm_resetter.bimanual_manipulation.foot_reset_pose_range[0],
                                                      self._cfg.fsm_resetter.bimanual_manipulation.foot_reset_pose_range[1])
            self._manipulator_reset_angles = np.random.uniform(self._cfg.fsm_resetter.bimanual_manipulation.manipulator_reset_angles_range[0],
                                                               self._cfg.fsm_resetter.bimanual_manipulation.manipulator_reset_angles_range[1])
            self._gripper_reset_angles = self._cfg.fsm_resetter.bimanual_manipulation.gripper_reset_angles.copy()

    def get_resetter(self):
        if not self._robot._cur_fsm_state == FSM_State.STANCE:
            self.init_trajectory()
            return self.construct_manipulation_resetter()
        else:
            return self.construct_stance_resetter()
        
    def construct_manipulation_resetter(self):
        self._robot._update_state(reset_estimator=False)
        if self._robot._cur_fsm_state == FSM_State.MANIPULATION:
            return ResetPlanner(self._robot, self._manipulation_reset_trajectory, self._manipulate_leg_idx, input_footeef_frame='world')
        elif self._robot._cur_fsm_state == FSM_State.BIMANIPULATION:
            return BimanualResetter(self._robot, self._manipulation_reset_trajectory)
        
    def construct_stance_resetter(self):
        self._robot._update_state(reset_estimator=False)
        return StanceResetter(self._robot)







