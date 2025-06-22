from fsm.finite_state_machine import FSM_State, Manipulate_Mode
from robot.base_robot import BaseRobot
from planner.switch_planner import SwitchPlanner
from planner.bimanipulation_switcher import BimanipulationSwitcher
from planner.loco_manipulation_switcher import LocoManipulationSwitcher
import numpy as np


class FSMSwitcher:
    def __init__(self, runner, env_ids=0):
        self._runner = runner
        self._robot: BaseRobot = runner._robot
        self._env_ids = env_ids
        self._cfg = self._robot._cfg
        self._use_gripper = self._robot._use_gripper

        # there are two steps from stance to manipulation:
        # 1. move the base (direction depends on left and right)
        # 2. move the foot and loco-manipulator(foot position and servo angles depends on the legged-arm state)
        self._base_right_movement = self._cfg.fsm_switcher.stance_and_manipulation.base_movement.copy()
        self._base_left_movement = self._cfg.fsm_switcher.stance_and_manipulation.base_movement.copy()
        self._base_left_movement[1] *= -1.0
        self._foot_movement = self._cfg.fsm_switcher.stance_and_manipulation.foot_movement.copy()
        self._manipulator_angles = self._cfg.fsm_switcher.stance_and_manipulation.manipulator_angles.copy()
        self._gripper_reset_pos = self._cfg.gripper.reset_pos_sim.copy()
        self._stance_to_manipulation_gripper_delta = self._cfg.fsm_switcher.stance_and_manipulation.stance_to_manipulation_gripper_delta
        self._base_action_time = self._cfg.fsm_switcher.stance_and_manipulation.base_action_time
        self._foot_action_time = self._cfg.fsm_switcher.stance_and_manipulation.foot_action_time
        self._eef_action_time = self._cfg.fsm_switcher.stance_and_manipulation.eef_action_time
        self._manipulate_leg_idx = 0
        self._no_manipulate_leg_idx = 0 if self._manipulate_leg_idx==1 else 1
        self._stance_to_manipulation_trajectory = np.zeros((2, 23))
        self._stance_to_manipulation_trajectory[0, 0:6] = self._base_right_movement
        self._stance_to_manipulation_trajectory[:, 9:12] = self._gripper_reset_pos[0:3]
        self._stance_to_manipulation_trajectory[:, 12:15] = self._gripper_reset_pos[4:7]
        self._stance_to_manipulation_trajectory[:, 19] = self._gripper_reset_pos[3]
        self._stance_to_manipulation_trajectory[:, 20] = self._gripper_reset_pos[7]
        self._stance_to_manipulation_trajectory[0, 15:19] = np.array([0, 1, 1, -1])  # mode(1d), reset_signal(1d), contact_state(1d), swing_leg(1d)
        self._stance_to_manipulation_trajectory[1, 15:19] = np.array([1, 0, 0, -1])
        self._stance_to_manipulation_trajectory[0, 21:23] = np.array([self._base_action_time, 0.1])  # duration (1d), stay time (1d)
        self._stance_to_manipulation_trajectory[1, 21:23] = np.array([self._foot_action_time, 0.1])


        self._manipulation_to_stance_trajectory = self._cfg.fsm_switcher.stance_and_manipulation.manipulation_to_stance_actions
        self._stance_to_locomanipulation_trajectory = self._cfg.fsm_switcher.stance_and_locomanipulation.stance_to_locomanipulation_actions
        # self._manipualtor_angles_ready_to_hold = self._cfg.loco_manipulation.manipualtor_angles_ready_to_hold

        self._stance_to_bimanipulation_switcher = BimanipulationSwitcher(self._robot, self._env_ids, stand_up=True)
        self._bimanipulation_to_stance_switcher = BimanipulationSwitcher(self._robot, self._env_ids, stand_up=False)

        if self._use_gripper:
            self._stance_to_locomanipulation_switcher = LocoManipulationSwitcher(self._robot, self._env_ids)

    def get_switcher(self):
        if self._robot._cur_fsm_state == FSM_State.STANCE and self._runner._fsm_state_buffer == FSM_State.MANIPULATION:
            return self.construct_stance_to_manipulation_switcher()
        elif self._robot._cur_fsm_state == FSM_State.MANIPULATION and self._runner._fsm_state_buffer == FSM_State.STANCE:
            return self.construct_manipulation_to_stance_switcher()
        elif self._robot._cur_fsm_state == FSM_State.STANCE and self._runner._fsm_state_buffer == FSM_State.BIMANIPULATION:
            return self._stance_to_bimanipulation_switcher
        elif self._robot._cur_fsm_state == FSM_State.BIMANIPULATION and self._runner._fsm_state_buffer == FSM_State.STANCE:
            return self._bimanipulation_to_stance_switcher
        elif self._robot._cur_fsm_state == FSM_State.STANCE and self._runner._fsm_state_buffer == FSM_State.LOCOMANIPULATION and self._use_gripper:
            return self._stance_to_locomanipulation_switcher

    def construct_stance_to_manipulation_switcher(self):
        self._robot._update_state(reset_estimator=True)  # to keep the current base pose
        self._stance_to_manipulation_trajectory[:, 9:12] = self._gripper_reset_pos[0:3]
        self._stance_to_manipulation_trajectory[:, 12:15] = self._gripper_reset_pos[4:7]
        self._stance_to_manipulation_trajectory[:, 19] = self._gripper_reset_pos[3]
        self._stance_to_manipulation_trajectory[:, 20] = self._gripper_reset_pos[7]
        self._stance_to_manipulation_trajectory[1, 21] = self._foot_action_time

        # firstly decide the direction of the base movement
        if self._runner._manipulate_mode_buffer in [Manipulate_Mode.LEFT_FOOT, Manipulate_Mode.LEFT_EEF]:
            self._stance_to_manipulation_trajectory[0, 0:6] = self._base_left_movement
            self._manipulate_leg_idx = 1
            self._no_manipulate_leg_idx = 0
        else:
            self._stance_to_manipulation_trajectory[0, 0:6] = self._base_right_movement
            self._manipulate_leg_idx = 0
            self._no_manipulate_leg_idx = 1

        # avoid self-collision
        self._stance_to_manipulation_trajectory[0, 9+3*self._manipulate_leg_idx:12+3*self._manipulate_leg_idx] += self._stance_to_manipulation_gripper_delta[0+4*self._manipulate_leg_idx:3+4*self._manipulate_leg_idx]
        self._stance_to_manipulation_trajectory[0, 19+self._manipulate_leg_idx] += self._stance_to_manipulation_gripper_delta[3+4*self._manipulate_leg_idx]

        # then decide the foot position and manipulator angles
        print("foot pos before base movement:", self._robot.foot_pos_w_np[self._env_ids, self._manipulate_leg_idx])
        foot_pos_w_before_base_movement = self._robot.foot_pos_w_np[self._env_ids, self._manipulate_leg_idx].copy()
        foot_pos_w_after_base_movement = foot_pos_w_before_base_movement - self._stance_to_manipulation_trajectory[0, 0:3]
        manipulation_init_foot_pos_w = foot_pos_w_after_base_movement + self._foot_movement
        self._stance_to_manipulation_trajectory[1, 6:9] = manipulation_init_foot_pos_w

        if self._runner._manipulate_mode_buffer in [Manipulate_Mode.LEFT_EEF, Manipulate_Mode.RIGHT_EEF]:
            self._stance_to_manipulation_trajectory[1, 21] = self._eef_action_time
            self._stance_to_manipulation_trajectory[1, 9+3*self._manipulate_leg_idx:12+3*self._manipulate_leg_idx] = self._manipulator_angles
            # self._stance_to_manipulation_trajectory[1, 9+3*self._no_manipulate_leg_idx:12+3*self._no_manipulate_leg_idx] = self._gripper_reset_pos[4*self._no_manipulate_leg_idx:4*self._no_manipulate_leg_idx+3]

        # print('--------trajectory:', self._stance_to_manipulation_trajectory)
        return SwitchPlanner(self._robot, self._stance_to_manipulation_trajectory, self._manipulate_leg_idx, input_footeef_frame='world')


    def construct_manipulation_to_stance_switcher(self):
        manipulation_to_stance_actions = self._manipulation_to_stance_trajectory.copy()
        manipulation_to_stance_actions[0, 0:3] = self._robot.base_pos_w_np[self._env_ids]
        manipulation_to_stance_actions[0, 3:6] = self._robot.base_rpy_w2b_np[self._env_ids]

        foot_pos_w = self._robot.foot_pos_w_np[self._env_ids].copy()

        self._manipulate_leg_idx = 0 if self._robot._cur_manipulate_mode in [Manipulate_Mode.RIGHT_FOOT, Manipulate_Mode.RIGHT_EEF] else 1
        self._no_manipulate_leg_idx = 0 if self._manipulate_leg_idx==1 else 1
        manipulation_to_stance_actions[0, 6:9] = foot_pos_w[self._manipulate_leg_idx]
        manipulation_to_stance_actions[1, 6:9] = foot_pos_w[self._no_manipulate_leg_idx] + (foot_pos_w[2+self._manipulate_leg_idx] - foot_pos_w[2+self._no_manipulate_leg_idx])

        foot_pos_w[self._manipulate_leg_idx] = manipulation_to_stance_actions[1, 6:9]
        manipulation_to_stance_actions[2, 0:2] = np.mean(foot_pos_w[:, 0:2], axis=0)

        manipulation_to_stance_actions[:, 9+3*self._manipulate_leg_idx:12+3*self._manipulate_leg_idx] += self._stance_to_manipulation_gripper_delta[0+4*self._manipulate_leg_idx:3+4*self._manipulate_leg_idx]
        if not self._robot._cfg.commander.reset_manipulator_when_switch:
            manipulation_to_stance_actions[:, 19+self._manipulate_leg_idx] = self._robot.gripper_angles_np[self._env_ids, self._manipulate_leg_idx]

        if not self._use_gripper:
            manipulation_to_stance_actions[0, -2:] = 0.1  # reduce the time for retreating the gripper

        self._manipulaton_to_stance_switcher = SwitchPlanner(self._robot, manipulation_to_stance_actions, self._manipulate_leg_idx, input_footeef_frame='world')
        return self._manipulaton_to_stance_switcher









