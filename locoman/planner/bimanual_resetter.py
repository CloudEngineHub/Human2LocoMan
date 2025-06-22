import torch
import numpy as np
from robot.base_robot import BaseRobot
from planner.body_leg_eefs_planner import BodyfooteefsPlanner


class BimanualResetter:
    def __init__(self, robot: BaseRobot, action_sequences, env_ids=0):
        self._robot = robot
        self._dt = self._robot._dt
        self._num_envs = self._robot._num_envs
        self._device = self._robot._device
        self._cfg = self._robot._cfg
        self._use_gripper = self._robot._use_gripper
        self._env_ids = env_ids

        self._all_env_idx = torch.arange(self._num_envs, device=self._device)
        self._current_action_idx = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._excute_or_during = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)  # excute: True, during: False
        self._finished_cycle_idx = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._contact_state = torch.ones((self._num_envs, 4), dtype=torch.bool, device=self._device)
        # self._contact_state = torch.full((self._num_envs, 4), True, dtype=torch.bool, device=self._device)
        # self._close_gripper_action = torch.ones((self._num_envs, len(self._robot._cfg.asset.eef_names)), dtype=torch.long, device=self._device)
        if self._robot._cfg.commander.reset_manipulator_when_switch:
            self._gripper_angles_cmd = torch.ones((self._num_envs, len(self._robot._cfg.asset.eef_names)), dtype=torch.float, device=self._device) * self._cfg.gripper.reset_pos_sim[3]
        else:
            self._gripper_angles_cmd = self._robot.gripper_angles.clone()
        self._last_desired_gripper_angles = self._gripper_angles_cmd.clone()
        self._next_desired_gripper_angles = self._gripper_angles_cmd.clone()

        self._init_body_state = torch.zeros((self._num_envs, self._cfg.bimanual_manipulation.body_action_dim), dtype=torch.float, device=self._device)
        # self._init_leg_states = torch.zeros((self._num_envs, self._robot._num_legs, self._cfg.bimanual_manipulation.leg_action_dim), dtype=torch.float, device=self._device)
        self._init_footeef_state = torch.zeros((self._num_envs, self._cfg.bimanual_manipulation.leg_action_dim), dtype=torch.float, device=self._device)
        self._init_eef_states = torch.zeros((self._num_envs, len(self._cfg.asset.eef_names), self._cfg.bimanual_manipulation.eef_action_dim), dtype=torch.float, device=self._device)

        self._manipulate_leg_idx = [0, 1]

        self._action_sequences = torch.tensor(action_sequences, dtype=torch.float, device=self._device)

        self._actions_num = self._action_sequences.shape[0]
        self._time_sequences = self._action_sequences[:, -2:].clone()
        # self._gripper_closing = self._action_sequences[:, -3].clone().to(torch.long)
        self._gripper_angles = self._action_sequences[:, -4:-2].clone()
        self._action_mode = self._action_sequences[:, -8].clone().to(torch.long)
        # 0: no switch, 1: stance/locomotion to manipulation, 2: manipulation to stance/locomotion
        # 0: manipulation to stance / no switch, 1: stance/eef-manipulation to foot-manipulation, 2: stance/foot-manipulation to eef-manipulation
        self._reset_mode = torch.zeros_like(self._action_mode).to(torch.long)
        for i in range(self._actions_num):
            if i==0 and self._action_mode[i]:
                self._reset_mode[i] = self._action_mode[i]
            else:
                j = np.clip(i+1, 0, self._actions_num-1)
                if self._action_mode[i] != self._action_mode[j]:
                    self._reset_mode[i] = self._action_mode[j]

        # expand 3d footeef state to 12d foot state for compatibility with the gait planner
        self._adaptive_footeef_state = torch.zeros((self._num_envs, 3, self._robot._num_legs*self._cfg.manipulation.leg_action_dim), dtype=torch.float, device=self._device)
        self._manipulate_dofs = torch.tensor([i for i in range(self._cfg.bimanual_manipulation.leg_action_dim)], dtype=torch.long, device=self._device)
        self._body_leg_eefs_planner = BodyfooteefsPlanner(self._dt, self._num_envs, self._device, self._cfg.bimanual_manipulation.body_action_dim, self._cfg.bimanual_manipulation.leg_action_dim, self._cfg.bimanual_manipulation.eef_action_dim)
        self._desired_rear_legs_pos = np.array([0, 2.35, -2.17, 0, 2.35, -2.17])
        self._desired_rear_legs_vel = np.zeros(6)
        self._desired_rear_legs_torque = np.zeros(6)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = self._all_env_idx.clone()
        self._current_action_idx[env_ids] = 0
        self._excute_or_during[env_ids] = True
        self._finished_cycle_idx[env_ids] = False
        self._contact_state[env_ids, :] = True
        self._contact_state[env_ids, 0] = False
        self._contact_state[env_ids, 1] = False  

        # self._save_init_leg_states(env_ids)
        self._update_initial_body_state(env_ids)
        self._update_initial_foot_eef_states(env_ids)

        self._body_leg_eefs_planner.set_final_state(body_final_state=self._action_sequences[self._current_action_idx, 0:6],
                                                    footeef_final_state=self._action_sequences[self._current_action_idx, 6:12],
                                                    eef_final_states=self._action_sequences[self._current_action_idx, 12:18],
                                                    action_duration=self._time_sequences[self._current_action_idx, 0],
                                                    env_ids=env_ids)

    # def _save_init_leg_states(self, env_ids):
    #     self._init_leg_states[env_ids] = self._robot._base_pos_w[env_ids].unsqueeze(1) + torch.matmul(self._robot._base_rot_mat_w2b[env_ids].unsqueeze(1), self._robot.origin_foot_pos_b[env_ids, :, :].unsqueeze(-1)).squeeze(-1)

    def _update_initial_body_state(self, env_ids=None):
        if env_ids is None:
            env_ids = self._all_env_idx.clone()

        self._init_body_state[env_ids] = torch.concatenate((self._robot._base_pos_w[env_ids], self._robot._base_rpy_w2b[env_ids]), dim=-1)
        self._body_leg_eefs_planner.set_init_state(body_init_state=self._init_body_state,
                                                   footeef_init_state=None,
                                                   eef_init_states=None,
                                                   env_ids=env_ids)
        # print('-----------------update_initial_body_state-----------------')
        # print('self._init_body_state: \n', self._init_body_state)


    def _update_initial_foot_eef_states(self, env_ids):
        if env_ids is None:
            env_ids = self._all_env_idx.clone()

        reset_mode = self._reset_mode[self._current_action_idx][env_ids]
        update_foot_eef_state_env_ids = env_ids[(reset_mode==1) | (reset_mode==2)]
        if len(update_foot_eef_state_env_ids)==0:
            return

        foot_manipulation_env_ids = env_ids[(reset_mode==1)]
        eef_manipulation_env_ids = env_ids[(reset_mode==2)]

        for manipulate_leg_idx in self._manipulate_leg_idx:
            self._init_footeef_state[foot_manipulation_env_ids, 3*manipulate_leg_idx:3*manipulate_leg_idx+3] = self._robot.foot_pos_w[foot_manipulation_env_ids, manipulate_leg_idx, :]
            if self._use_gripper:
                self._init_footeef_state[eef_manipulation_env_ids, 3*manipulate_leg_idx:3*manipulate_leg_idx+3] = self._robot.eef_pos_w[eef_manipulation_env_ids, manipulate_leg_idx, :]
                self._init_eef_states[foot_manipulation_env_ids, manipulate_leg_idx] = self._robot.joint_pos[foot_manipulation_env_ids, 3+6*manipulate_leg_idx:6+6*manipulate_leg_idx]
                self._init_eef_states[eef_manipulation_env_ids, manipulate_leg_idx] = self._robot.eef_rpy_w[eef_manipulation_env_ids, manipulate_leg_idx, :]

        self._body_leg_eefs_planner.set_init_state(body_init_state=None,
                                                   footeef_init_state=self._init_footeef_state,
                                                   eef_init_states=self._init_eef_states,
                                                   env_ids=update_foot_eef_state_env_ids)

    def _set_final_states(self, env_ids):
        self._contact_state[env_ids, :] = True
        self._contact_state[env_ids, 0] = False
        self._contact_state[env_ids, 1] = False  
        # move_manipulate_foot_env_ids = (env_ids[(self._action_sequences[self._current_action_idx, -6][env_ids]==0.0) & (self._action_sequences[self._current_action_idx, -5][env_ids]==-1)])
        # move_non_manipulate_foot_env_ids = (env_ids[(self._action_sequences[self._current_action_idx, -6][env_ids]==0.0) & (self._action_sequences[self._current_action_idx, -5][env_ids]!=-1)])

        body_final_state = self._action_sequences[self._current_action_idx, 0:6].clone()
        body_final_state[self._finished_cycle_idx] *= 0.0

        footeef_final_state = self._action_sequences[self._current_action_idx, 6:12].clone()
        # footeef_final_state[move_non_manipulate_foot_env_ids, :] += self._init_leg_states[move_non_manipulate_foot_env_ids, self._action_sequences[self._current_action_idx, -5][move_non_manipulate_foot_env_ids].to(torch.long), :]

        # this is for returing the foot to the initial state
        # full_contact_env_ids = env_ids[(self._action_sequences[self._current_action_idx[env_ids], -6]==1.0)]
        # footeef_final_state[full_contact_env_ids, :] = self._body_leg_eefs_planner._init_footeef_state[full_contact_env_ids, :]
        # print('footeef_final_state: \n', footeef_final_state)

        self._body_leg_eefs_planner.set_final_state(body_final_state=body_final_state,
                                                    footeef_final_state=footeef_final_state,
                                                    eef_final_states=self._action_sequences[self._current_action_idx, 12:18],
                                                    action_duration=self._time_sequences[self._current_action_idx, (~self._excute_or_during).to(torch.long)],
                                                    env_ids=env_ids)

    def compute_command_for_wbc(self):
        env_action_end = self._body_leg_eefs_planner.step()  # phase=1.0

        during_env_ids = (self._excute_or_during==False)
        if during_env_ids.any():
            self._gripper_angles_cmd[during_env_ids, :] = (1-self._body_leg_eefs_planner._current_phase[during_env_ids]) * self._last_desired_gripper_angles[during_env_ids, :] + self._body_leg_eefs_planner._current_phase[during_env_ids] * self._next_desired_gripper_angles[during_env_ids, :]
            # print('self._gripper_angles_cmd: \n', self._gripper_angles_cmd)

        env_action_end &= (~self._finished_cycle_idx)  # don't update the environments that have finished the cycle
        if torch.sum(env_action_end) != 0:
            update_gripper_action_env_ids = env_action_end & (self._excute_or_during==True)
            if torch.sum(update_gripper_action_env_ids) != 0:
                self._last_desired_gripper_angles[update_gripper_action_env_ids] = self._gripper_angles_cmd[update_gripper_action_env_ids]
                self._next_desired_gripper_angles[update_gripper_action_env_ids] = self._gripper_angles[self._current_action_idx[update_gripper_action_env_ids]]
            # self._close_gripper_action[update_gripper_action_env_ids, self._manipulate_leg_idx] = self._gripper_closing[self._current_action_idx[update_gripper_action_env_ids]]

            # must compute the enviroments that need to be reset before updating the general items
            reset_robot_env_ids = env_action_end & (self._action_sequences[self._current_action_idx, -7]==1.0) & (self._excute_or_during==False)
            if torch.sum(reset_robot_env_ids) != 0:
                reset_robot_env_ids = reset_robot_env_ids.nonzero(as_tuple=False).flatten()
                self._robot._update_state(reset_estimator=True, env_ids=reset_robot_env_ids)
                # self._save_init_leg_states(env_ids=reset_robot_env_ids)
                self._update_initial_body_state(env_ids=reset_robot_env_ids)
                self._update_initial_foot_eef_states(env_ids=reset_robot_env_ids)

            update_initial_foot_eef_state_env_ids = (env_action_end & (self._excute_or_during==False)).nonzero(as_tuple=False).flatten()

            # update the general items
            self._excute_or_during[env_action_end] = ~self._excute_or_during[env_action_end]
            self._finished_cycle_idx |= env_action_end & (self._current_action_idx==(self._actions_num-1)) & (self._excute_or_during==True)
            self._current_action_idx[env_action_end & self._excute_or_during] += 1
            self._current_action_idx = torch.clip(self._current_action_idx, 0, self._actions_num-1)

            # update the initial and final states of the planner
            env_action_end = env_action_end.nonzero(as_tuple=False).flatten()
            self._update_initial_foot_eef_states(env_ids=update_initial_foot_eef_state_env_ids)
            self._set_final_states(env_ids=env_action_end)

        return {"action_mode": self.get_action_mode()[self._env_ids],
                "body_pva": self.get_desired_body_pva()[self._env_ids].cpu().numpy(),
                "footeef_pva": self.get_desired_footeef_pva()[self._env_ids].cpu().numpy(),
                "eef_pva": self.get_desired_eef_pva()[self._env_ids].cpu().numpy(),
                "contact_state": self._contact_state[self._env_ids].cpu().numpy(),
                "gripper_angles": self._gripper_angles_cmd[self._env_ids].cpu().numpy(),
                "rear_legs_command": {"pos": self._desired_rear_legs_pos, "vel": self._desired_rear_legs_vel, "torque": self._desired_rear_legs_torque},
                }

    def feedback(self, command_executed):
        pass

    def check_finished(self):
        # print('self._finished_cycle_idx: \n', self._finished_cycle_idx)
        return torch.sum(self._finished_cycle_idx) > 0
    
    def get_action_mode(self):
        return self._action_mode[self._current_action_idx]
        
    def get_desired_body_pva(self):
        return self._body_leg_eefs_planner.get_desired_body_pva()
    
    def get_desired_footeef_pva(self):
        self._adaptive_footeef_state[:, :, self._manipulate_dofs] = self._body_leg_eefs_planner.get_desired_footeef_pva().reshape(self._num_envs, 3, -1)
        return self._adaptive_footeef_state

    def get_desired_eef_pva(self):
        return self._body_leg_eefs_planner.get_desired_eef_pva()

    def get_contact_state(self):
        return self._contact_state
    
    # def get_gripper_closing(self):
    #     return self._close_gripper_action
    
    def get_gripper_angles_cmd(self):
        return self._gripper_angles_cmd





