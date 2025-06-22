import torch
import numpy as np
from robot.base_robot import BaseRobot
from planner.body_leg_eefs_planner import BodyfooteefsPlanner


class ResetPlanner:
    def __init__(self, robot: BaseRobot, action_sequences, manipulate_leg_idx, input_footeef_frame='hip', output_footeef_frame='world', env_ids=0):
        self._robot = robot
        self._dt = self._robot._dt
        self._num_envs = self._robot._num_envs
        self._device = self._robot._device
        self._cfg = self._robot._cfg
        self._input_footeef_frame = input_footeef_frame
        self._output_footeef_frame = output_footeef_frame
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

        self._init_body_state = torch.zeros((self._num_envs, self._cfg.manipulation.body_action_dim), dtype=torch.float, device=self._device)
        self._init_leg_states = torch.zeros((self._num_envs, self._robot._num_legs, self._cfg.manipulation.leg_action_dim), dtype=torch.float, device=self._device)
        self._init_footeef_state = torch.zeros((self._num_envs, self._cfg.manipulation.leg_action_dim), dtype=torch.float, device=self._device)
        self._init_eef_states = torch.zeros((self._num_envs, len(self._cfg.asset.eef_names), self._cfg.manipulation.eef_action_dim), dtype=torch.float, device=self._device)

        self._manipulate_leg_idx = manipulate_leg_idx
        self._no_manipulate_leg_idx = 0 if self._manipulate_leg_idx==1 else 1

        self._action_sequences = torch.tensor(action_sequences, dtype=torch.float, device=self._device)
        if self._output_footeef_frame != self._input_footeef_frame:
            self.manipulate_leg_state_transform()
        self._actions_num = self._action_sequences.shape[0]
        self._time_sequences = self._action_sequences[:, -2:].clone()
        # self._gripper_closing = self._action_sequences[:, -3].clone().to(torch.long)
        self._gripper_angles = self._action_sequences[:, -4:-2].clone()


        self._reset_signal = self._action_sequences[:, -7].clone().to(torch.long)
        self._action_mode = self._action_sequences[:, -8].clone().to(torch.long)
        # 0: no switch, 1: stance/locomotion to manipulation, 2: manipulation to stance/locomotion
        # 0: manipulation to stance / no switch, 1: stance/eef-manipulation to foot-manipulation, 2: stance/foot-manipulation to eef-manipulation
        self._swith_mode = torch.zeros_like(self._action_mode).to(torch.long)
        # for i in range(self._actions_num):
        #     j = np.clip(i+1, 0, self._actions_num-1)
        #     if self._action_mode[i] != self._action_mode[j]:
        #         self._swith_mode[i] = self._action_mode[j]
        for i in range(self._actions_num):
            if i==0 and self._action_mode[i]:
                self._swith_mode[i] = self._action_mode[i]
            else:
                j = np.clip(i+1, 0, self._actions_num-1)
                if self._action_mode[i] != self._action_mode[j]:
                    self._swith_mode[i] = self._action_mode[j]
        # print('self._swith_mode: \n', self._swith_mode)

        # expand 3d footeef state to 12d foot state for compatibility with the gait planner
        self._adaptive_footeef_state = torch.zeros((self._num_envs, 3, self._robot._num_legs*self._cfg.manipulation.leg_action_dim), dtype=torch.float, device=self._device)
        self._manipulate_dofs = torch.tensor([3*self._manipulate_leg_idx, 3*self._manipulate_leg_idx+1, 3*self._manipulate_leg_idx+2], dtype=torch.long, device=self._device)
        self._body_leg_eefs_planner = BodyfooteefsPlanner(self._dt, self._num_envs, self._device, self._cfg.manipulation.body_action_dim, self._cfg.manipulation.leg_action_dim, self._cfg.manipulation.eef_action_dim)

        # print('-----------------ActionPlanner-----------------')
        # print('All actions: \n', self._action_sequences)
        # print('Action mode: \n', self._action_mode)
        # print('Switch mode: \n', self._swith_mode)

    def manipulate_leg_state_transform(self):
        locomotion_env_ids = (self._action_sequences[:, -8]==1.0).nonzero(as_tuple=False).flatten()
        if self._input_footeef_frame != 'world':
            if self._input_footeef_frame == 'hip':
                leg_state_in_base_frame = self._action_sequences[locomotion_env_ids, 6:9] + self._robot._HIP_OFFSETS[self._manipulate_leg_idx, :]
            else:
                leg_state_in_base_frame = self._action_sequences[locomotion_env_ids, 6:9]
            if self._output_footeef_frame == 'base':
                self._action_sequences[locomotion_env_ids, 6:9] = leg_state_in_base_frame
            elif self._output_footeef_frame == 'world':
                from utilities.rotation_utils import rpy_to_rot_mat
                T_w_b = torch.eye(4, dtype=torch.float, device=self._device).repeat(locomotion_env_ids.shape[0], 1, 1)
                T_w_b[:, :3, 3] = self._action_sequences[locomotion_env_ids, 0:3]
                T_w_b[:, :3, :3] = rpy_to_rot_mat(self._action_sequences[locomotion_env_ids, 3:6])
                self._action_sequences[locomotion_env_ids, 6:9] = torch.matmul(T_w_b, torch.cat([leg_state_in_base_frame, torch.ones_like(leg_state_in_base_frame[:, :1])], dim=-1).unsqueeze(-1)).squeeze(-1)[:, :3]

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = self._all_env_idx.clone()
        self._current_action_idx[env_ids] = 0
        self._excute_or_during[env_ids] = True
        self._finished_cycle_idx[env_ids] = False

        self._contact_state[env_ids, :] = True
        move_manipulate_foot_env_ids = (env_ids[(self._action_sequences[self._current_action_idx, -6][env_ids]==0.0) & (self._action_sequences[self._current_action_idx, -5][env_ids]==-1)])
        self._contact_state[move_manipulate_foot_env_ids, self._manipulate_leg_idx] = False
        move_non_manipulate_foot_env_ids = (env_ids[(self._action_sequences[self._current_action_idx, -6][env_ids]==0.0) & (self._action_sequences[self._current_action_idx, -5][env_ids]!=-1)])
        self._contact_state[move_non_manipulate_foot_env_ids, (self._action_sequences[self._current_action_idx[move_non_manipulate_foot_env_ids], -5]).to(torch.long)] = False

        self._save_init_leg_states(env_ids)
        self._update_initial_body_state(env_ids)
        self._update_initial_foot_eef_states(env_ids)

        self._body_leg_eefs_planner.set_final_state(body_final_state=self._action_sequences[self._current_action_idx, 0:6],
                                                    footeef_final_state=None,
                                                    eef_final_states=self._action_sequences[self._current_action_idx, 9:15],
                                                    action_duration=self._time_sequences[self._current_action_idx, 0],
                                                    env_ids=env_ids)

    def _save_init_leg_states(self, env_ids):
        if self._output_footeef_frame == 'hip':
            self._init_leg_states[env_ids] = self._robot.origin_foot_pos_hip[env_ids, :, :]
        elif self._output_footeef_frame == 'base':
            self._init_leg_states[env_ids] = self._robot.origin_foot_pos_b[env_ids, :, :]
        elif self._output_footeef_frame == 'world':
            self._init_leg_states[env_ids] = self._robot._base_pos_w[env_ids].unsqueeze(1) + torch.matmul(self._robot._base_rot_mat_w2b[env_ids].unsqueeze(1), self._robot.origin_foot_pos_b[env_ids, :, :].unsqueeze(-1)).squeeze(-1)
        # print('self._init_leg_states: \n', self._init_leg_states)

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

        switch_mode = self._swith_mode[self._current_action_idx][env_ids]
        update_foot_eef_state_env_ids = env_ids[(switch_mode==1) | (switch_mode==2)]
        if len(update_foot_eef_state_env_ids)==0:
            return

        switch_to_foot_manipulation_env_ids = env_ids[(switch_mode==1)]
        switch_to_eef_manipulation_env_ids = env_ids[(switch_mode==2)]

        if self._output_footeef_frame == 'world':
            self._init_footeef_state[switch_to_foot_manipulation_env_ids] = self._robot.foot_pos_w[switch_to_foot_manipulation_env_ids, self._manipulate_leg_idx, :]
        elif self._output_footeef_frame == 'base':
            self._init_footeef_state[switch_to_foot_manipulation_env_ids] = self._robot.foot_pos_b[switch_to_foot_manipulation_env_ids, self._manipulate_leg_idx, :]
        elif self._output_footeef_frame == 'hip':
            self._init_footeef_state[switch_to_foot_manipulation_env_ids] = self._robot.foot_pos_hip[switch_to_foot_manipulation_env_ids, self._manipulate_leg_idx, :]

        if self._use_gripper:
            self._init_footeef_state[switch_to_eef_manipulation_env_ids] = self._robot.eef_pos_w[switch_to_eef_manipulation_env_ids, self._manipulate_leg_idx, :]
            self._init_eef_states[switch_to_foot_manipulation_env_ids, self._manipulate_leg_idx] = self._robot.joint_pos[switch_to_foot_manipulation_env_ids, 3+6*self._manipulate_leg_idx:6+6*self._manipulate_leg_idx]
            self._init_eef_states[switch_to_eef_manipulation_env_ids, self._manipulate_leg_idx] = self._robot.eef_rpy_w[switch_to_eef_manipulation_env_ids, self._manipulate_leg_idx, :]
            self._init_eef_states[env_ids, self._no_manipulate_leg_idx] = self._robot.joint_pos[env_ids, 3+6*self._no_manipulate_leg_idx:6+6*self._no_manipulate_leg_idx]

        self._body_leg_eefs_planner.set_init_state(body_init_state=None,
                                                   footeef_init_state=self._init_footeef_state,
                                                   eef_init_states=self._init_eef_states,
                                                   env_ids=update_foot_eef_state_env_ids)

        # print('-----------------update_initial_foot_eef_states-----------------')
        # print('self._init_footeef_state: \n', self._init_footeef_state)
        # print('self._init_eef_states: \n', self._init_eef_states)


    def _set_final_states(self, env_ids):
        self._contact_state[env_ids, :] = True
        move_manipulate_foot_env_ids = (env_ids[(self._action_sequences[self._current_action_idx, -6][env_ids]==0.0) & (self._action_sequences[self._current_action_idx, -5][env_ids]==-1)])
        self._contact_state[move_manipulate_foot_env_ids, self._manipulate_leg_idx] = False
        move_non_manipulate_foot_env_ids = (env_ids[(self._action_sequences[self._current_action_idx, -6][env_ids]==0.0) & (self._action_sequences[self._current_action_idx, -5][env_ids]!=-1)])
        self._contact_state[move_non_manipulate_foot_env_ids, (self._action_sequences[self._current_action_idx[move_non_manipulate_foot_env_ids], -5]).to(torch.long)] = False

        body_final_state = self._action_sequences[self._current_action_idx, 0:6].clone()
        body_final_state[self._finished_cycle_idx] *= 0.0

        footeef_final_state = self._action_sequences[self._current_action_idx, 6:9].clone()
        footeef_final_state[move_non_manipulate_foot_env_ids, :] += self._init_leg_states[move_non_manipulate_foot_env_ids, self._action_sequences[self._current_action_idx, -5][move_non_manipulate_foot_env_ids].to(torch.long), :]

        # this is for returing the foot to the initial state
        # finish_all_target_env_ids = (self._current_action_idx==(self._swith_actions_num+self._manipulation_actions_num-1))
        # footeef_final_state[finish_all_target_env_ids, :] = self._init_leg_states[finish_all_target_env_ids, self._manipulate_leg_idx, :]
        full_contact_env_ids = env_ids[(self._action_sequences[self._current_action_idx[env_ids], -6]==1.0)]
        footeef_final_state[full_contact_env_ids, :] = self._body_leg_eefs_planner._init_footeef_state[full_contact_env_ids, :]
        # print('footeef_final_state: \n', footeef_final_state)

        # iterate between action execution and rest
        # rest duration can be regarded as a trajectory
        self._body_leg_eefs_planner.set_final_state(body_final_state=body_final_state,
                                                    footeef_final_state=footeef_final_state,
                                                    eef_final_states=self._action_sequences[self._current_action_idx, 9:15],
                                                    action_duration=self._time_sequences[self._current_action_idx, (~self._excute_or_during).to(torch.long)],
                                                    env_ids=env_ids)

        # print('-----------------set_final_states-----------------')
        # print('body_final_state: \n', body_final_state)
        # print('footeef_final_state: \n', footeef_final_state)
        # print('eef_final_states: \n', self._action_sequences[self._current_action_idx, 9:15])
        # print('contact_state: \n', self._contact_state)
        # print('action_idx: \n', self._current_action_idx)

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
                self._save_init_leg_states(env_ids=reset_robot_env_ids)
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

        # return int(torch.sum(self._finished_cycle_idx) > 0)
        # return {"action_mode": self.get_action_mode(),
        #         "body_pva": self.get_desired_body_pva(),
        #         "contact_state": self.get_contact_state(),
        #         "footeef_pva": self.get_desired_footeef_pva(),
        #         "eef_pva": self.get_desired_eef_pva(),
        #         }
        # if self._print_command:
        #     print('-----------------compute_command_for_wbc-----------------')
        #     print('action_mode: \n', self.get_action_mode())
        #     print('body_pva: \n', self.get_desired_body_pva())
        #     print('footeef_pva: \n', self.get_desired_footeef_pva())
        #     print('eef_pva: \n', self.get_desired_eef_pva())
        #     print('contact_state: \n', self.get_contact_state())
        #     print('gripper_angles: \n', self.get_gripper_angles_cmd())
        # if self._current_action_idx[0] ==1:
        #     input('any key to continue')

        return {"action_mode": self.get_action_mode()[self._env_ids],
                "body_pva": self.get_desired_body_pva()[self._env_ids].cpu().numpy(),
                "footeef_pva": self.get_desired_footeef_pva()[self._env_ids].cpu().numpy(),
                "eef_pva": self.get_desired_eef_pva()[self._env_ids].cpu().numpy(),
                "contact_state": self._contact_state[self._env_ids].cpu().numpy(),
                "gripper_angles": self._gripper_angles_cmd[self._env_ids].cpu().numpy(),
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





