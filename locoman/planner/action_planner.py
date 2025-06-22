import torch
from robot.base_robot import BaseRobot
from planner.body_leg_eefs_planner import BodyfooteefsPlanner


class ActionPlanner:
    def __init__(self, robot: BaseRobot, leg_frame='world', eef_input_frame='world'):
        self._robot = robot
        self._dt = self._robot._dt
        self._num_envs = self._robot._num_envs
        self._device = self._robot._device
        self._cfg = self._robot._cfg
        self._leg_frame = leg_frame
        self._eef_input_frame = eef_input_frame
        self._use_gripper = self._robot._use_gripper

        self._all_env_idx = torch.arange(self._num_envs, device=self._device)
        self._current_action_idx = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._excute_or_during = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)  # excute: True, during: False
        self._finished_cycle_idx = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._contact_state = torch.ones((self._num_envs, 4), dtype=torch.bool, device=self._device)
        # self._close_gripper_action = torch.ones((self._num_envs, len(self._robot._cfg.asset.eef_names)), dtype=torch.long, device=self._device)
        self._gripper_angles_cmd = torch.ones((self._num_envs, len(self._robot._cfg.asset.eef_names)), dtype=torch.float, device=self._device) * self._cfg.gripper.reset_pos_sim[3]
        self._last_desired_gripper_angles = self._gripper_angles_cmd.clone()
        self._next_desired_gripper_angles = self._gripper_angles_cmd.clone()

        self._init_body_state = torch.zeros((self._num_envs, self._cfg.manipulation.body_action_dim), dtype=torch.float, device=self._device)
        self._init_leg_states = torch.zeros((self._num_envs, self._robot._num_legs, self._cfg.manipulation.leg_action_dim), dtype=torch.float, device=self._device)
        self._init_footeef_state = torch.zeros((self._num_envs, self._cfg.manipulation.leg_action_dim), dtype=torch.float, device=self._device)
        self._init_eef_states = torch.zeros((self._num_envs, len(self._robot._cfg.asset.eef_names), self._cfg.manipulation.eef_action_dim), dtype=torch.float, device=self._device)

        self._manipulate_leg_idx = self._cfg.manipulation.manipulate_leg_idx
        self._no_manipulate_leg_idx = 0 if self._manipulate_leg_idx==1 else 1
        self._loco_manipulation_actions = torch.tensor(self._cfg.manipulation.loco_manipulation_actions, dtype=torch.float, device=self._device)
        self._manipulation_actions_num = self._loco_manipulation_actions.shape[0]
        self.manipulate_leg_state_transform()

        # expand 3d footeef state to 12d foot state for compatibility with the gait planner
        self._adaptive_footeef_state = torch.zeros((self._num_envs, 3, self._robot._num_legs*self._cfg.manipulation.leg_action_dim), dtype=torch.float, device=self._device)
        self._manipulate_dofs = torch.tensor([3*self._manipulate_leg_idx, 3*self._manipulate_leg_idx+1, 3*self._manipulate_leg_idx+2], dtype=torch.long, device=self._device)


        self._from_loco_to_locoman_actions = torch.tensor(self._cfg.manipulation.loco_locoman_switch_actions, dtype=torch.float, device=self._device)
        self._from_locoman_to_loco_actions = torch.flip(self._from_loco_to_locoman_actions, dims=[0])
        self._from_locoman_to_loco_actions[:, 0:9] *= -1.0
        self._from_locoman_to_loco_actions[:, 2] -= 0.185
        self._swith_actions_num = self._from_loco_to_locoman_actions.shape[0]
        self._all_actions = torch.cat([self._from_loco_to_locoman_actions, self._loco_manipulation_actions, self._from_locoman_to_loco_actions], dim=0)
        self._all_actions_num = self._all_actions.shape[0]
        self._time_sequences = self._all_actions[:, -2:].clone()
        # self._gripper_closing = self._all_actions[:, -3].clone().to(torch.long)
        self._gripper_angles = self._all_actions[:, -4:-2].clone()

        self._body_leg_eefs_planner = BodyfooteefsPlanner(self._dt, self._num_envs, self._device, self._cfg.manipulation.body_action_dim, self._cfg.manipulation.leg_action_dim, self._cfg.manipulation.eef_action_dim)

        self._reset_signal = self._all_actions[:, -7].clone().to(torch.long)
        self._action_mode = self._all_actions[:, -8].clone().to(torch.long)
        # 0: no switch, 1: stance/locomotion to manipulation, 2: manipulation to stance/locomotion
        self._swith_mode = torch.zeros_like(self._action_mode).to(torch.long)
        for i in range(self._action_mode.shape[0]):
            if self._reset_signal[i]:
                self._swith_mode[i] = 1 if self._action_mode[i]==2 else 2
            else:
                if (self._action_mode[i-1]<2) and self._action_mode[i]==2:
                    self._swith_mode[i] = 1
                elif (self._action_mode[i-1]==2) and (self._action_mode[i]<2):
                    self._swith_mode[i] = 2

        # print('-----------------ActionPlanner-----------------')
        # print('All actions: \n', self._all_actions)
        # print('Action mode: \n', self._action_mode)
        # print('Switch mode: \n', self._swith_mode)

    def manipulate_leg_state_transform(self):
        foot_mani_env_ids = (self._loco_manipulation_actions[:, -8]==1.0).nonzero(as_tuple=False).flatten()
        eef_mani_env_ids = (self._loco_manipulation_actions[:, -8]==2.0).nonzero(as_tuple=False).flatten()

        foot_pos_in_base_frame = self._loco_manipulation_actions[foot_mani_env_ids, 6:9] + self._robot._HIP_OFFSETS[self._manipulate_leg_idx, :]
        if self._leg_frame == 'base':
            self._loco_manipulation_actions[foot_mani_env_ids, 6:9] = foot_pos_in_base_frame
        elif self._leg_frame == 'world':
            from utilities.rotation_utils import rpy_to_rot_mat
            T_w_b = torch.eye(4, dtype=torch.float, device=self._device).repeat(foot_mani_env_ids.shape[0], 1, 1)
            T_w_b[:, :3, 3] = self._loco_manipulation_actions[foot_mani_env_ids, 0:3]
            T_w_b[:, :3, :3] = rpy_to_rot_mat(self._loco_manipulation_actions[foot_mani_env_ids, 3:6])
            self._loco_manipulation_actions[foot_mani_env_ids, 6:9] = torch.matmul(T_w_b, torch.cat([foot_pos_in_base_frame, torch.ones_like(foot_pos_in_base_frame[:, :1])], dim=-1).unsqueeze(-1)).squeeze(-1)[:, :3]

        if self._eef_input_frame != 'world':
            eef_pos_in_base_frame = self._loco_manipulation_actions[eef_mani_env_ids, 6:9]
            eef_rpy_in_base_frame = self._loco_manipulation_actions[eef_mani_env_ids, 3+3*self._manipulate_leg_idx:6+3*self._manipulate_leg_idx]
            if self._leg_frame == 'base':
                self._loco_manipulation_actions[eef_mani_env_ids, 6:9] = eef_pos_in_base_frame
                self._loco_manipulation_actions[eef_mani_env_ids, 3+3*self._manipulate_leg_idx:6+3*self._manipulate_leg_idx] = eef_rpy_in_base_frame
            elif self._leg_frame == 'world':
                from utilities.rotation_utils import rpy_to_rot_mat, rot_mat_to_rpy
                T_w_b = torch.eye(4, dtype=torch.float, device=self._device).repeat(eef_mani_env_ids.shape[0], 1, 1)
                T_w_b[:, :3, 3] = self._loco_manipulation_actions[eef_mani_env_ids, 0:3]
                T_w_b[:, :3, :3] = rpy_to_rot_mat(self._loco_manipulation_actions[eef_mani_env_ids, 3:6])
                print('T_w_b: \n', T_w_b)

                T_b_eef = torch.eye(4, dtype=torch.float, device=self._device).repeat(eef_mani_env_ids.shape[0], 1, 1)
                T_b_eef[:, :3, 3] = eef_pos_in_base_frame
                T_b_eef[:, :3, :3] = rpy_to_rot_mat(eef_rpy_in_base_frame)
                print('T_b_eef: \n', T_b_eef)

                T_w_eef = torch.matmul(T_w_b, T_b_eef)
                self._loco_manipulation_actions[eef_mani_env_ids, 6:9] = T_w_eef[:, :3, 3]
                self._loco_manipulation_actions[eef_mani_env_ids, 3+3*self._manipulate_leg_idx:6+3*self._manipulate_leg_idx] = rot_mat_to_rpy(T_w_eef[:, :3, :3])
                print('T_w_eef: \n', T_w_eef)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = self._all_env_idx.clone()
        self._current_action_idx[env_ids] = 0
        self._excute_or_during[env_ids] = True
        self._finished_cycle_idx[env_ids] = False

        self._contact_state[env_ids, :] = True
        move_manipulate_foot_env_ids = (env_ids[(self._all_actions[self._current_action_idx, -6][env_ids]==0.0) & (self._all_actions[self._current_action_idx, -5][env_ids]==-1)])
        self._contact_state[move_manipulate_foot_env_ids, self._manipulate_leg_idx] = False
        move_non_manipulate_foot_env_ids = (env_ids[(self._all_actions[self._current_action_idx, -6][env_ids]==0.0) & (self._all_actions[self._current_action_idx, -5][env_ids]!=-1)])
        self._contact_state[move_non_manipulate_foot_env_ids, (self._all_actions[self._current_action_idx[move_non_manipulate_foot_env_ids], -5]).to(torch.long)] = False

        self._save_init_leg_states(env_ids)
        self._update_initial_body_state(env_ids)
        self._update_initial_foot_eef_states(env_ids)

        self._body_leg_eefs_planner.set_final_state(body_final_state=self._all_actions[self._current_action_idx, 0:6],
                                                    footeef_final_state=None,
                                                    eef_final_states=self._all_actions[self._current_action_idx, 9:15],
                                                    action_duration=self._time_sequences[self._current_action_idx, 0],
                                                    env_ids=env_ids)

    def _save_init_leg_states(self, env_ids):
        if self._leg_frame == 'hip':
            self._init_leg_states[env_ids] = self._robot.origin_foot_pos_hip[env_ids, :, :]
        elif self._leg_frame == 'base':
            self._init_leg_states[env_ids] = self._robot.origin_foot_pos_b[env_ids, :, :]
        elif self._leg_frame == 'world':
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

        switch_to_manipulation_env_ids = env_ids[(switch_mode==1)]
        switch_to_locomotion_env_ids = env_ids[(switch_mode==2)]

        if self._use_gripper:
            self._init_footeef_state[switch_to_manipulation_env_ids] = self._robot.eef_pos_w[switch_to_manipulation_env_ids, self._manipulate_leg_idx, :]
        if self._leg_frame == 'hip':
            self._init_footeef_state[switch_to_locomotion_env_ids] = self._robot.foot_pos_hip[switch_to_locomotion_env_ids, self._manipulate_leg_idx, :]
        elif self._leg_frame == 'base':
            self._init_footeef_state[switch_to_locomotion_env_ids] = self._robot.foot_pos_b[switch_to_locomotion_env_ids, self._manipulate_leg_idx, :]
        elif self._leg_frame == 'world':
            self._init_footeef_state[switch_to_locomotion_env_ids] = self._robot.foot_pos_w[switch_to_locomotion_env_ids, self._manipulate_leg_idx, :]

        if self._use_gripper:
            self._init_eef_states[switch_to_locomotion_env_ids, self._manipulate_leg_idx] = self._robot.joint_pos[switch_to_locomotion_env_ids, 3+6*self._manipulate_leg_idx:6+6*self._manipulate_leg_idx]
            self._init_eef_states[switch_to_manipulation_env_ids, self._manipulate_leg_idx] = self._robot.eef_rpy_w[switch_to_manipulation_env_ids, self._manipulate_leg_idx, :]
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
        move_manipulate_foot_env_ids = (env_ids[(self._all_actions[self._current_action_idx, -6][env_ids]==0.0) & (self._all_actions[self._current_action_idx, -5][env_ids]==-1)])
        self._contact_state[move_manipulate_foot_env_ids, self._manipulate_leg_idx] = False
        move_non_manipulate_foot_env_ids = (env_ids[(self._all_actions[self._current_action_idx, -6][env_ids]==0.0) & (self._all_actions[self._current_action_idx, -5][env_ids]!=-1)])
        self._contact_state[move_non_manipulate_foot_env_ids, (self._all_actions[self._current_action_idx[move_non_manipulate_foot_env_ids], -5]).to(torch.long)] = False

        body_final_state = self._all_actions[self._current_action_idx, 0:6].clone()
        body_final_state[self._finished_cycle_idx] *= 0.0

        footeef_final_state = self._all_actions[self._current_action_idx, 6:9].clone()
        footeef_final_state[move_non_manipulate_foot_env_ids, :] += self._init_leg_states[move_non_manipulate_foot_env_ids, self._all_actions[self._current_action_idx, -5][move_non_manipulate_foot_env_ids].to(torch.long), :]
        finish_all_target_env_ids = (self._current_action_idx==(self._swith_actions_num+self._manipulation_actions_num-1))
        footeef_final_state[finish_all_target_env_ids, :] = self._init_leg_states[finish_all_target_env_ids, self._manipulate_leg_idx, :]
        full_contact_env_ids = env_ids[(self._all_actions[self._current_action_idx[env_ids], -6]==1.0)]
        footeef_final_state[full_contact_env_ids, :] = self._body_leg_eefs_planner._init_footeef_state[full_contact_env_ids, :]
        # print('footeef_final_state: \n', footeef_final_state)

        self._body_leg_eefs_planner.set_final_state(body_final_state=body_final_state,
                                                    footeef_final_state=footeef_final_state,
                                                    eef_final_states=self._all_actions[self._current_action_idx, 9:15],
                                                    action_duration=self._time_sequences[self._current_action_idx, (~self._excute_or_during).to(torch.long)],
                                                    env_ids=env_ids)

        # print('body_final_state: \n', body_final_state)
        # print('footeef_final_state: \n', footeef_final_state)
        # print('eef_final_states: \n', self._all_actions[self._current_action_idx, 9:15])
        # print('contact_state: \n', self._contact_state)
        # print('action_idx: \n', self._current_action_idx)

    def update(self):
        env_action_end = self._body_leg_eefs_planner.step()  # phase=1.0

        during_env_ids = (self._excute_or_during==False)
        if during_env_ids.any():
            self._gripper_angles_cmd[during_env_ids, :] = (1-self._body_leg_eefs_planner._current_phase[during_env_ids]) * self._last_desired_gripper_angles[during_env_ids, :] + self._body_leg_eefs_planner._current_phase[during_env_ids] * self._next_desired_gripper_angles[during_env_ids, :]
            # print('self._gripper_angles_cmd: \n', self._gripper_angles_cmd)

        env_action_end &= (~self._finished_cycle_idx)  # don't update the environments that have finished the cycle
        if torch.sum(env_action_end) == 0:
            return

        update_gripper_action_env_ids = env_action_end & (self._excute_or_during==True)
        if torch.sum(update_gripper_action_env_ids) != 0:
            self._last_desired_gripper_angles[update_gripper_action_env_ids] = self._gripper_angles_cmd[update_gripper_action_env_ids]
            self._next_desired_gripper_angles[update_gripper_action_env_ids] = self._gripper_angles[self._current_action_idx[update_gripper_action_env_ids]]
        # self._close_gripper_action[update_gripper_action_env_ids, self._manipulate_leg_idx] = self._gripper_closing[self._current_action_idx[update_gripper_action_env_ids]]

        # must compute the enviroments that need to be reset before updating the general items
        reset_robot_env_ids = env_action_end & (self._all_actions[self._current_action_idx, -7]==1.0) & (self._excute_or_during==False)
        if torch.sum(reset_robot_env_ids) != 0:
            reset_robot_env_ids = reset_robot_env_ids.nonzero(as_tuple=False).flatten()
            self._robot._update_state(reset_estimator=True, env_ids=reset_robot_env_ids)
            self._save_init_leg_states(env_ids=reset_robot_env_ids)
            self._update_initial_body_state(env_ids=reset_robot_env_ids)
            self._update_initial_foot_eef_states(env_ids=reset_robot_env_ids)
        
        update_initial_foot_eef_state_env_ids = (env_action_end & (self._excute_or_during==False)).nonzero(as_tuple=False).flatten()

        # update the general items
        self._excute_or_during[env_action_end] = ~self._excute_or_during[env_action_end]
        self._finished_cycle_idx |= env_action_end & (self._current_action_idx==(self._all_actions_num-1)) & (self._excute_or_during==True)
        self._current_action_idx[env_action_end & self._excute_or_during] += 1
        self._current_action_idx = torch.clip(self._current_action_idx, 0, self._all_actions_num-1)

        # update the initial and final states of the planer
        env_action_end = env_action_end.nonzero(as_tuple=False).flatten()
        self._update_initial_foot_eef_states(env_ids=update_initial_foot_eef_state_env_ids)
        self._set_final_states(env_ids=env_action_end)

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












if '__main__' == __name__:
    # import numpy as np
    # loco_locoman_switch_actions = np.array([[-0.04, -0.02, 0.02, .0, .0, .0, 0., 0., 0., 1, 1, 1.5, 0.5],
    #                                         [0., 0., 0., .0, .0, .0, 0., 0., 0.02, 0, 0, 0.5, 0.5],
    #                                         ])
    # from_loco_to_locoman_actions = torch.tensor(loco_locoman_switch_actions).repeat(2, 1, 1)
    # # print(from_loco_to_locoman_actions)
    # from_locoman_to_loco_actions = torch.flip(from_loco_to_locoman_actions, dims=[1])
    # # print(from_locoman_to_loco_actions)
    # all_actions = torch.cat([from_loco_to_locoman_actions, from_locoman_to_loco_actions], dim=1)
    # print(all_actions)

    curent_phase = torch.tensor([0.5, 1.5, 0.5, 1.0])
    endix = curent_phase>=1.0
    print(endix)
    print(curent_phase[endix])
    print(endix.nonzero(as_tuple=False).flatten())
    print(curent_phase[endix.nonzero(as_tuple=False).flatten()])


