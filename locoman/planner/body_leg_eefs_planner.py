import torch
from planner.trajectory_planner import TrajectoryPlanner


class BodyfooteefsPlanner:
    def __init__(self, dt, num_envs, device, body_action_dim, footeef_action_dim, eef_action_dim):
        self._dt = dt
        self._num_envs = num_envs
        self._device = device
        self._body_action_dim = body_action_dim
        self._footeef_action_dim = footeef_action_dim
        self._eef_action_dim = eef_action_dim
        self._init_buffers()

    def _init_buffers(self):
        # general information
        self._all_env_idx = torch.arange(self._num_envs, device=self._device)
        self._action_duration = torch.ones(self._num_envs, device=self._device)
        self._current_phase = torch.zeros(self._num_envs, device=self._device)

        # body trajectory
        self._init_body_state = torch.zeros((self._num_envs, self._body_action_dim), device=self._device)
        self._final_body_state = torch.zeros((self._num_envs, self._body_action_dim), device=self._device)
        self._body_trajectory_planner = TrajectoryPlanner(num_envs=self._num_envs, action_dim=self._body_action_dim, device=self._device)
        self._body_trajectory_planner.setInitialPosition(self._all_env_idx, self._init_body_state[self._all_env_idx])
        self._body_trajectory_planner.setFinalPosition(self._all_env_idx, self._final_body_state[self._all_env_idx])
        self._body_trajectory_planner.setDuration(self._all_env_idx, self._action_duration[self._all_env_idx])

        # leg or eef position trajectory
        self._init_footeef_state = torch.zeros((self._num_envs, self._footeef_action_dim), device=self._device)
        self._final_footeef_state = torch.zeros((self._num_envs, self._footeef_action_dim), device=self._device)
        self._footeef_trajectory_planner = TrajectoryPlanner(num_envs=self._num_envs, action_dim=self._footeef_action_dim, device=self._device)
        self._footeef_trajectory_planner.setInitialPosition(self._all_env_idx, self._init_footeef_state[self._all_env_idx])
        self._footeef_trajectory_planner.setFinalPosition(self._all_env_idx, self._final_footeef_state[self._all_env_idx])
        self._footeef_trajectory_planner.setDuration(self._all_env_idx, self._action_duration[self._all_env_idx])

        # eef trajectories
        self._init_eef_states = torch.zeros((self._num_envs, 2, self._eef_action_dim), device=self._device)
        self._final_eef_states = torch.zeros((self._num_envs, 2, self._eef_action_dim), device=self._device)
        self._eef_trajectory_planners = [TrajectoryPlanner(num_envs=self._num_envs, action_dim=self._eef_action_dim, device=self._device) for _ in range(2)]
        for i in range(2):
            self._eef_trajectory_planners[i].setInitialPosition(self._all_env_idx, self._init_eef_states[self._all_env_idx, i])
            self._eef_trajectory_planners[i].setFinalPosition(self._all_env_idx, self._final_eef_states[self._all_env_idx, i])
            self._eef_trajectory_planners[i].setDuration(self._all_env_idx, self._action_duration[self._all_env_idx])

    def step(self):
        self._current_phase[:] += self._dt / self._action_duration[:]
        self._current_phase[:] = torch.clip(self._current_phase[:], 0.0, 1.0)
        self._body_trajectory_planner.update(self._current_phase)
        self._footeef_trajectory_planner.update(self._current_phase)
        for i in range(2):
            self._eef_trajectory_planners[i].update(self._current_phase)
        return self._current_phase==1.0

    # only update the init_state, will not affect the final state, but will make the current phase to be 0
    def set_init_state(self, body_init_state, footeef_init_state, eef_init_states, env_ids):
        self._current_phase[env_ids] = .0
        if body_init_state is not None:
            self._init_body_state[env_ids] = body_init_state[env_ids]
            self._body_trajectory_planner.setInitialPosition(env_ids, self._init_body_state[env_ids])
        if footeef_init_state is not None:
            self._init_footeef_state[env_ids] = footeef_init_state[env_ids]
            self._footeef_trajectory_planner.setInitialPosition(env_ids, self._init_footeef_state[env_ids])
        if eef_init_states is not None:
            for i in range(2):
                self._init_eef_states[env_ids, i] = eef_init_states[env_ids, i]
                self._eef_trajectory_planners[i].setInitialPosition(env_ids, self._init_eef_states[env_ids, i])
        # print('-----------------set_init_state-----------------')
        # if body_init_state is not None:
        #     print('body_init_state: ', body_init_state)
        # if footeef_init_state is not None:
        #     print('footeef_init_state: ', footeef_init_state)
        # if eef_init_states is not None:
        #     print('eef_init_states: ', eef_init_states)


    def set_final_state(self, body_final_state, footeef_final_state, eef_final_states, action_duration, env_ids):
        # print('-----------------set initila and final states-----------------')
        # print('current_phase: ', self._current_phase)

        # update the action duration and current phase
        self._action_duration[env_ids] = action_duration[env_ids]
        self._action_duration[self._action_duration==0] = 1.0
        self._current_phase[env_ids] = .0

        # update the start and end state of the body trajectory
        self._init_body_state[env_ids] = self._body_trajectory_planner.getPosition()[env_ids]
        self._final_body_state[env_ids] = self._init_body_state[env_ids] if body_final_state is None else body_final_state[env_ids]
        self._body_trajectory_planner.setInitialPosition(env_ids, self._init_body_state[env_ids])
        self._body_trajectory_planner.setFinalPosition(env_ids, self._final_body_state[env_ids])
        self._body_trajectory_planner.setDuration(env_ids, self._action_duration[env_ids])

        # update the start and end state of the footeef trajectory
        self._init_footeef_state[env_ids] = self._footeef_trajectory_planner.getPosition()[env_ids]
        self._final_footeef_state[env_ids] = self._init_footeef_state[env_ids] if footeef_final_state is None else footeef_final_state[env_ids]
        self._footeef_trajectory_planner.setInitialPosition(env_ids, self._init_footeef_state[env_ids])
        self._footeef_trajectory_planner.setFinalPosition(env_ids, self._final_footeef_state[env_ids])
        self._footeef_trajectory_planner.setDuration(env_ids, self._action_duration[env_ids])

        # update the start and end state of the eef trajectories
        for i in range(2):
            self._init_eef_states[env_ids, i] = self._eef_trajectory_planners[i].getPosition()[env_ids]
            self._final_eef_states[env_ids, i] = self._init_eef_states[env_ids, i] if eef_final_states is None else eef_final_states[env_ids, 3*i:3*i+3]
            self._eef_trajectory_planners[i].setInitialPosition(env_ids, self._init_eef_states[env_ids, i])
            self._eef_trajectory_planners[i].setFinalPosition(env_ids, self._final_eef_states[env_ids, i])
            self._eef_trajectory_planners[i].setDuration(env_ids, self._action_duration[env_ids])

        # print('-----------------set initila and final states-----------------')
        # print('body_init_state: ', self._init_body_state)
        # print('body_final_state: ', self._final_body_state)
        # print('footeef_init_state: ', self._init_footeef_state)
        # print('footeef_final_state: ', self._final_footeef_state)
        # print('eef_init_states: ', self._init_eef_states)
        # print('eef_final_states: ', self._final_eef_states)

    def get_desired_body_pva(self):
        return torch.cat((self._body_trajectory_planner._p, self._body_trajectory_planner._v, self._body_trajectory_planner._a), dim=1)

    def get_desired_footeef_pva(self):
        return torch.cat((self._footeef_trajectory_planner._p, self._footeef_trajectory_planner._v, self._footeef_trajectory_planner._a), dim=1)

    def get_desired_eef_pva(self):
        pva = []
        for i in range(2):
            pva.append(torch.cat((self._eef_trajectory_planners[i]._p, self._eef_trajectory_planners[i]._v, self._eef_trajectory_planners[i]._a), dim=1))
        return torch.stack(pva, dim=1)

    def reset(self, body_state=None, action_duration=None, env_ids=None, footeef_state=None):
        if env_ids is None:
            return
        if action_duration is None:
            action_duration = torch.ones_like(env_ids, device=self._device)
        self._action_duration[env_ids] = action_duration[env_ids]
        self._current_phase[env_ids] = .0

        if body_state is None:
            body_state = torch.zeros((self._num_envs, self._body_action_dim), device=self._device)
        self._init_body_state[env_ids] = body_state[env_ids]
        self._final_body_state[env_ids] = body_state[env_ids]
        self._body_trajectory_planner.setInitialPosition(env_ids, self._init_body_state[env_ids])
        self._body_trajectory_planner.setFinalPosition(env_ids, self._final_body_state[env_ids])
        self._body_trajectory_planner.setDuration(env_ids, self._action_duration[env_ids])

        if footeef_state is None:
            footeef_state = torch.zeros((self._num_envs, self._footeef_action_dim), device=self._device)
        self._init_footeef_state[env_ids] = footeef_state[env_ids]
        self._final_footeef_state[env_ids] = footeef_state[env_ids]
        self._footeef_trajectory_planner.setInitialPosition(env_ids, self._init_footeef_state[env_ids])
        self._footeef_trajectory_planner.setFinalPosition(env_ids, self._final_footeef_state[env_ids])
        self._footeef_trajectory_planner.setDuration(env_ids, self._action_duration[env_ids])


