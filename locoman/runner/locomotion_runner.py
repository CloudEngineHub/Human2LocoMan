from config.config import Cfg
from typing import List
from planner.gait_planner import GaitPlanner, LegState
from planner.raibert_swing_leg_planner import RaibertSwingLegPlanner
from controller.whole_body_controller import WholeBodyController
import torch
import numpy as np
from robot.motors import MotorCommand
import time
from planner.trajectory_planner import TrajectoryPlanner
from utilities.rotation_utils import rpy_to_rot_mat, rot_mat_to_rpy
import rospy
from fsm.finite_state_machine import FSM_State


class LocoMotionRunner:
    def __init__(self, cfg: Cfg = None):
        self._cfg = cfg
        # initialize robot
        if self._cfg.sim.use_real_robot:
            from robot.real_robot import RealRobot
            # from robot.real_robot_single_thread import RealRobot
            self._robot = RealRobot(self._cfg)
        else:
            self._sim_conf = self._cfg.get_sim_config()
            self._cfg.sim.sim_device = self._sim_conf.sim_device
            self._sim, self._viewer = self._create_sim()
            from robot.sim_robot import SimRobot
            self._robot = SimRobot(self._cfg, self._sim, self._viewer)
        self._robot._pre_fsm_state = self._robot._cur_fsm_state
        self._robot._cur_fsm_state = FSM_State.LOCOMOTION
        self._visualize_target = not self._cfg.sim.use_real_robot

        # inputs for wbc
        self._action_mode = 3
        self._desired_body_pva = np.zeros(18)
        self._desired_body_pva[0:6] = self._cfg.locomotion.desired_pose
        self._desired_body_pva[6:12] = self._cfg.locomotion.desired_velocity
        self._desired_foot_pva_w = np.zeros((3, 12))
        self._contact_state = np.ones(4, dtype=bool)
        self._contact_state_torch = torch.ones((self._robot._num_envs, 4), dtype=torch.bool, device=self._robot._device, requires_grad=False)

        # planner, controller
        self._gait_generator_list: List[GaitPlanner] = []
        self._swing_leg_controller_list: List[RaibertSwingLegPlanner] = []
        self._wbc_list: List[WholeBodyController] = []
        foot_landing_clearance = self._cfg.locomotion.foot_landing_clearance_real if self._cfg.sim.use_real_robot else self._cfg.locomotion.foot_landing_clearance_sim
        for i in range(self._robot._num_envs):
            self._gait_generator_list.append(GaitPlanner(self._robot, i))
            # self._swing_leg_controller_list.append(RaibertSwingLegPlanner(self._robot, i, self._gait_generator_list[i], self._desired_body_pva[6:8], self._desired_body_pva[8:10], self._desired_body_pva[11]))
            self._swing_leg_controller_list.append(RaibertSwingLegPlanner(self._robot, i, self._gait_generator_list[i], self._desired_body_pva[6:8], self._desired_body_pva[11], foot_landing_clearance=foot_landing_clearance, foot_height=self._cfg.locomotion.foot_height))
            self._wbc_list.append(WholeBodyController(self._robot, i))

        # initialize action
        self._contact_state_idx = torch.zeros((self._robot._num_envs, self._robot._num_joints), dtype=torch.bool, device=self._robot._device, requires_grad=False)
        self._desired_joint_pos = torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device, requires_grad=False)
        self._desired_joint_vel = torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device, requires_grad=False)
        self._desired_joint_torque = torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device, requires_grad=False)
        self._kps = self._robot._motors.kps_stance_mani.clone()
        self._kds = self._robot._motors.kds_stance_mani.clone()
        self._action = MotorCommand(desired_position=torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device),
                        kp=self._kps,
                        desired_velocity=torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device),
                        kd=self._kds,
                        desired_extra_torque=torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device))

        # add error detection
        self._overload_occured = False
        self._overload_threshold = 0.4
        self._reset_action_mode = torch.ones(self._robot._num_envs, dtype=torch.long, device=self._robot._device, requires_grad=False)
        self._last_body_pva = torch.zeros((self._robot._num_envs, 18), device=self._robot._device, requires_grad=False)
        self._last_foot_pva = torch.zeros((self._robot._num_envs, 3, 12), device=self._robot._device, requires_grad=False)
        self._last_contact_state = torch.zeros((self._robot._num_envs, 4), dtype=torch.bool, device=self._robot._device, requires_grad=False)

        # eef trajectories
        self._desired_eef_pva = np.zeros((2, 9))
        if self._robot._use_gripper:
            self._desired_eef_pva[0, 0:3] = self._cfg.gripper.reset_pos_sim[0:3]
            self._desired_eef_pva[1, 0:3] = self._cfg.gripper.reset_pos_sim[4:7]
            self._init_eef_states = torch.zeros((self._robot._num_envs, 2, 3), device=self._robot._device)
            self._final_eef_states = torch.zeros((self._robot._num_envs, 2, 3), device=self._robot._device)
            self._final_eef_states[:] = (self._robot._joint_init_pos[self._robot._gripper_joint_idx]).reshape(2, -1)
            self._eef_reset_planners = [TrajectoryPlanner(num_envs=self._robot._num_envs, action_dim=3, device=self._robot._device) for _ in range(2)]
            self._eef_reset_time = torch.ones(self._robot._num_envs, dtype=torch.float, device=self._robot._device, requires_grad=False) * self._cfg.gripper.reset_time

        # reset the robot
        self._robot.reset()
        for i in range(self._robot._num_envs):
            self._gait_generator_list[i].reset()
            self._swing_leg_controller_list[i].reset()

        # update robot model for getting the eef state
        # self.update_wbc_state_and_robot_eef_state()


    def _create_sim(self):
        from isaacgym import gymapi, gymutil
        self._gym = gymapi.acquire_gym()
        _, sim_device_id = gymutil.parse_device_str(self._sim_conf.sim_device)
        if self._cfg.sim.show_gui:
            graphics_device_id = sim_device_id
        else:
            graphics_device_id = -1

        sim = self._gym.create_sim(sim_device_id, graphics_device_id,
                            self._sim_conf.physics_engine, self._sim_conf.sim_params)
        if self._cfg.sim.show_gui:
            viewer = self._gym.create_viewer(sim, gymapi.CameraProperties())
            self._gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
            self._gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_viewer_sync")
        else:
            viewer = None
        self._gym.add_ground(sim, self._sim_conf.plane_params)
        return sim, viewer

    def step(self):
        for i in range(self._robot._num_envs):
            self._gait_generator_list[i].update()
            self._swing_leg_controller_list[i].update()

            self._contact_state[:] = np.array([state in (LegState.STANCE, LegState.EARLY_CONTACT, LegState.LOSE_CONTACT) for state in self._gait_generator_list[i].leg_state])
            # self._desired_body_pva[0:3] = self._robot.base_pos_w_np[i]
            # self._desired_body_pva[3:6] = self._robot.base_rpy_w2b_np[i]
            self._desired_foot_pva_w[0, :] = self._swing_leg_controller_list[i].get_desired_foot_positions().flatten()

            self._contact_state_torch[:] = torch.tensor(self._contact_state, dtype=torch.bool, device=self._robot._device, requires_grad=False)
            self._robot.set_desired_foot_contact(self._contact_state_torch)
            self._contact_state_idx[:, :6] = self._contact_state_torch[:, 0:1].repeat(1, 6)
            self._contact_state_idx[:, 6:12] = self._contact_state_torch[:, 1:2].repeat(1, 6)
            if self._robot._use_gripper:
                self._contact_state_idx[:, 12:15] = self._contact_state_torch[:, 2:3].repeat(1, 3)
                self._contact_state_idx[:, 15:18] = self._contact_state_torch[:, 3:].repeat(1, 3)


            # if self._robot._log_info_now:
            #     print('contact_state: ', self._contact_state)
            #     print('action_mode: ', self._action_mode)
            #     print('desired_body_pva: ', self._desired_body_pva)
            #     print('desired_foot_pva_w: ', self._desired_foot_pva_w)
            #     print('desired_eef_pva: ', self._desired_eef_pva)

            command_executed, desired_q, desired_dq, motor_torques = self._wbc_list[i].update(self._action_mode, self._desired_body_pva, self._contact_state, self._desired_foot_pva_w, self._desired_eef_pva)
            self._desired_joint_pos[i] = torch.tensor(desired_q, device=self._robot._device, requires_grad=False)
            self._desired_joint_vel[i] = torch.tensor(desired_dq, device=self._robot._device, requires_grad=False)
            self._desired_joint_torque[i] = torch.tensor(motor_torques, device=self._robot._device, requires_grad=False)

            # # if max(abs(desired_q - self._robot._joint_pos[i].cpu().numpy())) > self._overload_threshold or self._action_planner._current_action_idx[i] == 3:
            # if max(abs(desired_q - self._robot._joint_pos[i].cpu().numpy())) > self._overload_threshold:
            #     self._last_contact_state[i] = self._contact_state_torch[i]
            #     self._last_body_pva[i, 0:6] = torch.tensor(self._desired_body_pva[0:6], device=self._robot._device, requires_grad=False)
            #     self._last_foot_pva[i, 0, self._action_planner._manipulate_dofs] = self._robot.foot_pos_w[i, contact_state[i]==0, 0:3]
            #     self._init_eef_states[i] = self._robot.joint_pos[i, self._robot._gripper_joint_idx].reshape(2, -1)
            #     print('-----------------reset-----------------')
            #     print('init_eef_states: ', self._init_eef_states)
            #     print('final_eef_states: ', self._final_eef_states)
            #     for k in range(2):
            #         self._eef_reset_planners[k].setInitialPosition(i, self._init_eef_states[i, k])
            #         self._eef_reset_planners[k].setFinalPosition(i, self._final_eef_states[i, k])
            #         self._eef_reset_planners[k].setDuration(i, self._eef_reset_time)
            #     self.run_stablization()

        # if self._visualize_target:
        #     self.visualize_target_recording([self._action_mode for _ in range(self._robot._num_envs)])
            
        self.construct_apply_action()
        # self.update_wbc_state_and_robot_eef_state()


    def run_stablization(self):
        phase = torch.zeros(self._robot._num_envs, dtype=torch.float, device=self._robot._device, requires_grad=False)
        while not rospy.is_shutdown():
            phase += self._robot._dt / self._eef_reset_time
            phase = torch.clip(phase, 0.0, 1.0)
            pva = []
            for j in range(2):
                self._eef_reset_planners[j].update(phase)
                pva.append(torch.cat((self._eef_reset_planners[j]._p, self._eef_reset_planners[j]._v, self._eef_reset_planners[j]._a), dim=1))
            footeef_pva = torch.stack(pva, dim=1)

            self._robot.set_desired_foot_contact(self._last_contact_state)
            self._contact_state_idx[:, :6] = self._last_contact_state[:, 0:1].repeat(1, 6)
            self._contact_state_idx[:, 6:12] = self._last_contact_state[:, 1:2].repeat(1, 6)
            self._contact_state_idx[:, 12:15] = self._last_contact_state[:, 2:3].repeat(1, 3)
            self._contact_state_idx[:, 15:18] = self._last_contact_state[:, 3:].repeat(1, 3)

            if self._visualize_target:
                self.visualize_target_recording(self._reset_action_mode)

            for i in range(self._robot._num_envs):
                command_executed, desired_q, desired_dq, motor_torques = self._wbc_list[i].update(self._reset_action_mode[i], self._last_body_pva[i].cpu().numpy(), self._last_contact_state[i].cpu().numpy(), self._last_foot_pva[i].cpu().numpy(), footeef_pva[i].cpu().numpy())
                self._desired_joint_pos[i] = torch.tensor(desired_q, device=self._robot._device, requires_grad=False)
                self._desired_joint_vel[i] = torch.tensor(desired_dq, device=self._robot._device, requires_grad=False)
                self._desired_joint_torque[i] = torch.tensor(motor_torques, device=self._robot._device, requires_grad=False)
            
            self.construct_apply_action()
            # self.update_wbc_state_and_robot_eef_state()
        quit()


    def construct_apply_action(self):
        # update the kp and kd based on whether the foot is in contact
        self._kps[self._contact_state_idx] = self._robot._motors.kps_stance_loco[self._contact_state_idx] if self._action_mode == 3 else self._robot._motors.kps_stance_mani[self._contact_state_idx]
        self._kps[~self._contact_state_idx] = self._robot._motors.kps_swing_loco[~self._contact_state_idx] if self._action_mode == 3 else self._robot._motors.kps_swing_mani[~self._contact_state_idx]
        self._kds[self._contact_state_idx] = self._robot._motors.kds_stance_loco[self._contact_state_idx] if self._action_mode == 3 else self._robot._motors.kds_stance_mani[self._contact_state_idx]
        self._kds[~self._contact_state_idx] = self._robot._motors.kds_swing_loco[~self._contact_state_idx] if self._action_mode == 3 else self._robot._motors.kds_swing_mani[~self._contact_state_idx]
        # print('kps: ', self._kps)
        # print('kds: ', self._kds)

        # self._kps[self._contact_state_idx] = 30#self._robot._motors.kps_stance[self._contact_state_idx]
        # self._kps[~self._contact_state_idx] = 30#self._robot._motors.kps_swing[~self._contact_state_idx]
        # self._kds[self._contact_state_idx] = 1#self._robot._motors.kds_stance[self._contact_state_idx]
        # self._kds[~self._contact_state_idx] =1#self._robot._motors.kds_swing[~self._contact_state_idx]


        # # update the action
        # if max(abs(self._desired_joint_pos - self._robot.joint_pos).flatten()) < self._overload_threshold:
        #     self._action.desired_position[:] = self._desired_joint_pos
        #     self._action.desired_velocity[:] = self._desired_joint_vel
        #     self._action.desired_extra_torque[:] = self._desired_joint_torque
        #     self._action.kp[:] = self._kps
        #     self._action.kd[:] = self._kds

        self._action.desired_position[:] = self._desired_joint_pos
        self._action.desired_velocity[:] = self._desired_joint_vel
        self._action.desired_extra_torque[:] = self._desired_joint_torque
        self._action.kp[:] = self._kps
        self._action.kd[:] = self._kds

        self._robot.step(self._action, gripper_cmd=True)

    def update_wbc_state_and_robot_eef_state(self):
        if self._robot._use_gripper:
            for i in range(self._robot._num_envs):
                self._wbc_list[i].update_robot_state()
                for idx, eef_frame_id in enumerate(self._wbc_list[i]._eef_frame_ids):
                    self._robot._eef_pos_w[i, idx] = torch.tensor(self._wbc_list[i]._robot_model.framePlacement(None, eef_frame_id, update_kinematics=False).translation, device=self._robot._device, requires_grad=False)
                    self._robot._eef_rot_w[i, idx] = torch.tensor(self._wbc_list[i]._robot_model.framePlacement(None, eef_frame_id, update_kinematics=False).rotation, device=self._robot._device, requires_grad=False)
            self._robot._eef_rpy_w[:] = rot_mat_to_rpy(self._robot._eef_rot_w.reshape(-1, 3, 3)).reshape(self._robot._num_envs, -1, 3)
        else:
            for i in range(self._robot._num_envs):
                self._wbc_list[i].update_robot_state()

    def visualize_target_recording(self, action_mode):
        from isaacgym import gymapi, gymutil
        self._gym.clear_lines(self._viewer)

        # show the target foot position for locomotion, show the target eef 6d pose for manipulation

        if 1 in action_mode:
            target_sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, 0, 0)), color=(1, 0, 0))
            # target_sphere_geom = gymutil.WireframeSphereGeometry(0.001, 1, 1, gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, 0, 0)), color=(1, 0, 0))
            current_sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, 0, 0)), color=(0, 0, 1))
        if 2 in action_mode:
            target_axes_geom = gymutil.AxesGeometry(.1, gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, 0, 0)))
            current_axes_geom = gymutil.AxesGeometry(.1, gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, 0, 0)))

        for i in range(self._robot._num_envs):
            if action_mode[i] == 1:
                target_foot_pose = gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, 0, 0))
                target_foot_pos_sim = self._robot._state_estimator[self._robot._cur_fsm_state]._world_rot_mat_w2sim[i].transpose(-2, -1) @ (self._action_planner._body_leg_eefs_planner._final_footeef_state[i, 0:3].unsqueeze(-1)) + self._robot._state_estimator[self._robot._cur_fsm_state]._world_pos_sim[i].unsqueeze(-1)
                target_foot_pose.p = gymapi.Vec3(target_foot_pos_sim[0], target_foot_pos_sim[1], target_foot_pos_sim[2])
                gymutil.draw_lines(target_sphere_geom, self._gym, self._viewer, self._robot._envs[i], target_foot_pose)

                current_foot_pose = gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, 0, 0))
                current_foot_pose.p = gymapi.Vec3(self._robot._foot_pos_sim[i, self._action_planner._manipulate_leg_idx, 0],
                                                  self._robot._foot_pos_sim[i, self._action_planner._manipulate_leg_idx, 1],
                                                  self._robot._foot_pos_sim[i, self._action_planner._manipulate_leg_idx, 2])
                gymutil.draw_lines(current_sphere_geom, self._gym, self._viewer, self._robot._envs[i], current_foot_pose)
            elif action_mode[i] == 2:
                target_eef_pose = gymapi.Transform()
                target_eef_pos_sim = self._robot._state_estimator[self._robot._cur_fsm_state]._world_rot_mat_w2sim[i].transpose(-2, -1) @ (self._action_planner._body_leg_eefs_planner._final_footeef_state[i, 0:3].unsqueeze(-1)) + self._robot._state_estimator[self._robot._cur_fsm_state]._world_pos_sim[i].unsqueeze(-1)
                target_eef_pose.p = gymapi.Vec3(target_eef_pos_sim[0], target_eef_pos_sim[1], target_eef_pos_sim[2])
                target_eef_rpy_sim = rot_mat_to_rpy(self._robot._state_estimator[self._robot._cur_fsm_state]._world_rot_mat_w2sim[i].unsqueeze(0).transpose(-2, -1) @ rpy_to_rot_mat(self._action_planner._body_leg_eefs_planner._final_eef_states[i, self._action_planner._manipulate_leg_idx].unsqueeze(0))).squeeze(0)
                target_eef_pose.r = gymapi.Quat.from_euler_zyx(target_eef_rpy_sim[0], target_eef_rpy_sim[1], target_eef_rpy_sim[2])
                gymutil.draw_lines(target_axes_geom, self._gym, self._viewer, self._robot._envs[i], target_eef_pose)

                current_eef_pose = gymapi.Transform()
                current_eef_pose.p = gymapi.Vec3(self._robot._eef_pos_sim[i, self._action_planner._manipulate_leg_idx, 0],
                                                self._robot._eef_pos_sim[i, self._action_planner._manipulate_leg_idx, 1],
                                                self._robot._eef_pos_sim[i, self._action_planner._manipulate_leg_idx, 2])
                current_eef_pose.r = gymapi.Quat(self._robot._eef_quat_sim[i, self._action_planner._manipulate_leg_idx, 0],
                                                 self._robot._eef_quat_sim[i, self._action_planner._manipulate_leg_idx, 1],
                                                self._robot._eef_quat_sim[i, self._action_planner._manipulate_leg_idx, 2],
                                                self._robot._eef_quat_sim[i, self._action_planner._manipulate_leg_idx, 3])
                gymutil.draw_lines(current_axes_geom, self._gym, self._viewer, self._robot._envs[i], current_eef_pose)
                


        # sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        # sphere_pose = gymapi.Transform(r=sphere_rot)
        # sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 0, 0))
        # temp_sphere_pose = gymapi.Transform(r=sphere_rot)
        # temp_sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, temp_sphere_pose, color=(0, 1, 0))
        # last_sphere_pose = gymapi.Transform(r=sphere_rot)
        # last_sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, last_sphere_pose, color=(0, 0, 1))

        # target_rot = gymapi.Quat.from_euler_zyx(self._action_planner._body_leg_eefs_planner._final_eef_states[i, ])

        # target_6d_pose = gymutil.AxesGeometry
        # for i in range(self._robot._num_envs):


        #     target_sphere_pose = gymapi.Transform()
        #     target_sphere_pose.p = gymapi.Vec3(self.target_position_global[i, 0],
        #                                        self.target_position_global[i, 1],
        #                                        self.target_position_global[i, 2])
        #     gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], target_sphere_pose)
        #     temp_target_sphere_pose = gymapi.Transform()
        #     temp_target_sphere_pose.p = gymapi.Vec3(self.temp_target_position_global[i, 0],
        #                                             self.temp_target_position_global[i, 1],
        #                                             self.temp_target_position_global[i, 2])
        #     last_target_sphere_pose = gymapi.Transform()
        #     last_target_sphere_pose.p = gymapi.Vec3(self.last_target_position_global[i, 0],
        #                                             self.last_target_position_global[i, 1],
        #                                             self.last_target_position_global[i, 2])
        #     gymutil.draw_lines(temp_sphere_geom, self.gym, self.viewer, self.envs[i], temp_target_sphere_pose)
        #     gymutil.draw_lines(last_sphere_geom, self.gym, self.viewer, self.envs[i], last_target_sphere_pose)


    def step_action(self, action: MotorCommand):
        for i in range(self._robot._num_envs):
            self._gait_generator_list[i].update()
            self._swing_leg_controller_list[i].update()

            self._contact_state[:] = np.array([state in (LegState.STANCE, LegState.EARLY_CONTACT, LegState.LOSE_CONTACT) for state in self._gait_generator_list[i].leg_state])
            self._desired_foot_pva_w[0, :] = self._swing_leg_controller_list[i].get_desired_foot_positions().flatten()

            self._contact_state_torch[:] = torch.tensor(self._contact_state, dtype=torch.bool, device=self._robot._device, requires_grad=False)
            self._robot.set_desired_foot_contact(self._contact_state_torch)
            self._contact_state_idx[:, :6] = self._contact_state_torch[:, 0:1].repeat(1, 6)
            self._contact_state_idx[:, 6:12] = self._contact_state_torch[:, 1:2].repeat(1, 6)
            if self._robot._use_gripper:
                self._contact_state_idx[:, 12:15] = self._contact_state_torch[:, 2:3].repeat(1, 3)
                self._contact_state_idx[:, 15:18] = self._contact_state_torch[:, 3:].repeat(1, 3)

            command_executed, desired_q, desired_dq, motor_torques = self._wbc_list[i].update(self._action_mode, self._desired_body_pva, self._contact_state, self._desired_foot_pva_w, self._desired_eef_pva)
            self._desired_joint_pos[i] = torch.tensor(desired_q, device=self._robot._device, requires_grad=False)
            self._desired_joint_vel[i] = torch.tensor(desired_dq, device=self._robot._device, requires_grad=False)
            self._desired_joint_torque[i] = torch.tensor(motor_torques, device=self._robot._device, requires_grad=False)

        self._robot.step(action, gripper_cmd=True)
        # self.update_wbc_state_and_robot_eef_state()

