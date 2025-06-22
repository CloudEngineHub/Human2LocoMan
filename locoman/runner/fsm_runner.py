import time
from matplotlib import pyplot as plt
from config.config import Cfg
from typing import List
from controller.whole_body_controller import WholeBodyController
# from controller.whole_body_controller_follow import WholeBodyController
import torch
import numpy as np
from robot.motors import MotorCommand, MotorControlMode
from planner.fsm_switcher import FSMSwitcher
from planner.fsm_resetter import FSMResetter
from planner.bimanual_resetter import BimanualResetter
from commander.fsm_commander import FSMCommander
from planner.trajectory_planner import TrajectoryPlanner
from utilities.rotation_utils import rpy_to_rot_mat, rot_mat_to_rpy
import rospy
from std_msgs.msg import Int32
from fsm.finite_state_machine import FSM_State, FSM_OperatingMode, fsm_command_to_fsm_state_and_manipulate_mode
from commander.bi_manipulation_commander import BiManipulationCommander
from planner.bimanipulation_switcher import BimanipulationSwitcher
from commander.eef_manipulate_commander import EEFManipulateCommander
from commander.foot_manipulate_commander import FootManipulateCommander
from planner.stance_resetter import StanceResetter

class FSMRunner:
    def __init__(self, cfg: Cfg = None):
        self._cfg = cfg
        if self._cfg.sim.use_real_robot:
            from robot.real_robot import RealRobot
            self._robot = RealRobot(self._cfg)
        else:
            self._sim_conf = self._cfg.get_sim_config()
            self._cfg.sim.sim_device = self._sim_conf.sim_device
            self._sim, self._viewer = self._create_sim()
            from robot.sim_robot_tv import SimRobot
            self._robot = SimRobot(self._cfg, self._sim, self._viewer)
        self._visualize_target = not self._cfg.sim.use_real_robot

        # inputs for wbc
        self._action_mode = 0
        # the position, velocity, and acceleration of the body (torso) 6d pose: [6 * 3]
        self._desired_body_pva = np.zeros(18)
        self._desired_body_pva[0:6] = self._cfg.locomotion.desired_pose
        self._desired_body_pva[6:12] = self._cfg.locomotion.desired_velocity
        # 3 = pva, 12 = 6d pose * 2 eefs?
        self._desired_footeef_pva_w = np.zeros((3, 12))
        # 2 eefs, only 3d pose (orientation?)
        self._desired_eef_pva = np.zeros((2, 9))
        self._pre_contact_state = np.ones(4, dtype=bool)
        self._contact_state = np.ones(4, dtype=bool)
        self._contact_state_torch = torch.ones((self._robot._num_envs, 4), dtype=torch.bool, device=self._robot._device, requires_grad=False)

        # planner, controller
        self._fsm_switcher = FSMSwitcher(self, 0)
        self._fsm_resetter = FSMResetter(self, 0)
        self._fsm_commander = FSMCommander(self, 0)
        self._command_generator = None
        self._wbc_list: List[WholeBodyController] = []
        for i in range(self._robot._num_envs):
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

        # for bimanipulation
        self._bimanual_action = MotorCommand(desired_position=torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device),
                        kp=self._kps,
                        desired_velocity=torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device),
                        kd=self._kds,
                        desired_extra_torque=torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device))
        if self._cfg.sim.use_real_robot:
            self._kp_bimanual_switch = torch.tensor(self._cfg.motor_controller.sim.kp_bimanual_switch, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._kd_bimanual_switch = torch.tensor(self._cfg.motor_controller.sim.kd_bimanual_switch, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._kp_bimanual_command = torch.tensor(self._cfg.motor_controller.sim.kp_bimanual_command, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._kd_bimanual_command = torch.tensor(self._cfg.motor_controller.sim.kd_bimanual_command, device=self._robot._device, dtype=torch.float, requires_grad=False)
        else:
            self._kp_bimanual_switch = torch.tensor(self._cfg.motor_controller.real.kp_bimanual_switch, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._kd_bimanual_switch = torch.tensor(self._cfg.motor_controller.real.kd_bimanual_switch, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._kp_bimanual_command = torch.tensor(self._cfg.motor_controller.real.kp_bimanual_command, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._kd_bimanual_command = torch.tensor(self._cfg.motor_controller.real.kd_bimanual_command, device=self._robot._device, dtype=torch.float, requires_grad=False)

        # if self._robot._use_gripper:
        #     self._bimanual_action.desired_position[:, self._robot._gripper_joint_idx] = self._robot._joint_init_pos[self._robot._gripper_joint_idx]
        #     self._bimanual_action.desired_velocity[:, self._robot._gripper_joint_idx] = 0
        #     self._bimanual_action.desired_extra_torque[:, self._robot._gripper_joint_idx] = 0

        # add error detection
        self._overload_occured = False
        self._overload_threshold = 0.4
        self._reset_action_mode = torch.ones(self._robot._num_envs, dtype=torch.long, device=self._robot._device, requires_grad=False)
        self._last_body_pva = torch.zeros((self._robot._num_envs, 18), device=self._robot._device, requires_grad=False)
        self._last_foot_pva = torch.zeros((self._robot._num_envs, 3, 12), device=self._robot._device, requires_grad=False)
        self._last_contact_state = torch.zeros((self._robot._num_envs, 4), dtype=torch.bool, device=self._robot._device, requires_grad=False)
        if self._robot._use_gripper:
            self._init_eef_states = torch.zeros((self._robot._num_envs, 2, 3), device=self._robot._device)
            self._final_eef_states = torch.zeros((self._robot._num_envs, 2, 3), device=self._robot._device)
            self._final_eef_states[:] = (self._robot._joint_init_pos[self._robot._gripper_joint_idx]).reshape(2, -1)
            self._eef_reset_planners = [TrajectoryPlanner(num_envs=self._robot._num_envs, action_dim=3, device=self._robot._device) for _ in range(2)]
            self._eef_reset_time = torch.ones(self._robot._num_envs, dtype=torch.float, device=self._robot._device, requires_grad=False) * self._cfg.gripper.reset_time

        # fsm buffers
        self._fsm_state_buffer = self._robot._cur_fsm_state
        self._manipulate_mode_buffer = self._robot._cur_manipulate_mode
        self._fsm_command_sub = rospy.Subscriber(self._cfg.fsm_switcher.fsm_state_topic, Int32, self._fsm_command_callback)

        # reset
        self._robot.reset()
        self._command_generator = self._fsm_commander.get_command_generator()
        self._command_generator.use_commander()
        self._command_generator.reset()

        # self._fsm_state_buffer = FSM_State.LOCOMOTION  # for testing locomotion mode

        self._fsm_reset_sub = rospy.Subscriber(self._cfg.teleoperation.robot_reset_topic, Int32, self._fsm_reset_callback)
        self._fsm_reset = False
        
        self._ready_signal_publisher = rospy.Publisher(self._cfg.teleoperation.receive_action_topic, Int32, queue_size=1)
        self.checkpt = 0

        self._bimanual_init_q = np.zeros(self._robot._num_joints)
        
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
        # fps = 1 / (time.time() - self.checkpt)
        # print('fps', fps)
        # self.checkpt = time.time()
        if type(self._command_generator) in [BiManipulationCommander, BimanipulationSwitcher, BimanualResetter]:
            # bimanual
            if type(self._command_generator) is BimanipulationSwitcher:
                desired_q, desired_dq, desired_torque = self._command_generator.compute_motor_command()
                for i in range(self._robot._num_envs):
                    self._bimanual_action.kp[i, self._robot._dog_joint_idx] = self._kp_bimanual_switch
                    self._bimanual_action.kd[i, self._robot._dog_joint_idx] = self._kd_bimanual_switch
            else:
                command = self._command_generator.compute_command_for_wbc()
                if self._robot._use_gripper:
                    self._robot.set_gripper_angles(torch.tensor(command["gripper_angles"], device=self._robot._device, requires_grad=False))
                for i in range(self._robot._num_envs):
                    self._wbc_list[i].update_robot_model()
                    command_executed, desired_q, desired_dq, desired_torque = self._wbc_list[i].compute_action_from_command(command)
                    desired_q[-6:] = command["rear_legs_command"]["pos"]
                    desired_dq[-6:] = command["rear_legs_command"]["vel"]
                    desired_torque[-6:] = command["rear_legs_command"]["torque"]
                    self._bimanual_action.kp[i, self._robot._dog_joint_idx] = self._kp_bimanual_command
                    self._bimanual_action.kd[i, self._robot._dog_joint_idx] = self._kd_bimanual_command
                    if not command_executed:
                        safe_executed_command = self._wbc_list[i].last_executed_command
                        if self._wbc_list[i].last_executed_command['action_mode'] == 0:
                            print('bimanual initial command not executed!')
                            desired_q[:-6] = self._bimanual_init_q[:-6]
                            desired_dq[:-6] = 0
                            desired_torque[:-6] = 0
                self._command_generator.feedback(command_executed)
                if not command_executed:
                    if self._robot._fsm_operation_mode != FSM_OperatingMode.RESET:
                        self._command_generator.reset_last_command(safe_executed_command)
            self._bimanual_action.desired_position[i, :] = torch.tensor(desired_q, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._bimanual_action.desired_velocity[i, :] = torch.tensor(desired_dq, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._bimanual_action.desired_extra_torque[i, :] = torch.tensor(desired_torque, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._robot.step(self._bimanual_action, gripper_cmd=True)
        elif type(self._command_generator) is StanceResetter:
            motor_command = self._command_generator.compute_motor_command()
            if self._robot._use_gripper:
                self._robot.step(motor_command, MotorControlMode.POSITION, gripper_cmd=True)
            else:
                self._robot.step(motor_command, MotorControlMode.POSITION, gripper_cmd=False)
        else:
            command = self._command_generator.compute_command_for_wbc()
            self._update_contact_state_idx(command["contact_state"])
            if self._robot._use_gripper:
                self._robot.set_gripper_angles(torch.tensor(command["gripper_angles"], device=self._robot._device, requires_grad=False))
            for i in range(self._robot._num_envs):
                self._wbc_list[i].update_robot_model()
                command_executed, desired_q, desired_dq, motor_torques = self._wbc_list[i].compute_action_from_command(command)
                # print('command_executed: ', command_executed)
                self._desired_joint_pos[i] = torch.tensor(desired_q, device=self._robot._device, requires_grad=False)
                self._desired_joint_vel[i] = torch.tensor(desired_dq, device=self._robot._device, requires_grad=False)
                self._desired_joint_torque[i] = torch.tensor(motor_torques, device=self._robot._device, requires_grad=False)
                if not command_executed:
                    safe_executed_command = self._wbc_list[i].last_executed_command
            self._command_generator.feedback(command_executed)
            if not command_executed:
                if self._robot._fsm_operation_mode != FSM_OperatingMode.RESET:
                    self._command_generator.reset_last_command(safe_executed_command)
            self._construct_and_apply_action()
        self._check_switching()
        self._check_reset()
        # if not self._cfg.sim.use_real_robot:
        #     self.visualize_frames()

    def _update_contact_state_idx(self, contact_state):
        self._contact_state_torch[:] = torch.tensor(contact_state, dtype=torch.bool, device=self._robot._device, requires_grad=False)
        self._robot.set_desired_foot_contact(self._contact_state_torch)
        self._contact_state_idx[:, :6] = self._contact_state_torch[:, 0:1].repeat(1, 6)
        self._contact_state_idx[:, 6:12] = self._contact_state_torch[:, 1:2].repeat(1, 6)
        if self._robot._use_gripper:
            self._contact_state_idx[:, 12:15] = self._contact_state_torch[:, 2:3].repeat(1, 3)
            self._contact_state_idx[:, 15:18] = self._contact_state_torch[:, 3:].repeat(1, 3)

    def _construct_and_apply_action(self):
        # update the kp and kd based on whether the foot is in contact
        if self._robot._fsm_operation_mode == FSM_OperatingMode.NORMAL:
            self._action_mode = self._command_generator._action_mode
        self._kps[self._contact_state_idx] = self._robot._motors.kps_stance_loco[self._contact_state_idx] if self._action_mode == 3 else self._robot._motors.kps_stance_mani[self._contact_state_idx]
        self._kps[~self._contact_state_idx] = self._robot._motors.kps_swing_loco[~self._contact_state_idx] if self._action_mode == 3 else self._robot._motors.kps_swing_mani[~self._contact_state_idx]
        self._kds[self._contact_state_idx] = self._robot._motors.kds_stance_loco[self._contact_state_idx] if self._action_mode == 3 else self._robot._motors.kds_stance_mani[self._contact_state_idx]
        self._kds[~self._contact_state_idx] = self._robot._motors.kds_swing_loco[~self._contact_state_idx] if self._action_mode == 3 else self._robot._motors.kds_swing_mani[~self._contact_state_idx]

        # update the action
        self._action.desired_position[:] = self._desired_joint_pos
        self._action.desired_velocity[:] = self._desired_joint_vel
        self._action.desired_extra_torque[:] = self._desired_joint_torque
        self._action.kp[:] = self._kps
        self._action.kd[:] = self._kds
        self._action.kp = self._action.kp.to(self._robot._device)
        self._action.kd = self._action.kd.to(self._robot._device)

        try:
            ret = self._robot.step(self._action, gripper_cmd=True)
            # if (isinstance(self._command_generator, EEFManipulateCommander) or isinstance(self._command_generator, FootManipulateCommander)) \
            #     and self._command_generator._cmd_from_human is not None:
            #     self._command_generator.collect_or_rollout(ret, self._cfg.sim.use_real_robot) # TODO: add record to all command generators?
        except Exception as e:
            raise e

    def _check_switching(self):
        if self._robot._fsm_operation_mode == FSM_OperatingMode.TRANSITION:
            if self._command_generator.check_finished():
                # print(self._command_generator)
                print('-------------switch to normal mode-------------')
                if type(self._command_generator) == BimanipulationSwitcher:
                    self._bimanual_init_q = self._robot.joint_pos_np[0]
                self._switch_to_normal_mode()
        elif self._fsm_state_buffer != self._robot._cur_fsm_state or self._manipulate_mode_buffer != self._robot._cur_manipulate_mode:
            # if self._robot.finish_reset:
            print('-------------switch to transition mode-------------')
            print("previous commander: ", self._command_generator)
            print(self._robot._cur_fsm_state, " -> ", self._fsm_state_buffer)
            self._switch_to_transition_mode()

    def _fsm_command_callback(self, msg: Int32):
        if self._robot._fsm_operation_mode == FSM_OperatingMode.NORMAL:
            self._fsm_state_buffer, self._manipulate_mode_buffer = fsm_command_to_fsm_state_and_manipulate_mode(msg.data, self._robot._use_gripper)

    def _fsm_reset_callback(self, msg: Int32):
        # indicate starting to perform reset motions within a specific mode
        if self._robot._fsm_operation_mode == FSM_OperatingMode.NORMAL:
            self._fsm_reset = True
    
    def _check_reset(self):
        if self._robot._fsm_operation_mode == FSM_OperatingMode.RESET:
            if self._command_generator.check_finished():
                self._command_generator = self._fsm_commander.get_command_generator()
                self._command_generator.use_commander()
                if self._robot._cur_fsm_state is FSM_State.STANCE:
                    self._command_generator.reset(reset_estimator=True)
                else:
                    self._command_generator.reset(reset_estimator=False)
                self._robot._fsm_operation_mode = FSM_OperatingMode.NORMAL
                self._fsm_reset = False
                print('reset done')
        elif self._robot._fsm_operation_mode == FSM_OperatingMode.NORMAL:
            if self._fsm_reset:
                if self._robot._cur_fsm_state in [FSM_State.MANIPULATION, FSM_State.BIMANIPULATION, FSM_State.STANCE]:
                    self._command_generator.unuse_commander()
                    self._command_generator = self._fsm_resetter.get_resetter()
                    self._command_generator.reset()
                    self._robot._fsm_operation_mode = FSM_OperatingMode.RESET
                    print('perform reset motions within the (bi)manipulation mode')

    def _switch_to_normal_mode(self):
        self._robot._cur_fsm_state = self._robot._nex_fsm_state
        self._robot._cur_manipulate_mode = self._robot._nex_manipulate_mode
        self._command_generator = self._fsm_commander.get_command_generator()
        self._command_generator.use_commander()
        self._command_generator.reset()
        # self._ready_signal_publisher.publish(2)
        self._robot._fsm_operation_mode = FSM_OperatingMode.NORMAL

    def _switch_to_transition_mode(self):
        self._robot._fsm_operation_mode = FSM_OperatingMode.TRANSITION
        self._robot._nex_fsm_state = self._fsm_state_buffer
        self._robot._nex_manipulate_mode = self._manipulate_mode_buffer
        # switch between stance and (bi)manipulation needs complex transition actions
        if (self._robot._cur_fsm_state, self._fsm_state_buffer) in [(FSM_State.STANCE, FSM_State.MANIPULATION), (FSM_State.MANIPULATION, FSM_State.STANCE), (FSM_State.STANCE, FSM_State.BIMANIPULATION), (FSM_State.BIMANIPULATION, FSM_State.STANCE)]:
            print('switching between stance and (bi)manipulation')
            self._command_generator.unuse_commander()
            self._command_generator = self._fsm_switcher.get_switcher()
            while self._command_generator is None:
                time.sleep(0.1)
            self._command_generator.reset()
        # directly switch from stance to locomotion
        elif self._robot._cur_fsm_state == FSM_State.STANCE and self._fsm_state_buffer == FSM_State.LOCOMOTION:
            pass
        # from locomotion to stance, first slow down and then stand when all feet are in contact
        elif self._robot._cur_fsm_state in (FSM_State.LOCOMOTION, FSM_State.LOCOMANIPULATION) and self._fsm_state_buffer == FSM_State.STANCE:
            self._command_generator.prepare_to_stand()
        # switch between locomotion and (bi)manipulation needs to switch to stance first
        elif (self._robot._cur_fsm_state, self._fsm_state_buffer) in [(FSM_State.MANIPULATION, FSM_State.LOCOMOTION), (FSM_State.LOCOMOTION, FSM_State.MANIPULATION), (FSM_State.BIMANIPULATION, FSM_State.LOCOMOTION), (FSM_State.LOCOMOTION, FSM_State.BIMANIPULATION)]:
            self._fsm_state_buffer = FSM_State.STANCE
            self._robot._fsm_operation_mode = FSM_OperatingMode.NORMAL
        # switch to/from bimanipulation from/to no stance needs to switch to stance first
        elif FSM_State.BIMANIPULATION in [self._robot._cur_fsm_state, self._fsm_state_buffer] and FSM_State.STANCE not in [self._robot._cur_fsm_state, self._fsm_state_buffer]:
            self._fsm_state_buffer = FSM_State.STANCE
            self._robot._fsm_operation_mode = FSM_OperatingMode.NORMAL
        elif self._robot._cur_fsm_state == FSM_State.STANCE and self._fsm_state_buffer == FSM_State.LOCOMANIPULATION:
            print('switching between stance and loco-manipulation')
            self._command_generator.unuse_commander()
            self._command_generator = self._fsm_switcher.get_switcher()
            self._command_generator.reset()

    def visualize_frames(self):
        from isaacgym import gymapi, gymutil
        self._gym.clear_lines(self._viewer)

        world_frame = gymutil.AxesGeometry(.15, gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, 0, 0)))
        body_frame = gymutil.AxesGeometry(.15, gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, 0, 0)))
        # cur_eef_frame = gymutil.AxesGeometry(.5, gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, 0, 0)))
        # des_eef_frame = gymutil.AxesGeometry(.5, gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, 0, 0)))
        # foot_frame = gymutil.AxesGeometry(.5, gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, 0, 0)))

        for i in range(self._robot._num_envs):
            world_pose = gymapi.Transform()
            world_pose.p = gymapi.Vec3(self._robot._state_estimator[self._robot._cur_fsm_state]._world_pos_sim[i, 0],
                                        self._robot._state_estimator[self._robot._cur_fsm_state]._world_pos_sim[i, 1],
                                        self._robot._state_estimator[self._robot._cur_fsm_state]._world_pos_sim[i, 2])
            
            world_rpy_sim = rot_mat_to_rpy(self._robot._state_estimator[self._robot._cur_fsm_state]._world_rot_mat_w2sim.transpose(-2, -1))[0]
            world_pose.r = gymapi.Quat.from_euler_zyx(world_rpy_sim[0], world_rpy_sim[1], world_rpy_sim[2])
            gymutil.draw_lines(world_frame, self._gym, self._viewer, self._robot._envs[i], world_pose)

            body_pose = gymapi.Transform()
            body_pos_sim = self._robot._state_estimator[self._robot._cur_fsm_state]._world_pos_sim[i] +\
                            (self._robot._state_estimator[self._robot._cur_fsm_state]._world_rot_mat_w2sim[i].transpose(-2, -1) @ (self._robot._base_pos_w[i].unsqueeze(-1))).squeeze(-1)
            body_pose.p = gymapi.Vec3(body_pos_sim[0], body_pos_sim[1], body_pos_sim[2])
            body_rpy_sim = rot_mat_to_rpy((self._robot._state_estimator[self._robot._cur_fsm_state]._world_rot_mat_w2sim[i].transpose(-2, -1) @ self._robot.base_rot_mat_w2b[i]).unsqueeze(0)).squeeze(0)
            body_pose.r = gymapi.Quat.from_euler_zyx(body_rpy_sim[0], body_rpy_sim[1], body_rpy_sim[2])
            gymutil.draw_lines(body_frame, self._gym, self._viewer, self._robot._envs[i], body_pose)

            # if self._robot._use_gripper:
            #     cur_eef_pose = gymapi.Transform()
            #     cur_eef_pos_sim = self._robot._state_estimator[self._robot._cur_fsm_state]._world_pos_sim[i] +\
            #                     (self._robot._state_estimator[self._robot._cur_fsm_state]._world_rot_mat_w2sim[i].transpose(-2, -1) @ (self._robot._eef_pos_w[i, 1].unsqueeze(-1))).squeeze(-1)
            #     cur_eef_pose.p = gymapi.Vec3(cur_eef_pos_sim[0], cur_eef_pos_sim[1], cur_eef_pos_sim[2])
            #     # cur_eef_rpy_sim = rot_mat_to_rpy((self._robot._state_estimator[self._robot._cur_fsm_state]._world_rot_mat_w2sim[i].transpose(-2, -1) @ (rpy_to_rot_mat(self._robot.eef_rpy_w[:, 0])[i])).unsqueeze(0)).squeeze(0)
            #     cur_eef_rpy_sim = rot_mat_to_rpy(self._robot._state_estimator[self._robot._cur_fsm_state]._world_rot_mat_w2sim.transpose(-2, -1) @ self._robot.eef_rot_w[:, 1])[i]
            #     cur_eef_pose.r = gymapi.Quat.from_euler_zyx(cur_eef_rpy_sim[0], cur_eef_rpy_sim[1], cur_eef_rpy_sim[2])
            #     gymutil.draw_lines(cur_eef_frame, self._gym, self._viewer, self._robot._envs[i], cur_eef_pose)

                # print('eef_rpy_w: ', self._robot.eef_rpy_w_np[:, 0])
                # print('eef_rot_w: ', self._robot.eef_rot_w_np[:, 0])
                # print('eef_rpy_from_computed: ', rpy_to_rot_mat(self._robot.eef_rpy_w[:, 0])[0])

                # des_eef_pose = gymapi.Transform()
                # # des_eef_pos_world = torch.tensor(self._wbc_list[i]._des_footeef_pos[3:6], device=self._robot._device, dtype=torch.float)
                # # des_eef_pos_sim = self._robot._state_estimator[self._robot._cur_fsm_state]._world_pos_sim[i] +\
                # #                 (self._robot._state_estimator[self._robot._cur_fsm_state]._world_rot_mat_w2sim[i].transpose(-2, -1) @ des_eef_pos_world.unsqueeze(-1)).squeeze(-1)
                # # des_eef_pose.p = gymapi.Vec3(des_eef_pos_sim[0], des_eef_pos_sim[1], des_eef_pos_sim[2])
                # des_eef_pose.p = gymapi.Vec3(cur_eef_pos_sim[0], cur_eef_pos_sim[1], cur_eef_pos_sim[2]+0.05)
                # des_eef_rpy_world = torch.tensor(self._wbc_list[i]._des_eef_pos, device=self._robot._device, dtype=torch.float)
                # des_eef_rpy_sim = rot_mat_to_rpy((self._robot._state_estimator[self._robot._cur_fsm_state]._world_rot_mat_w2sim[i].transpose(-2, -1) @ (rpy_to_rot_mat(des_eef_rpy_world)[1].unsqueeze(0)))).squeeze(0)
                # des_eef_pose.r = gymapi.Quat.from_euler_zyx(des_eef_rpy_sim[0], des_eef_rpy_sim[1], des_eef_rpy_sim[2])
                # gymutil.draw_lines(des_eef_frame, self._gym, self._viewer, self._robot._envs[i], des_eef_pose)


                # foot_pose = gymapi.Transform()
                # foot_pos_sim = self._robot._foot_pos_sim[i, 1]
                # foot_pose.p = gymapi.Vec3(foot_pos_sim[0], foot_pos_sim[1], foot_pos_sim[2])
                # foot_rot_world = torch.tensor(self._wbc_list[i]._robot_model.framePlacement(None, self._wbc_list[i]._foot_frame_ids[1], update_kinematics=False).rotation, device=self._robot._device, dtype=torch.float)
                # foot_rpy_sim = rot_mat_to_rpy((self._robot._state_estimator[self._robot._cur_fsm_state]._world_rot_mat_w2sim[i].transpose(-2, -1) @ (foot_rot_world.unsqueeze(0)))).squeeze(0)
                # foot_pose.r = gymapi.Quat.from_euler_zyx(foot_rpy_sim[0], foot_rpy_sim[1], foot_rpy_sim[2])
                # gymutil.draw_lines(foot_frame, self._gym, self._viewer, self._robot._envs[i], foot_pose)


            # current_eef_pose = gymapi.Transform()
            # current_eef_pose.p = gymapi.Vec3(self._robot._eef_pos_sim[i, self._action_planner._manipulate_leg_idx, 0],
            #                                 self._robot._eef_pos_sim[i, self._action_planner._manipulate_leg_idx, 1],
            #                                 self._robot._eef_pos_sim[i, self._action_planner._manipulate_leg_idx, 2])
            # current_eef_pose.r = gymapi.Quat(self._robot._eef_quat_sim[i, self._action_planner._manipulate_leg_idx, 0],
            #                                     self._robot._eef_quat_sim[i, self._action_planner._manipulate_leg_idx, 1],
            #                                 self._robot._eef_quat_sim[i, self._action_planner._manipulate_leg_idx, 2],
            #                                 self._robot._eef_quat_sim[i, self._action_planner._manipulate_leg_idx, 3])
            # gymutil.draw_lines(current_axes_geom, self._gym, self._viewer, self._robot._envs[i], current_eef_pose)



