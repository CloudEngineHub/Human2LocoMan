import copy
from config.config import Cfg
from robot.base_robot import BaseRobot
from robot.motors import MotorCommand, MotorControlMode
from utilities.rotation_utils import rpy_vel_to_skew_synmetric_mat
import unitree_legged_sdk.lib.python.amd64.robot_interface as sdk
import torch
import time
# for communication with gripper
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from multiprocessing import shared_memory
import ros_numpy
from std_msgs.msg import Float32MultiArray


class RealRobot(BaseRobot):
    def __init__(self, cfg: Cfg):
        super().__init__(cfg)
        self._init_interface()
        self._init_buffers()
        self.first_reset = True
        self._torques = torch.zeros(1, 18, dtype=torch.float, device=self._device, requires_grad=False)
        self.robot_state_publisher = rospy.Publisher(Cfg.teleoperation.robot_state_topic, Float32MultiArray, queue_size=1)
        self.ros_proprio_msg = Float32MultiArray()

    def _init_interface(self):
        if self._use_gripper:
            self.des_pos_sim_pub = rospy.Publisher(self._cfg.gripper.gripper_des_pos_sim_topic, JointState, queue_size=1)
            self.des_pos_sim_msg = JointState()
            self.des_pos_sim_msg.name = ['do_not_update_state']

            self.cur_state_sim_sub = rospy.Subscriber(self._cfg.gripper.gripper_cur_state_sim_topic, JointState, self.update_gripper_state_callback)

            manipulator_motor_number = len(self._cfg.gripper.motor_ids)
            self._gripper_joint_state_buffer = torch.zeros((2, manipulator_motor_number), dtype=torch.float, device=self._device, requires_grad=False)
            self._received_first_gripper_state = False

            self._gripper_reset_pos = torch.tensor(self._cfg.gripper.reset_pos_sim, dtype=torch.float, device=self._device, requires_grad=False)
            self._des_gripper_pos = torch.zeros_like(self._gripper_reset_pos)
            self._dof_within_manipulator_idx = self._cfg.gripper.dof_idx
            self._gripper_within_manipulator_idx = self._cfg.gripper.gripper_idx
            # self._close_gripper = True
            self._open_degree = 0.5
            self._close_degree = self._gripper_reset_pos[3]
            self._update_gripper_interval = 1.0 / self._cfg.gripper.update_gripper_freq
            self._last_time_gripper_updated = time.time()
            self._requesting_gripper_state = False

        self._udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
        self._safe = sdk.Safety(sdk.LeggedType.Go1)
        self._power_protect_level = self._cfg.motor_controller.power_protect_level
        self._cmd = sdk.LowCmd()
        self._raw_state = sdk.LowState()
        self._udp.InitCmdData(self._cmd)

    def _init_buffers(self):
        self._last_update_state_time = time.time()
        self._contact_force_threshold = torch.zeros(self._num_legs, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_init_pos = self._motors.init_positions

    def update_gripper_state_callback(self, joint_msg: JointState):
        # self._joint_pos[:, self._gripper_joint_idx] = torch.tensor(joint_msg.position, dtype=torch.float, device=self._device, requires_grad=False)
        # self._joint_vel[:, self._gripper_joint_idx] = torch.tensor(joint_msg.velocity, dtype=torch.float, device=self._device, requires_grad=False)
        if not self._received_first_gripper_state:
            self._received_first_gripper_state = True
        self._last_time_gripper_updated = time.time()
        self._gripper_joint_state_buffer[0, :] = torch.tensor(joint_msg.position, dtype=torch.float, device=self._device, requires_grad=False)
        self._gripper_joint_state_buffer[1, :] = torch.tensor(joint_msg.velocity, dtype=torch.float, device=self._device, requires_grad=False)
        self._requesting_gripper_state = False

    def reset(self, reset_time: float = 2.5):
        # make sure the communication is ready
        if not self.first_reset:
            time.sleep(0.1)
            return
        
        zero_action = MotorCommand(desired_position=torch.zeros((self._num_envs, self._num_joints), device=self._device),
                        kp=torch.zeros((self._num_envs, self._num_joints), device=self._device),
                        desired_velocity=torch.zeros((self._num_envs, self._num_joints), device=self._device),
                        kd=torch.zeros((self._num_envs, self._num_joints), device=self._device),
                        desired_extra_torque=torch.zeros((self._num_envs, self._num_joints), device=self._device))
        
        if self._use_gripper:
            rospy.sleep(1.0)
            self._last_time_gripper_updated = time.time()
            self._requesting_gripper_state = True
            self.des_pos_sim_msg.name = ['update_state']
            self.des_pos_sim_msg.header.stamp = rospy.Time.now()
            self.des_pos_sim_msg.position = list(self._gripper_reset_pos.cpu().numpy())
            self.des_pos_sim_pub.publish(self.des_pos_sim_msg)
            zero_action.desired_position[:, self._gripper_joint_idx] = self._gripper_reset_pos[self._dof_within_manipulator_idx]

        for _ in range(10):
            self.step(zero_action, gripper_cmd=True)
        print("Ready to reset the robot!")
        
        initial_joint_pos = self.joint_pos
        stable_joint_pos = self._motors.init_positions
        # Stand up in 1 second, and collect the foot contact forces afterwards
        reset_time = self._cfg.motor_controller.reset_time + 1.0
        standup_time = self._cfg.motor_controller.reset_time
        stand_foot_forces = []
        # time_before = time.time()
        # thredshold = 0.0
        for t in torch.arange(0, reset_time, self._dt):
            blend_ratio = min(t / standup_time, 1)
            desired_joint_pos = blend_ratio * stable_joint_pos + (1 - blend_ratio) * initial_joint_pos
            stand_up_action = MotorCommand(desired_position=desired_joint_pos,
                                kp=self._motors.kps,
                                desired_velocity=torch.zeros((self._num_envs, self._num_joints), device=self._device),
                                kd=self._motors.kds,
                                desired_extra_torque=torch.zeros((self._num_envs, self._num_joints), device=self._device))
            
            if not rospy.is_shutdown():
                self.step(stand_up_action, MotorControlMode.POSITION, gripper_cmd=False)

            if t > standup_time:
                stand_foot_forces.append(self.raw_contact_force)
        # Calibrate foot force sensors
        if stand_foot_forces:
            stand_foot_forces_tensor = torch.stack(stand_foot_forces)
            mean_foot_forces = torch.mean(stand_foot_forces_tensor, dim=0)
        else:
            mean_foot_forces = torch.zeros_like(self._contact_force_threshold)
        self._contact_force_threshold[:] = mean_foot_forces * 0.8
        self._update_state(reset_estimator=True, env_ids=torch.arange(self._num_envs, device=self._device))  # for updating foot contact state
        self._num_step[:] = 0
        print("Robot reset done!")
        self.first_reset = False


    def step(self, action: MotorCommand, motor_control_mode: MotorControlMode = None, gripper_cmd=True):
                
        self._num_step[:] += 1
        self._log_info_now = self._log_info and self._num_step[0] % self._log_interval == 0
        self._apply_action(action, motor_control_mode, gripper_cmd=gripper_cmd)
        time.sleep(max(self._dt- (time.time()-self._last_update_state_time), 0))
        self._update_state()
        
        # proprioceptive state
        # body 6d pose in the world frame: [6]
        # right eef 6d pose in the world frame: [6]
        # left eef 6d pose in the world frame: [6]
        # gripper angles: [2]
        # joint positions and velocities: [18 + 18]
        body_pos = self.base_pos_w_np.flatten()
        body_rpy = self.base_rpy_w2b_np.flatten()
        eef_pos = self.eef_pos_w_np.flatten()
        eef_rpy = self.eef_rpy_w_np.flatten()
        gripper_angles = self.gripper_angles_np.flatten()
        joint_pos = self.joint_pos_np.flatten()
        joint_vel = self.joint_vel_np.flatten()
        proprio_states = np.concatenate([body_pos, body_rpy, eef_pos[:3], eef_rpy[:3], eef_pos[3:], eef_rpy[3:], gripper_angles, joint_pos, joint_vel])
        self.ros_proprio_msg.data = proprio_states.tolist()
        self.robot_state_publisher.publish(self.ros_proprio_msg)
        # cam_image = None
        # try:
        #     shm = shared_memory.SharedMemory(name=Cfg.teleoperation.shm_name)
        #     cam_image = np.ndarray(shape=(Cfg.teleoperation.stereo_rgb_resolution[1], Cfg.teleoperation.stereo_rgb_resolution[0], 3), 
        #                        dtype=np.uint8, buffer=shm.buf)
        #     shm.close()
        # except:
        #     cam_image = None
        # try:
        #     wrist_shm = shared_memory.SharedMemory(name=Cfg.teleoperation.wrist_shm_name)
        #     wrist_image = np.ndarray(shape=(Cfg.teleoperation.rgb_resolution[1], Cfg.teleoperation.rgb_resolution[0], 3), 
        #                        dtype=np.uint8, buffer=wrist_shm.buf)
        #     wrist_shm.close()
        # except:
        #     wrist_image = None
            
        # obs_dict = {
        #     'qpos': self._joint_pos[0],
        #     'images': {
        #         'main': copy.copy(cam_image) if cam_image is not None else None,
        #         'wrist': copy.copy(wrist_image) if wrist_image is not None else None
        #     }
        # }
        
        # return {
        #     'observation': obs_dict,
        #     'reward': 0.
        # }
    def _action_to_torque(self, action: MotorCommand, motor_control_mode: MotorControlMode = None):
        if motor_control_mode is None:
            motor_control_mode = self._motors._motor_control_mode
        if motor_control_mode == MotorControlMode.POSITION:
            self._torques[:] = action.kp * (action.desired_position - self._joint_pos) - action.kd * self._joint_vel
        elif motor_control_mode == MotorControlMode.TORQUE:
            self._torques[:] = action.desired_extra_torque
        elif motor_control_mode == MotorControlMode.HYBRID:
            # print('delta p control', abs(action.kp * (action.desired_position - self._joint_pos))[0, :6])
            self._torques[:] = action.kp * (action.desired_position - self._joint_pos) +\
                               action.kd * (action.desired_velocity - self._joint_vel) +\
                               action.desired_extra_torque
            # if torch.max(abs(self._torques[0, :6])) > 3.5:
            #     print('delta q', (action.desired_position - self._joint_pos)[0, :6])
            #     print('delta dq', (action.desired_velocity - self._joint_vel)[0, :6])
            #     print('desired torque', action.desired_extra_torque)
            #     print('applied torques', self._torques[0, :6])
            #     print('applied torques max', torch.max(abs(self._torques[0, :6])))            
        else:
            raise ValueError('Unknown motor control mode for Go1 robot: {}.'.format(motor_control_mode))
        # if self._log_info_now:
        #     print('kp: ', action.kp[0].cpu().numpy())
        #     print('kd: ', action.kd[0].cpu().numpy())
        #     print('desired_position: ', action.desired_position[0].cpu().numpy())
        #     print('joint_pos: ', self._joint_pos[0].cpu().numpy())
        #     print('joint_vel: ', self._joint_vel[0].cpu().numpy())
        #     print('torques: ', self._torques[0].cpu().numpy())

    def _apply_action(self, action: MotorCommand, motor_control_mode: MotorControlMode = None, gripper_cmd=True):
        if motor_control_mode is None:
            motor_control_mode = self._motors._motor_control_mode
        if motor_control_mode == MotorControlMode.POSITION:
            for motor_id in range(self._dog_num_joints):
                self._cmd.motorCmd[motor_id].q = action.desired_position.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].Kp = action.kp.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].dq = 0.
                self._cmd.motorCmd[motor_id].Kd = action.kd.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].tau = 0.
        elif motor_control_mode == MotorControlMode.TORQUE:
            for motor_id in range(self._dog_num_joints):
                self._cmd.motorCmd[motor_id].q = 0.
                self._cmd.motorCmd[motor_id].Kp = 0.
                self._cmd.motorCmd[motor_id].dq = 0.
                self._cmd.motorCmd[motor_id].Kd = 0.
                self._cmd.motorCmd[motor_id].tau = action.desired_torque.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
        elif motor_control_mode == MotorControlMode.HYBRID:
            for motor_id in range(self._dog_num_joints):
                self._cmd.motorCmd[motor_id].q = action.desired_position.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].Kp = action.kp.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].dq = action.desired_velocity.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].Kd = action.kd.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].tau = action.desired_extra_torque.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
        else:
            raise ValueError('Unknown motor control mode for Go1 robot: {}.'.format(motor_control_mode))
        self._action_to_torque(action, motor_control_mode)
        self._safe.PowerProtect(self._cmd, self._raw_state, self._power_protect_level)
        self._udp.SetSend(self._cmd)
        self._udp.Send()
        # if self._log_info_now:
        #     print('kp: ', action.kp[0].cpu().numpy())
        #     print('kd: ', action.kd[0].cpu().numpy())

        if gripper_cmd and self._use_gripper and not self._requesting_gripper_state:
            des_dof_pos = action.desired_position[0, self._gripper_joint_idx]
            self._des_gripper_pos[0:3] = des_dof_pos[0:3]
            self._des_gripper_pos[4:7] = des_dof_pos[3:6]
            # self._des_gripper_pos[3] = self._close_degree if self._close_gripper[0, 0] else self._open_degree
            # self._des_gripper_pos[7] = self._close_degree if self._close_gripper[0, 1] else self._open_degree
            self._des_gripper_pos[[3, 7]] = self._gripper_desired_angles[0, :]

            if (time.time()-self._last_time_gripper_updated) > self._update_gripper_interval:
                self.des_pos_sim_msg.name = ['update_state']
                self._requesting_gripper_state = True
            else:
                self.des_pos_sim_msg.name = ['do_not_update_state']
            self.des_pos_sim_msg.header.stamp = rospy.Time.now()
            self.des_pos_sim_msg.position = list(self._des_gripper_pos.cpu().numpy())
            self.des_pos_sim_pub.publish(self.des_pos_sim_msg)
            # print('des_pos_sim.name: ', self.des_pos_sim_msg.name)
            # print('des_pos_sim.position: ', self.des_pos_sim_msg.position)
            # print('requesting_gripper_state: ', self._requesting_gripper_state)


    def _update_sensors(self):
        self._last_update_state_time = time.time()

        # gripper
        if self._use_gripper:
            if not self._received_first_gripper_state:
                self._joint_pos[:, self._gripper_joint_idx] = self._gripper_reset_pos[self._dof_within_manipulator_idx]
                self._joint_vel[:, self._gripper_joint_idx] = 0.0
                self._gripper_angles[:] = self._gripper_reset_pos[self._gripper_within_manipulator_idx]
            else:
                self._joint_pos[:, self._gripper_joint_idx] = self._gripper_joint_state_buffer[0, self._dof_within_manipulator_idx]
                self._joint_vel[:, self._gripper_joint_idx] = self._gripper_joint_state_buffer[1, self._dof_within_manipulator_idx]
                self._gripper_angles[:] = self._gripper_joint_state_buffer[0, self._gripper_within_manipulator_idx]

        # dog
        self._udp.Recv()
        self._udp.GetRecv(self._raw_state)
        for motor_id in range(self._dog_num_joints):
            self._joint_pos[:, self._dog_joint_idx[motor_id]] = self._raw_state.motorState[motor_id].q
            self._joint_vel[:, self._dog_joint_idx[motor_id]] = self._raw_state.motorState[motor_id].dq

    def _update_foot_contact_state(self):
        self._foot_contact[:] = self.raw_contact_force > self._contact_force_threshold

    def _update_foot_jocabian_position_velocity(self):
        self._compute_foot_jacobian()

        self._foot_pos_hip[:], self._foot_vel_hip[:] = self._forward_kinematics(return_frame='hip')
        self._foot_pos_b[:] = self._foot_pos_hip + self._HIP_OFFSETS
        self._foot_vel_b[:] = self._foot_vel_hip

    def _update_foot_global_state(self):
        # ----------------- compute foot global position -----------------
        self._foot_pos_w[:] = torch.bmm(self._base_rot_mat_w2b, self._foot_pos_b.transpose(1, 2)).transpose(1, 2) + self._base_pos_w.unsqueeze(1)

        # ----------------- compute foot global velocity -----------------
        # Vf^w = Vb^w + [w_b^w] * R_w^b * pf^b + R^w2b * Vf^b
        self._foot_vel_w[:] = self._base_lin_vel_w.unsqueeze(1) + \
                            torch.bmm(rpy_vel_to_skew_synmetric_mat(self._base_ang_vel_w), torch.bmm(self._base_rot_mat_b2w.transpose(-2, -1), self._foot_pos_b.transpose(-2, -1))).transpose(1, 2) +\
                            torch.bmm(self._base_rot_mat_w2b, self._foot_vel_b.transpose(1, 2)).transpose(1, 2)

    @property
    def raw_contact_force(self):
        return torch.tensor(self._raw_state.footForce, dtype=torch.float, device=self._device, requires_grad=False)


