import argparse
import os
import time
import rospy
from std_msgs.msg import Float32MultiArray, Int32
import numpy as np
from locoman.config.config import Cfg
from locoman.utilities.orientation_utils_numpy import rot_mat_to_rpy, rpy_to_rot_mat
from tele_vision import OpenTeleVision
from sensor_msgs.msg import Image
import ros_numpy
import cv2
import threading
from camera_utils import list_video_devices, find_device_path_by_name
from multiprocessing import Array, Process, shared_memory
import sys
import signal
import h5py
from datetime import datetime

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class AVPTeleoperator:
    def __init__(self, args):
        rospy.init_node('avp_teleoperation')
        self.args = args
        self.checkpt = 0
        self.command_publisher = rospy.Publisher(Cfg.commander.human_command_topic, Float32MultiArray, queue_size=1)
        self.fsm_publisher = rospy.Publisher(Cfg.fsm_switcher.fsm_state_topic, Int32, queue_size = 1)
        self.fsm_to_teleop_mode_mapping = Cfg.teleoperation.human_teleoperator.fsm_to_teleop_mode_mapping
        
        if self.args.use_real_robot:
            self.init_cameras()
            if self.args.head_camera_type == 0:
                img_shape = (self.head_frame_res[0], self.head_frame_res[1], 3)
                self.shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize, name=Cfg.teleoperation.shm_name)
                self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)
                self.img_array[:] = np.zeros(img_shape, dtype=np.uint8)
                self.tele_vision = OpenTeleVision(self.head_frame_res, self.shm.name, False)
            elif self.args.head_camera_type == 1:
                img_shape = (self.head_frame_res[0], 2 * self.head_frame_res[1], 3)
                self.shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize, name=Cfg.teleoperation.shm_name)
                self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)
                self.img_array[:] = np.zeros(img_shape, dtype=np.uint8)
                self.tele_vision = OpenTeleVision(self.head_frame_res, self.shm.name, True)
            else:
                raise NotImplementedError("Not supported camera.")
        else:
            # simulation view subscriber
            self.sim_view_subscriber = rospy.Subscriber(Cfg.teleoperation.teleop_view_topic, Image, self.sim_view_callback)
            img_shape = (Cfg.teleoperation.fpv_height, Cfg.teleoperation.fpv_width, 3)
            self.shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
            self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)
            self.img_array[:] = np.zeros(img_shape, dtype=np.uint8)
            self.sim_view_frame = np.zeros_like(self.img_array)
            self.tele_vision = OpenTeleVision((Cfg.teleoperation.fpv_height, Cfg.teleoperation.fpv_width), self.shm.name, False)
        
        # command buffer
        self.command = np.zeros(20)  # body: xyzrpy, eef_r: xyzrpy, eef_l: xyzrpy, grippers: 2 angles
        self.command_msg = Float32MultiArray()
        self.command_msg.data = self.command.tolist()
        self.robot_command = np.zeros(20)
        # eef 6d pose in two types of defined unified frame
        self.eef_uni_command = np.zeros((2, 6))
        
        self.fsm_state_msg = Int32()
        self.fsm_state_msg.data = 0

        # initial values
        self.initial_receive = True
        self.init_body_pos = np.zeros(3)
        self.init_body_rot = np.eye(3)
        self.init_eef_pos = np.zeros((2, 3)) 
        self.init_eef_rot = np.array([np.eye(3), np.eye(3)])
        self.init_gripper_angles = np.zeros(2)

        self.is_changing_reveive_status = False
        self.begin_to_receive = False
        self.reset_signal_subscriber = rospy.Subscriber(Cfg.teleoperation.receive_action_topic, Int32, self.reset_signal_callback, queue_size=1)
        self.teleoperation_mode = Cfg.teleoperation.human_teleoperator.mode
        self.is_changing_teleop_mode = False
        self.teleop_mode_subscriber = rospy.Subscriber(Cfg.teleoperation.human_teleoperator.mode_updata_topic, Int32, self.teleop_mode_callback, queue_size=1)
        self.rate = rospy.Rate(200)
        
        self.robot_reset_publisher = rospy.Publisher(Cfg.teleoperation.robot_reset_topic, Int32, queue_size = 1)
        self.robot_reset_msg = Int32()
        self.robot_reset_msg.data = 0

        self.robot_reset_pose_subscriber = rospy.Subscriber(Cfg.commander.robot_reset_pose_topic, Float32MultiArray, self.robot_reset_pose_callback, queue_size=1)
        self.robot_reset_pose = np.zeros(20)
        self.on_reset = False
        self.reset_finished = False

        self.init_embodiment_proprio_states()
        self.robot_state_subscriber = rospy.Subscriber(Cfg.teleoperation.robot_state_topic, Float32MultiArray, self.robot_proprio_state_callback, queue_size=1)
        
        # self.pinch_gripper_angle_scale = 10.0
        self.pinch_dist_gripper_full_close = 0.02
        self.pinch_dist_gripper_full_open = 0.15
        # self.gripper_full_close_angle = 0.33
        self.gripper_full_close_angle = Cfg.commander.gripper_angle_range[0]
        self.gripper_full_open_angle = Cfg.commander.gripper_angle_range[1]
        self.eef_xyz_scale = 1.0
        self.manipulate_eef_idx = [0]
        
        # to get the ratation matrix of the world frame built by the apple vision pro, relative to the world frame of IssacGym or the real world used by LocoMan
        # first rotate along x axis with 90 degrees, then rotate along z axis with -90 degrees, in the frame of IsaacGym
        self.operator_pov_transform = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ]) @ np.array ([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])
        
        self.last_reset_time = 0.0
        self.last_calib_time = 0.0
        self.pause_commands = False
        self.record_episode_counter = 0
        
        # prepare for data processing and collection
        self.body_xyz_scale = Cfg.teleoperation.human_teleoperator.body_xyz_scale
        self.body_rpy_scale = Cfg.teleoperation.human_teleoperator.body_rpy_scale
        self.eef_xyz_scale = Cfg.teleoperation.human_teleoperator.eef_xyz_scale
        self.eef_rpy_scale = Cfg.teleoperation.human_teleoperator.eef_rpy_scale     
        self.gripper_angle_scale = Cfg.teleoperation.human_teleoperator.gripper_angle_scale
        self.human_command_body_rpy_range = np.array([Cfg.teleoperation.human_teleoperator.body_r_range,
                                                      Cfg.teleoperation.human_teleoperator.body_p_range,
                                                      Cfg.teleoperation.human_teleoperator.body_y_range,])
        
        self.reset_trajectories()
        self.init_embodiment_command_states()

        # data collection
        if self.args.collect_data:
            self.robot_data_folder = 'demonstrations'
            self.exp_name = self.args.exp_name
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_data_folder = self.robot_data_folder + '/' + self.exp_name + '/locoman/' + current_time
            if not os.path.exists(self.exp_data_folder):
                os.makedirs(self.exp_data_folder)

    def reset_pov_transform(self):
        self.operator_pov_transform = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ]) @ np.array ([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])

    def get_gripper_angle_from_pinch_dist(self, pinch_dist):
        scale = (self.gripper_full_open_angle - self.gripper_full_close_angle) / (self.pinch_dist_gripper_full_open - self.pinch_dist_gripper_full_close)
        angle = (pinch_dist - self.pinch_dist_gripper_full_close) * scale + self.gripper_full_close_angle
        return np.clip(angle, self.gripper_full_close_angle, self.gripper_full_open_angle)
    
    def reset_signal_callback(self, msg):
        self.is_changing_reveive_status = True
        if msg.data == 0:
            self.begin_to_receive = False
            self.initial_receive = True
            print("No longer receiving. Initial receive status reset.")
        elif msg.data == 1:
            self.begin_to_receive = True
            self.initial_receive = True
            if self.args.teleop_mode == 1:
                # right gripper
                self.fsm_state_msg.data = 3
            elif self.args.teleop_mode == 2:
                # left gripper
                self.fsm_state_msg.data = 4
            elif self.args.teleop_mode == 3:
                # bi-manual
                self.fsm_state_msg.data = 7
            else:
                # stance
                self.fsm_state_msg.data = 0
            # 0:stance, 1:right manipulation, 2: left manipulation 3: bi-manual manipulation 4: None
            # fsm_to_teleop_mode_mapping = [0, 1, 2, 1, 2, 4, 4, 3]
            self.teleoperation_mode = self.fsm_to_teleop_mode_mapping[self.fsm_state_msg.data]
            self.fsm_publisher.publish(self.fsm_state_msg)
            print("Begin to receive. Initial receive status reset.")
        self.is_changing_reveive_status = False
    
    def update_manipulate_eef_idx(self):
        if self.teleoperation_mode == 1:
            # right gripper manipulation
            self.manipulate_eef_idx = [0]
        elif self.teleoperation_mode == 2:
            # left gripper manipulation
            self.manipulate_eef_idx = [1]
        elif self.teleoperation_mode == 3:
            # bimanual manipulation
            self.manipulate_eef_idx = [0, 1]
        
    def teleop_mode_callback(self, msg):
        self.is_changing_teleop_mode = True
        self.teleoperation_mode = msg.data
        print(f"Teleoperation mode updated to {self.teleoperation_mode}.")
        self.is_changing_teleop_mode = False

    def sim_view_callback(self, msg):
        self.sim_view_frame = ros_numpy.numpify(msg)
        np.copyto(self.img_array, self.sim_view_frame)
        # self.tele_vision.modify_shared_image(sim_view_image)
        
    def robot_proprio_state_callback(self, msg):
        # update proprio states of locoman
        self.body_pose_proprio_callback = np.array(msg.data)[:6]
        self.eef_pose_proprio_callback = np.array(msg.data)[6:18]
        self.gripper_angle_proprio_callback = np.array(msg.data)[18:20]
        self.joint_pos_proprio_callback = np.array(msg.data)[20:38]
        self.joint_vel_proprio_callback = np.array(msg.data)[38:56]

    def get_proprio_state(self, unify_bimanual_body_frame=True):
        self.body_pose_proprio = self.body_pose_proprio_callback.copy()
        if self.teleoperation_mode == 3 and unify_bimanual_body_frame:
            # modify the torso local frame to roughly align with the world frame (x forward, y leftward, z upward) during the bimanual mode
            body_pose_proprio_rpy = self.body_pose_proprio[3:6]
            body_pose_proprio_rot_mat = rpy_to_rot_mat(body_pose_proprio_rpy)
            body_pose_proprio_rot_mat_new = np.zeros_like(body_pose_proprio_rot_mat)
            body_pose_proprio_rot_mat_new[:, 0] = -body_pose_proprio_rot_mat[:, 2]
            body_pose_proprio_rot_mat_new[:, 1] = body_pose_proprio_rot_mat[:, 1]
            body_pose_proprio_rot_mat_new[:, 2] = body_pose_proprio_rot_mat[:, 0]
            self.body_pose_proprio[3:6] = rot_mat_to_rpy(body_pose_proprio_rot_mat_new)

        self.eef_pose_proprio = self.eef_pose_proprio_callback.copy()
        self.gripper_angle_proprio = self.gripper_angle_proprio_callback.copy()
        self.joint_pos_proprio = self.joint_pos_proprio_callback.copy()
        self.joint_vel_proprio = self.joint_vel_proprio_callback.copy()

        right_eef_rpy = self.eef_pose_proprio[3:6]
        left_eef_rpy = self.eef_pose_proprio[9:12]
        right_eef_rot_mat = rpy_to_rot_mat(right_eef_rpy)
        left_eef_rot_mat = rpy_to_rot_mat(left_eef_rpy)

        # use the unified definition for eef local frames 
        new_right_eef_rot_mat = np.zeros_like(right_eef_rot_mat)
        new_left_eef_rot_mat = np.zeros_like(left_eef_rot_mat)
        new_right_eef_rot_mat[:, 0] = -right_eef_rot_mat[:, 2]
        new_right_eef_rot_mat[:, 1] = right_eef_rot_mat[:, 0]
        new_right_eef_rot_mat[:, 2] = -right_eef_rot_mat[:, 1]
        new_left_eef_rot_mat[:, 0] = -left_eef_rot_mat[:, 2]
        new_left_eef_rot_mat[:, 1] = left_eef_rot_mat[:, 0]
        new_left_eef_rot_mat[:, 2] = -left_eef_rot_mat[:, 1]

        self.eef_pose_proprio[3:6] = rot_mat_to_rpy(new_right_eef_rot_mat)
        self.eef_pose_proprio[9:12] = rot_mat_to_rpy(new_left_eef_rot_mat)

        self.eef_to_body_pose_proprio = self.eef_to_body_pose_proprio.copy()
        self.eef_to_body_pose_proprio[:3] = self.eef_pose_proprio[:3] - self.body_pose_proprio[:3]
        self.eef_to_body_pose_proprio[6:9] = self.eef_pose_proprio[6:9] - self.body_pose_proprio[:3]
        self.eef_to_body_pose_proprio[3:6] = rot_mat_to_rpy(rpy_to_rot_mat(self.body_pose_proprio[3:6]).T @ new_right_eef_rot_mat)
        self.eef_to_body_pose_proprio[9:12] = rot_mat_to_rpy(rpy_to_rot_mat(self.body_pose_proprio[3:6]).T @ new_left_eef_rot_mat)

    def robot_reset_pose_callback(self, msg):
        self.robot_reset_pose = np.array(msg.data)
        if self.on_reset:
            self.reset_finished = True
            self.on_reset = False
            print('robot reset pose', self.robot_reset_pose)

    def init_cameras(self):
        import pyrealsense2 as rs
        # initialize all cameras
        self.desired_stream_fps = self.args.desired_stream_fps
        # initialize head camera
        # realsense as head camera
        if self.args.head_camera_type == 0:
            self.head_camera_resolution = Cfg.teleoperation.realsense_resolution
            self.head_frame_res = Cfg.teleoperation.head_view_resolution
            self.head_color_frame = np.zeros((self.head_frame_res[0], self.head_frame_res[1], 3), dtype=np.uint8)
            self.head_cam_pipeline = rs.pipeline()
            self.head_cam_config = rs.config()
            pipeline_wrapper = rs.pipeline_wrapper(self.head_cam_pipeline)
            pipeline_profile = self.head_cam_config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("a head camera is required for real-robot teleoperation")
                exit(0)
            self.head_cam_config.enable_stream(rs.stream.color, self.head_camera_resolution[1], self.head_camera_resolution[0], rs.format.bgr8, 30)
            # start streaming head cam
            self.head_cam_pipeline.start(self.head_cam_config)
        # stereo rgb camera (dual lens) as the head camera 
        elif self.args.head_camera_type == 1:
            # self.head_camera_resolution: the supported resoltion of the camera, to get the original frame without cropping
            self.head_camera_resolution = Cfg.teleoperation.stereo_rgb_resolution
            # self.head_view_resolution: the resolition of the images that are seen and recorded
            self.head_view_resolution = Cfg.teleoperation.head_view_resolution
            self.crop_size_w = 0
            self.crop_size_h = 0
            self.head_frame_res = (self.head_view_resolution[0] - self.crop_size_h, self.head_view_resolution[1] - 2 * self.crop_size_w)
            self.head_color_frame = np.zeros((self.head_frame_res[0], 2 * self.head_frame_res[1], 3), dtype=np.uint8)
            device_map = list_video_devices()
            head_camera_name = "3D USB Camera"
            device_path = find_device_path_by_name(device_map, head_camera_name)
            self.head_cap = cv2.VideoCapture(device_path)
            self.head_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.head_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * self.head_camera_resolution[1])
            self.head_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.head_camera_resolution[0])
            self.head_cap.set(cv2.CAP_PROP_FPS, self.desired_stream_fps)
        else:
            raise NotImplementedError("Not supported camera.")
        
        # initialize wrist camera/cameras
        if self.args.use_wrist_camera:
            self.wrist_camera_resolution = Cfg.teleoperation.rgb_resolution
            self.wrist_view_resolution = Cfg.teleoperation.wrist_view_resolution
            self.wrist_color_frame = np.zeros((self.wrist_view_resolution[0], self.wrist_view_resolution[1], 3), dtype=np.uint8)
            device_map = list_video_devices()
            wrist_camera_name = "Global Shutter Camera"
            device_path = find_device_path_by_name(device_map, wrist_camera_name)
            self.wrist_cap1 = cv2.VideoCapture(device_path)
            self.wrist_cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.wrist_cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.wrist_camera_resolution[1])
            self.wrist_cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.wrist_camera_resolution[0])
            self.wrist_cap1.set(cv2.CAP_PROP_FPS, self.desired_stream_fps)

    def head_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        if self.args.head_camera_type == 0:
            try:
                while not rospy.is_shutdown():
                    start_time = time.time()
                    # handle head camera (realsense) streaming 
                    frames = self.head_cam_pipeline.wait_for_frames()
                    # depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        return
                    head_color_frame = np.asanyarray(color_frame.get_data())
                    head_color_frame = cv2.resize(head_color_frame, (self.head_frame_res[1], self.head_frame_res[0]))
                    self.head_color_frame = cv2.cvtColor(head_color_frame, cv2.COLOR_BGR2RGB)
                    np.copyto(self.img_array, self.head_color_frame)
                    elapsed_time = time.time() - start_time
                    sleep_time = frame_duration - elapsed_time
            finally:
                self.head_cam_pipeline.stop()
        elif self.args.head_camera_type == 1:
            try:
                while not rospy.is_shutdown():
                    start_time = time.time()
                    ret, frame = self.head_cap.read()
                    frame = cv2.resize(frame, (2 * self.head_frame_res[1], self.head_frame_res[0]))
                    image_left = frame[:, :self.head_frame_res[1], :]
                    image_right = frame[:, self.head_frame_res[1]:, :]
                    if self.crop_size_w != 0:
                        bgr = np.hstack((image_left[self.crop_size_h:, self.crop_size_w:-self.crop_size_w],
                                        image_right[self.crop_size_h:, self.crop_size_w:-self.crop_size_w]))
                    else:
                        bgr = np.hstack((image_left[self.crop_size_h:, :],
                                        image_right[self.crop_size_h:, :]))

                    self.head_color_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    np.copyto(self.img_array, self.head_color_frame)
                    elapsed_time = time.time() - start_time
                    sleep_time = frame_duration - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            finally:
                self.head_cap.release()
        else:
            raise NotImplementedError('Not supported camera.')
        
    def wrist_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        try:
            while not rospy.is_shutdown():
                start_time = time.time()
                ret, frame = self.wrist_cap1.read()
                wrist_color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.wrist_color_frame = cv2.resize(wrist_color_frame, (self.wrist_view_resolution[1], self.wrist_view_resolution[0]))
                elapsed_time = time.time() - start_time
                sleep_time = frame_duration - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self.head_cap.release()
        
    def publish_command(self, event=None):
        # publish teleop commands at a fixed rate 
        # need to check the duration to finish one execution and if a separate thread is needed: takes 0.0002-0.0003s
        
        # left_hand_data: [4, 4]
        left_hand_data = self.tele_vision.left_hand
        # right_hand_data: [4, 4]
        right_hand_data = self.tele_vision.right_hand
        # left_landmarks_data: [25, 3]
        left_landmarks_data = self.tele_vision.left_landmarks
        # right_landmarks_data: [25, 3]
        right_landmarks_data = self.tele_vision.right_landmarks
        # head_data: [4, 4]
        head_data = self.tele_vision.head_matrix
        
        left_wrist_pos = left_hand_data[:3, 3]
        left_wrist_rot = left_hand_data[:3, :3]
        right_wrist_pos = right_hand_data[:3, 3]
        right_wrist_rot = right_hand_data[:3, :3]
        head_pos = head_data[:3, 3]
        head_rot = head_data[:3, :3]        
        
        ### define new head rot and wrist rot to align with the robot torso's and eef's local frame at reset, respectively
        ### this is used for teleoperation to control the robot, so it does not use the unified definition, instead uses the original local frame of the robot
        head_rot_new = np.zeros_like(head_rot)
        head_rot_new[:, 0] = -head_rot[:, 2]
        head_rot_new[:, 1] = -head_rot[:, 0]
        head_rot_new[:, 2] = head_rot[:, 1]
        head_rot_old = head_rot
        head_rot = head_rot_new
        
        left_wrist_rot_new = np.zeros_like(left_wrist_rot)
        left_wrist_rot_new[:, 0] = -left_wrist_rot[:, 0]
        left_wrist_rot_new[:, 1] = -left_wrist_rot[:, 1]
        left_wrist_rot_new[:, 2] = left_wrist_rot[:, 2]
        left_wrist_rot = left_wrist_rot_new
        
        right_wrist_rot_new = np.zeros_like(right_wrist_rot)
        right_wrist_rot_new[:, 0] = -right_wrist_rot[:, 0]
        right_wrist_rot_new[:, 1] = -right_wrist_rot[:, 1]
        right_wrist_rot_new[:, 2] = right_wrist_rot[:, 2]
        right_wrist_rot = right_wrist_rot_new

        # thumb to index finger
        left_pinch_dist0 = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[9]) 
        right_pinch_dist0 = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[9]) 
        # thumb to middle finger
        left_pinch_dist1 = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[14]) 
        # thumb to ring finger
        left_pinch_dist2 = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[19])
        # thumb to little finger
        left_pinch_dist3 = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[24])

        eef_pos = [right_wrist_pos, left_wrist_pos]
        eef_rot = [right_wrist_rot, left_wrist_rot]
        
        self.update_manipulate_eef_idx()
        # reset the robot
        if self.begin_to_receive:
            if left_pinch_dist1 < 0.005 and left_pinch_dist0 > 0.05 and left_pinch_dist2 > 0.05 and left_pinch_dist3 > 0.05 and time.time() - self.last_reset_time > 3.0:
                self.robot_reset_publisher.publish(self.robot_reset_msg)
                self.on_reset = True
                self.reset_finished = False
                self.last_reset_time = time.time()
                self.initial_receive = True
                self.reset_pov_transform()
                self.flush_trajectory()
                print("reset robot")
        self.command[:] = 0
        # initialize and calibrate
        if self.initial_receive and self.begin_to_receive and self.reset_finished:
            # the initialization only begins at a certain signal (e.g., a gesture)
            if left_pinch_dist0 < 0.005 and left_pinch_dist1 > 0.05 and left_pinch_dist2 > 0.05 and left_pinch_dist3 > 0.05:
                # extract the head rotation around y axis in the world frame of apple vision pro (head yaw)
                head_rpy = self.rot_mat_to_rpy_zxy(head_rot_old)
                # use the head yaw to build a new frame to map the 6d poses of the human hand and head
                # the new operator_pov_transform represents the rotation matrix of the apple vision pro frame relative to the IsaacGym/world frame
                z_axis_rot = rpy_to_rot_mat(np.array([0, 0, -head_rpy[1]]))
                self.operator_pov_transform = z_axis_rot @ self.operator_pov_transform
                self.init_body_pos[:] = self.operator_pov_transform @ head_pos
                self.init_body_rot[:] = self.operator_pov_transform @ head_rot
                    
                for eef_idx in self.manipulate_eef_idx:
                    self.init_eef_pos[eef_idx] = self.operator_pov_transform @ eef_pos[eef_idx]
                    self.init_eef_rot[eef_idx] = self.operator_pov_transform @ eef_rot[eef_idx]
                # self.init_gripper_angles[eef_idx] = right_pinch_dist0 * self.pinch_gripper_angle_scale
            else:
                return
            
            self.command_msg.data = self.command.tolist()
            self.command_publisher.publish(self.command_msg)
            self.initial_receive = False
            if self.args.collect_data:
                # command as zeros to keep the reset pose
                # get the embodiment states at reset
                self.get_proprio_state()
                self.reset_embodiment_command_states()
                self.update_embodiment_command_states(reset=True)
                self.update_trajectories()
            
        # start teleop
        elif self.begin_to_receive and self.reset_finished:
            if left_pinch_dist3 < 0.005 and left_pinch_dist0 > 0.05 and left_pinch_dist1 > 0.05 and left_pinch_dist2 > 0.05:
                self.pause_commands = True
                print("Pause sending commands")
                return
            if self.pause_commands and left_pinch_dist0 < 0.005 and left_pinch_dist1 > 0.05 and left_pinch_dist2 > 0.05 and left_pinch_dist3 > 0.05:
                print("Restart sending commands")
                self.pause_commands = False
            if self.pause_commands:
                return

            # torso 6d pose
            self.command[0:6] = np.zeros(6)
            self.command[3:6] = rot_mat_to_rpy(self.init_body_rot[:].T @ self.operator_pov_transform @ head_rot)
            gripper_command_source = [right_pinch_dist0, left_pinch_dist0]
            for eef_idx in self.manipulate_eef_idx:
                # eef 6d pose
                self.command[eef_idx*6+6:eef_idx*6+9] = self.operator_pov_transform @ eef_pos[eef_idx] - self.init_eef_pos[eef_idx]
                self.command[eef_idx*6+9:eef_idx*6+12] = rot_mat_to_rpy(self.init_eef_rot[eef_idx].T @ self.operator_pov_transform @ eef_rot[eef_idx])
                # gripper angle
                self.command[eef_idx+18] = self.get_gripper_angle_from_pinch_dist(gripper_command_source[eef_idx])

            self.command_msg.data = self.command.tolist()
            self.command_publisher.publish(self.command_msg)

            if self.args.collect_data:
                self.get_proprio_state()
                self.reset_embodiment_command_states()
                self.update_embodiment_command_states(reset=False)
                self.update_trajectories()

    def init_embodiment_proprio_states(self):
        self.body_pose_proprio_callback = np.zeros(6)
        self.eef_pose_proprio_callback = np.zeros(12)
        self.eef_to_body_pose_proprio_callback = np.zeros(12)
        self.gripper_angle_proprio_callback = np.zeros(2)
        self.joint_pos_proprio_callback = np.zeros(18)
        self.joint_vel_proprio_callback = np.zeros(18)

        self.body_pose_proprio = np.zeros(6)
        self.eef_pose_proprio = np.zeros(12)
        self.eef_to_body_pose_proprio = np.zeros(12)
        self.gripper_angle_proprio = np.zeros(2)
        self.joint_pos_proprio = np.zeros(18)
        self.joint_vel_proprio = np.zeros(18)

    def init_embodiment_command_states(self):
        self.body_pose_uni = np.zeros(6)
        self.delta_body_pose_uni = np.zeros(6)
        self.eef_pose_uni = np.zeros(12)
        self.delta_eef_pose_uni = np.zeros(12)
        self.eef_to_body_pose = np.zeros(12)
        self.gripper_angle = np.zeros(2)
        self.delta_gripper_angle = np.zeros(2)
        
    def reset_embodiment_command_states(self):
        # need to reset: make sure add the updated state to the trajectory meanwhile keep previous ones unchanged
        # not reset to zeros: copy the value to compute delta
        self.body_pose_uni = self.body_pose_uni.copy()
        self.delta_body_pose_uni = self.delta_body_pose_uni.copy()
        self.eef_pose_uni = self.eef_pose_uni.copy()
        self.delta_eef_pose_uni = self.delta_eef_pose_uni.copy()
        self.eef_to_body_pose = self.eef_to_body_pose.copy()
        self.gripper_angle = self.gripper_angle.copy()
        self.delta_gripper_angle = self.delta_gripper_angle.copy()

    def update_embodiment_command_states(self, reset=False):
        ### transform the command data to the motion data (unified for both the human and the robot) that can be directly used for training

        ## set the unified frame for both robots and the human: x forward, y leftward, z upward, which aligns with the issacgym world frame
        ## the unified frame is defined to align with LocoMan's torso frame when it firstly swiitch to the specific mode
        ## the unified frame is defined to align with human's head frame (while changin the xyz axes) at reset 
        ## the local frame of head/torso/eef should align with the unified frame with x forward, y leftward, z upward
        # body 6d pose [6] - not used by bimanual
        body_rot_mat_init = rpy_to_rot_mat(self.robot_reset_pose[3:6])
        body_rot_mat = body_rot_mat_init @ rpy_to_rot_mat(self.command[3:6] * self.body_rpy_scale)
        body_rot_uni = rot_mat_to_rpy(body_rot_mat)
        body_pos_uni = self.command[0:3] * self.body_xyz_scale + self.robot_reset_pose[0:3]
        body_pose_uni = np.concatenate((body_pos_uni, body_rot_uni))
        if reset:
            self.delta_body_pose_uni = np.zeros_like(self.body_pose_uni)
        else:
            self.delta_body_pose_uni = body_pose_uni - self.body_pose_uni
        self.body_pose_uni = body_pose_uni
        # eef 6d pose [6 * 2]
        if reset:
            self.delta_eef_pose_uni = np.zeros_like(self.eef_pose_uni)
        for eef_idx in self.manipulate_eef_idx:
            eef_rot_mat_init = rpy_to_rot_mat(self.robot_reset_pose[9+6*eef_idx:12+6*eef_idx])
            eef_rot_mat = eef_rot_mat_init @ rpy_to_rot_mat(self.command[9+6*eef_idx:12+6*eef_idx] * self.eef_rpy_scale)
            eef_rot_mat_new = np.zeros_like(eef_rot_mat)
            eef_rot_mat_new[:, 0] = -eef_rot_mat[:, 2]
            eef_rot_mat_new[:, 1] = eef_rot_mat[:, 0]
            eef_rot_mat_new[:, 2] = -eef_rot_mat[:, 1]
            eef_rot_mat = eef_rot_mat_new
            eef_rot_uni = rot_mat_to_rpy(eef_rot_mat)
            eef_pos_uni = self.command[6+6*eef_idx:9+6*eef_idx] * self.eef_xyz_scale + self.robot_reset_pose[6+6*eef_idx:9+6*eef_idx]
            eef_pose_uni = np.concatenate((eef_pos_uni, eef_rot_uni))
            if not reset:
                self.delta_eef_pose_uni[6*eef_idx:6+6*eef_idx] = eef_pose_uni - self.eef_pose_uni[6*eef_idx:6+6*eef_idx]
            self.eef_pose_uni[6*eef_idx:6+6*eef_idx] = eef_pose_uni
            # eef 6d pose relative to the body frame [6 * 2]
            eef_to_body_pos = eef_pos_uni - body_pos_uni
            eef_to_body_rot = rot_mat_to_rpy(body_rot_mat.T @ eef_rot_mat)
            self.eef_to_body_pose[6*eef_idx:6+6*eef_idx] = np.concatenate((eef_to_body_pos, eef_to_body_rot))
        # gripper angle [2]
        gripper_angle = self.command[18:20] * self.gripper_angle_scale + self.robot_reset_pose[18:20]
        if reset:
            self.delta_gripper_angle = np.zeros_like(self.gripper_angle)
        else:
            self.delta_gripper_angle = gripper_angle - self.gripper_angle
        self.gripper_angle = gripper_angle


    def reset_trajectories(self):
        self.main_cam_image_history = []
        self.wrist_cam_image_history = []

        self.body_pose_proprio_history = []
        self.eef_pose_proprio_history = []
        self.eef_to_body_pose_proprio_history = []
        self.gripper_angle_proprio_history = []
        self.joint_pos_proprio_history = []
        self.joint_vel_proprio_history = []

        self.body_pose_history = []
        self.delta_body_pose_history = []
        self.eef_pose_history = []
        self.delta_eef_pose_history = []
        self.eef_to_body_pose_history = []
        self.gripper_angle_history = []
        self.delta_gripper_angle_history = []

    def update_trajectories(self):
        if self.args.use_real_robot:
            self.main_cam_image_history.append(self.head_color_frame)
        else:
            self.main_cam_image_history.append(self.sim_view_frame)
        if self.args.use_wrist_camera:
            self.wrist_cam_image_history.append(self.wrist_color_frame) 

        self.body_pose_history.append(self.body_pose_uni)
        self.delta_body_pose_history.append(self.delta_body_pose_uni)
        self.eef_pose_history.append(self.eef_pose_uni)
        self.delta_eef_pose_history.append(self.delta_eef_pose_uni)
        self.eef_to_body_pose_history.append(self.eef_to_body_pose)
        self.gripper_angle_history.append(self.gripper_angle)
        self.delta_gripper_angle_history.append(self.delta_gripper_angle)

        self.body_pose_proprio_history.append(self.body_pose_proprio)
        self.eef_pose_proprio_history.append(self.eef_pose_proprio)
        self.eef_to_body_pose_proprio_history.append(self.eef_to_body_pose_proprio)
        self.gripper_angle_proprio_history.append(self.gripper_angle_proprio)
        self.joint_pos_proprio_history.append(self.joint_pos_proprio)
        self.joint_vel_proprio_history.append(self.joint_vel_proprio)

    def get_embodiment_masks(self):
        self.img_main_mask = np.array([True])
        self.img_wrist_mask = np.array([False, False])
        if self.args.use_wrist_camera:
            for eef_idx in self.manipulate_eef_idx:
                self.img_wrist_mask[self.manipulate_eef_idx] = True
        # though disable the robot body xyz actions for stability, it should be included the proprio info.
        self.proprio_body_mask = np.array([True, True, True, True, True, True])
        # within single gripper manipulation mode: (1) the inactive gripper does not directly have the 6d pose; (2) introduces irrelevant info.
        # even if we would consider bimanual mode, the inactive gripper 6d pose of the single gripper manipulation mode is still less relevant
        self.proprio_eef_mask = np.array([True] * 12)
        # self.proprio_eef_mask = np.array([False] * 12)
        # self.proprio_eef_mask[6*self.manipulate_eef_idx:6+6*self.manipulate_eef_idx] = True
        # joint position and joint velocity
        self.proprio_gripper_mask = np.array([True, True])
        # self.proprio_gripper_mask = np.array([False, False])
        # self.proprio_gripper_mask[self.manipulate_eef_idx] = True
        self.proprio_other_mask = np.array([True, True])
        if self.teleoperation_mode == 3:
            self.act_body_mask = np.array([False, False, False, False, False, False])
        else:
            self.act_body_mask = np.array([False, False, False, True, True, True])
        self.act_eef_mask = np.array([False] * 12)
        for eef_idx in self.manipulate_eef_idx:
            self.act_eef_mask[6*eef_idx:6+6*eef_idx] = True
        self.act_gripper_mask = np.array([False, False])
        for eef_idx in self.manipulate_eef_idx:
            self.act_gripper_mask[eef_idx] = True

    def flush_trajectory(self):
        if self.args.collect_data and len(self.joint_pos_proprio_history) > 0:
            self.get_embodiment_masks()
            save_path = self.exp_data_folder
            episode_num = self.increment_counter
            with h5py.File(os.path.join(save_path, f'episode_{episode_num}.hdf5'), 'a') as f:
                # add the embodiment tag
                f.attrs['embodiment'] = 'locoman'
                # save observations
                obs_group = f.create_group('observations')
                image_group = obs_group.create_group('images')
                image_group.create_dataset('main', data=self.main_cam_image_history)
                image_group.create_dataset('wrist', data=self.wrist_cam_image_history)
                proprio_group = obs_group.create_group('proprioceptions')
                proprio_group.create_dataset('body', data=self.body_pose_proprio_history)
                proprio_group.create_dataset('eef', data=self.eef_pose_proprio_history)
                proprio_group.create_dataset('eef_to_body', data=self.eef_to_body_pose_proprio_history)
                proprio_group.create_dataset('gripper', data=self.gripper_angle_proprio_history)
                proprio_other_group = proprio_group.create_group('other')
                proprio_other_group.create_dataset('joint_pos', data=self.joint_pos_proprio_history)
                proprio_other_group.create_dataset('joint_vel', data=self.joint_vel_proprio_history)
                # save actions
                action_group = f.create_group('actions')
                action_group.create_dataset('body', data=self.body_pose_history)
                action_group.create_dataset('delta_body', data=self.delta_body_pose_history)
                action_group.create_dataset('eef', data=self.eef_pose_history)
                action_group.create_dataset('delta_eef', data=self.delta_eef_pose_history)
                action_group.create_dataset('eef_to_body', data=self.eef_to_body_pose_history)
                action_group.create_dataset('gripper', data=self.gripper_angle_history)
                action_group.create_dataset('delta_gripper', data=self.delta_gripper_angle_history)
                # save masks
                mask_group = f.create_group('masks')
                mask_group.create_dataset('img_main', data=self.img_main_mask)
                mask_group.create_dataset('img_wrist', data=self.img_wrist_mask)
                mask_group.create_dataset('proprio_body', data=self.proprio_body_mask)
                mask_group.create_dataset('proprio_eef', data=self.proprio_eef_mask)
                mask_group.create_dataset('proprio_gripper', data=self.proprio_gripper_mask)
                mask_group.create_dataset('proprio_other', data=self.proprio_other_mask)
                mask_group.create_dataset('act_body', data=self.act_body_mask)
                mask_group.create_dataset('act_eef', data=self.act_eef_mask)
                mask_group.create_dataset('act_gripper', data=self.act_gripper_mask)
            # save videos
            if self.args.save_video:
                h, w, _ = self.main_cam_image_history[0].shape
                freq = self.args.control_freq
                main_cam_video = cv2.VideoWriter(os.path.join(save_path, f'episode_{episode_num}_main_cam_video.mp4'), 
                                                 cv2.VideoWriter_fourcc(*'mp4v'), freq, (w, h))
                
                for image in self.main_cam_image_history:
                    # swap back to bgr for opencv
                    image = image[:, :, [2, 1, 0]] 
                    main_cam_video.write(image)
                main_cam_video.release()
                if self.args.use_wrist_camera:
                    h, w, _ = self.wrist_cam_image_history[0].shape
                    freq = self.args.control_freq
                    wrist_cam_video = cv2.VideoWriter(os.path.join(save_path, f'episode_{episode_num}_wrist_cam_video.mp4'), 
                                                    cv2.VideoWriter_fourcc(*'mp4v'), freq, (w, h))
                    for image in self.wrist_cam_image_history:
                        # swap back to bgr for opencv
                        image = image[:, :, [2, 1, 0]] 
                        wrist_cam_video.write(image)
                    wrist_cam_video.release()
            
            self.reset_trajectories()
    
    @property
    def increment_counter(self):
        self.record_episode_counter += 1
        return self.record_episode_counter
    
    def run(self):
        if self.args.use_real_robot:
            head_camera_streaming_thread = threading.Thread(target=self.head_camera_stream_thread, daemon=True)
            head_camera_streaming_thread.start()
            if self.args.use_wrist_camera:
                wrist_camera_streaming_thread = threading.Thread(target=self.wrist_camera_stream_thread, daemon=True)
                wrist_camera_streaming_thread.start()
        rospy.Timer(rospy.Duration(1.0 / self.args.control_freq), self.publish_command)
        rospy.spin()

    def rot_mat_to_rpy_zxy(self, R):
        """
        Convert a rotation matrix (RzRxRy) to Euler angles with ZXY order (first rotate with y, then x, then z).
        This method is more numerically stable, especially near singularities.
        """

        sx = R[2, 1]
        singular_threshold = 1e-6
        cx = np.sqrt(R[2, 0]**2 + R[2, 2]**2)

        if cx < singular_threshold:
            x = np.arctan2(sx, cx)
            y = np.arctan2(R[0, 2], R[0, 0])
            z = 0
        else:
            x = np.arctan2(sx, cx)
            y = np.arctan2(-R[2, 0], R[2, 2])
            z = np.arctan2(-R[0, 1], R[1, 1])

        return np.array([x, y, z])

    def close(self):
        self.shm.close()
        self.shm.unlink()       
        print('clean up shared memory')

def signal_handler(sig, frame):
    print('pressed Ctrl+C! exiting...')
    rospy.signal_shutdown('Ctrl+C pressed')
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(prog="teleoperation with apple vision pro and vuer")
    parser.add_argument("--use_real_robot", type=str2bool, default=False, help="whether to use real robot.")
    parser.add_argument("--teleop_mode", type=int, default=1, help="1=right gripper, 2=left gripper, 3=bimanual")
    parser.add_argument("--head_camera_type", type=int, default=1, help="0=realsense, 1=stereo rgb camera")
    parser.add_argument("--use_wrist_camera", type=str2bool, default=False, help="whether to use wrist camera for real-robot teleop")
    parser.add_argument("--desired_stream_fps", type=int, default=60, help="desired camera streaming fps to vuer")
    parser.add_argument("--control_freq", type=int, default=60, help="control frequency")
    parser.add_argument("--collect_data", type=str2bool, default=False, help="whether to collect data")
    parser.add_argument('--save_video', type=str2bool, default=False, help="whether to collect save videos of camera views when storing the data")
    parser.add_argument("--exp_name", type=str, default='test')

    args = parser.parse_args()
    
    avp_teleoperator = AVPTeleoperator(args)
    try:
        avp_teleoperator.run()
    finally:
        avp_teleoperator.close()









