import argparse
import os
import time

import hydra
from omegaconf import OmegaConf
import rospy
from std_msgs.msg import Float32MultiArray, Int32
import numpy as np
from locoman.config.config import Cfg
from locoman.utilities.orientation_utils_numpy import rot_mat_to_rpy, rpy_to_rot_mat
from sensor_msgs.msg import Image
import ros_numpy
import cv2
import threading
from data_collection.camera_utils import list_video_devices, find_device_path_by_name
import sys
import signal
import h5py
from pynput import keyboard
import torch
from algos.policy import make_policy
from einops import rearrange
import pickle
import yaml
from scipy.spatial.transform import Rotation as R
from locoman.utilities.rotation_interpolation import interpolate_rpy
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../hpt_locoman'))
from hpt.utils import utils as hpt_utils

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Rollout():
    def __init__(self, args, config=None, policy_config=None):
        rospy.init_node('rollout')
        self.args = args
        self.command_publisher = rospy.Publisher(Cfg.commander.auto_policy_topic, Float32MultiArray, queue_size=1)
        self.fsm_publisher = rospy.Publisher(Cfg.fsm_switcher.fsm_state_topic, Int32, queue_size = 1)
        self.fsm_to_teleop_mode_mapping = Cfg.teleoperation.human_teleoperator.fsm_to_teleop_mode_mapping
        self.model_time = 0
        
        self.device = args.device
        self.act_dict = {}
        self.act_dict = {}
        
        if self.args.use_real_robot:
            self.init_cameras()
        else:
            # simulation view subscriber
            self.sim_view_subscriber = rospy.Subscriber(Cfg.teleoperation.teleop_view_topic, Image, self.sim_view_callback)
            img_shape = (Cfg.teleoperation.fpv_height, Cfg.teleoperation.fpv_width, 3)
            self.sim_view_frame = np.zeros(img_shape, dtype=np.uint8)
        # command buffer
        self.command = np.zeros(20)  # body: xyzrpy, eef_r: xyzrpy, eef_l: xyzrpy, grippers: 2 angles
        self.command_msg = Float32MultiArray()
        self.command_msg.data = self.command.tolist()
        self.robot_command = np.zeros(20)
        self.last_command = np.zeros_like(self.command)
        
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
        # self.teleop_mode_subscriber = rospy.Subscriber(Cfg.teleoperation.human_teleoperator.mode_updata_topic, Int32, self.teleop_mode_callback, queue_size=1)
        self.rate = rospy.Rate(200)
        
        self.robot_reset_publisher = rospy.Publisher(Cfg.teleoperation.robot_reset_topic, Int32, queue_size = 1)
        self.robot_reset_msg = Int32()
        self.robot_reset_msg.data = 0

        self.robot_reset_pose_subscriber = rospy.Subscriber(Cfg.commander.robot_reset_pose_topic, Float32MultiArray, self.robot_reset_pose_callback, queue_size=1)
        self.robot_reset_pose = np.zeros(20)
        self.on_reset = False
        self.reset_finished = False

        self.robot_state_subscriber = rospy.Subscriber(Cfg.teleoperation.robot_state_topic, Float32MultiArray, self.robot_proprio_state_callback, queue_size=1)
        
        # self.pinch_gripper_angle_scale = 10.0
        self.pinch_dist_gripper_full_close = 0.02
        self.pinch_dist_gripper_full_open = 0.15
        self.grippr_full_close_angle = Cfg.commander.gripper_angle_range[0]
        self.grippr_full_open_angle = Cfg.commander.gripper_angle_range[1]
        self.eef_xyz_scale = 1.0
        self.manipulate_eef_idx = [0]

        self.agg_modalities = self.args.agg_modalities

        self.init_embodiment_proprio_states()
        self.get_embodiment_masks(embodiment='locoman')
        if not self.args.replay:
            self.init_policy()
        # actions masks
        self.act_mask_dict = {}
        self.act_mask_dict['delta_body_pose'] = torch.tensor(self.act_body_mask).to(self.device).unsqueeze(0)
        self.act_mask_dict['delta_eef_pose'] = torch.tensor(self.act_eef_mask).to(self.device).unsqueeze(0)
        self.act_mask_dict['delta_gripper'] = torch.tensor(self.act_gripper_mask).to(self.device).unsqueeze(0)
        
        self.pause_commands = False

        self.head_cam_image_history = []
        self.wrist_cam_image_history = []
        self.command_history = []
        self.robot_state_history = []
        
        self.rollout_counter = 0
        self.action_idx = 0
        action_horizon = 60
        self.command_traj = np.zeros((action_horizon, 20))
        self.command_trajs = []
        self.infer_interval = self.args.inference_interval
        self.chunk_size = self.args.action_chunk_size
        
        # prepare for data processing and collection
        self.body_xyz_scale = Cfg.teleoperation.human_teleoperator.body_xyz_scale
        self.body_rpy_scale = Cfg.teleoperation.human_teleoperator.body_rpy_scale
        self.eef_xyz_scale = Cfg.teleoperation.human_teleoperator.eef_xyz_scale
        self.eef_rpy_scale = Cfg.teleoperation.human_teleoperator.eef_rpy_scale     
        self.gripper_angle_scale = Cfg.teleoperation.human_teleoperator.gripper_angle_scale
        self.human_command_body_rpy_range = np.array([Cfg.teleoperation.human_teleoperator.body_r_range,
                                                      Cfg.teleoperation.human_teleoperator.body_p_range,
                                                      Cfg.teleoperation.human_teleoperator.body_y_range,])
        
        self.state_flag = 0
    
    def reset_signal_callback(self, msg):
        self.is_changing_reveive_status = True
        if msg.data == 0:
            self.begin_to_receive = False
            self.initial_receive = True
            print("No longer receiving. Initial receive status reset.")
        elif msg.data == 1:
            self.begin_to_receive = True
            self.initial_receive = True
            if self.args.operate_mode == 1:
                # right gripper
                self.fsm_state_msg.data = 3
            elif self.args.operate_mode == 2:
                # left gripper
                self.fsm_state_msg.data = 4
            elif self.args.operate_mode == 3:
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
        # elif msg.data == 2:
        #     self.begin_to_receive = True
        #     self.initial_receive = True
        #     print("Ready to record!")
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
        
    # def teleop_mode_callback(self, msg):
    #     self.is_changing_teleop_mode = True
    #     self.teleoperation_mode = msg.data
    #     print(f"Teleoperation mode updated to {self.teleoperation_mode}.")
    #     self.is_changing_teleop_mode = False

    def sim_view_callback(self, msg):
        self.sim_view_frame = ros_numpy.numpify(msg)
        
    def robot_state_callback(self, msg):
        # update joint positions
        self.robot_state = np.array(msg.data)

    def robot_reset_pose_callback(self, msg):
        self.robot_reset_pose = np.array(msg.data)
        if self.on_reset:
            self.reset_finished = True
            self.on_reset = False
            print('robot reset pose', self.robot_reset_pose)

    def robot_proprio_state_callback(self, msg):
        # update proprio states of locoman
        self.body_pose_proprio_callback = np.array(msg.data)[:6]
        self.eef_pose_proprio_callback = np.array(msg.data)[6:18]
        self.gripper_angle_proprio_callback = np.array(msg.data)[18:20]
        self.joint_pos_proprio_callback = np.array(msg.data)[20:38]
        self.joint_vel_proprio_callback = np.array(msg.data)[38:56]

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
            print('device_path',device_path)
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
                    elapsed_time = time.time() - start_time
                    sleep_time = frame_duration - elapsed_time
            finally:
                self.head_cam_pipeline.stop()
        elif self.args.head_camera_type == 1:
            try:
                while not rospy.is_shutdown():
                    start_time = time.time()
                    ret, frame = self.head_cap.read()
                    # print('frame 0', frame.shape)
                    frame = cv2.resize(frame, (2 * self.head_frame_res[1], self.head_frame_res[0]))
                    # print('frame 1', frame.shape)
                    image_left = frame[:, :self.head_frame_res[1], :]
                    # print('image_left', image_left.shape)
                    image_right = frame[:, self.head_frame_res[1]:, :]
                    # print('image right', image_right.shape)
                    if self.crop_size_w != 0:
                        bgr = np.hstack((image_left[self.crop_size_h:, self.crop_size_w:-self.crop_size_w],
                                        image_right[self.crop_size_h:, self.crop_size_w:-self.crop_size_w]))
                    else:
                        bgr = np.hstack((image_left[self.crop_size_h:, :],
                                        image_right[self.crop_size_h:, :]))

                    self.head_color_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    # print('self.head_color_frame', self.head_color_frame.shape)
                    elapsed_time = time.time() - start_time
                    sleep_time = frame_duration - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    # print(1/(time.time() - start_time))
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
                # print('wrist_color_frame', wrist_color_frame.shape)
                self.wrist_color_frame = cv2.resize(wrist_color_frame, (self.wrist_view_resolution[1], self.wrist_view_resolution[0]))
                # print('self.wrist_color_frame', self.wrist_color_frame.shape)
                elapsed_time = time.time() - start_time
                sleep_time = frame_duration - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # print(1/(time.time() - start_time))
        finally:
            self.head_cap.release()

    def init_policy(self):
        if self.args.policy_type == 0:
            policy_class = 'MXT'
            # toy collect bimanual
            embodiment_config_path = self.args.embodiment_config_path
            transformer_trunk_config_path = self.args.trunk_config_path
            with open(embodiment_config_path, 'r') as f:
                emb_dict = yaml.load(f, Loader=yaml.FullLoader)
            with open(transformer_trunk_config_path, 'r') as f:
                transformer_dict = yaml.load(f, Loader=yaml.FullLoader)
            # a good one (use best checkpoint)
            model_path = self.args.model_path

            stats_path = os.path.join(model_path, 'dataset_stats.pkl')
            with open(stats_path, 'rb') as f:
                stats_dict = pickle.load(f)
            share_cross_attn = False
            policy_config = {
                'norm_stats': stats_dict, 
                "embodiment_args_dict": emb_dict,
                "transformer_args": transformer_dict,
                "share_cross_attn": share_cross_attn,
                "agg_modalities": self.agg_modalities, 
                }
            policy = make_policy(policy_class, policy_config)
            loading_status = policy.deserialize(torch.load(f'{model_path}/policy_last.ckpt', map_location=self.device), eval=True)
            print(f'loaded model from {model_path}')
            self.bc_model = policy

        elif self.args.policy_type == 1 or self.args.policy_type == 2:
            # ACT or HIT
            model_path = self.args.model_path # '/home/yaru/research/locoman_learning/human2locoman/checkpoints_evaluation/pour_bimanual/60trajs/hit/hit_pour_bimanual_bs24_cs180_HIT_resnet18_True'

            config_path = os.path.join(model_path, 'all_configs.json')
            with open(config_path, "r") as f:
                config = json.load(f)
            policy_class = config['policy_class']
            policy_config = config['policy_config']
            stats_path = os.path.join(model_path, 'dataset_stats.pkl')
            with open(stats_path, 'rb') as f:
                stats_dict = pickle.load(f)
            # not just qpos, all the proprio states
            self.pre_process = lambda s_qpos: (s_qpos - stats_dict['qpos_mean']) / stats_dict['qpos_std']
            self.post_process = lambda a: a * stats_dict['action_std'] + stats_dict['action_mean']
            policy = make_policy(policy_class, policy_config)
            loading_status = policy.load_state_dict(torch.load(f'{model_path}/policy_last.ckpt', map_location=self.device))
            policy.eval()
            print(f'loaded model from {model_path}')
            self.bc_model = policy
        else:
            # HPT
            domain = self.args.hpt_domain # 'train_toy_collect_locoman_smaller'
            device = self.device
            
            model_path = self.args.model_path # '/home/yaru/research/locoman_learning/human2locoman/checkpoints_evaluation/toy_collect_single/40trajs/hpt_base'
            
            hpt_config = OmegaConf.load(os.path.join(model_path, 'config.yaml'))
            hpt_config = OmegaConf.structured(hpt_config)
            hpt_config.network["_target_"] = "hpt.models.policy.Policy"
            # initialize policy
            policy = hydra.utils.instantiate(hpt_config.network).to(device)
            policy.init_domain_stem(domain, hpt_config.stem)
            policy.init_domain_head(domain, None, hpt_config.head)
            policy.finalize_modules()
            policy.print_model_stats()
            hpt_utils.set_seed(hpt_config.seed)

            # add encoders into policy parameters
            if hpt_config.network.finetune_encoder:
                hpt_utils.get_image_embeddings(np.zeros((320, 240, 3), dtype=np.uint8), self.args.dataset.image_encoder)
                from hpt.utils.utils import global_vision_model
                policy.init_encoders("image", global_vision_model)

            # load the full model
            policy.load_model(os.path.join(model_path, "model.pth"))
            policy.to(device)
            policy.eval()
            policy.train_mode = False
            self.bc_model = policy
            n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            print(f"number of params (M): {n_parameters / 1.0e6:.2f}")

    def get_embodiment_proprio_state(self, embodiment='locoman', unify_bimanual_body_frame=True):
        if embodiment == 'locoman':
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


    def get_embodiment_masks(self, embodiment='locoman'):
        if embodiment == 'locoman':
            # masks support single gripper manipulation mode for now
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
            self.proprio_gripper_mask = np.array([True, True])
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

    def transform_eef_pose_trajectory(self, eef_pose_history):
        for eef_idx in self.manipulate_eef_idx:
            active_eef_rot_history = eef_pose_history[:, eef_idx*6+3:eef_idx*6+6]
            active_eef_rot_mat_history = R.from_euler('xyz', active_eef_rot_history).as_matrix()
            active_eef_rot_mat_history_new = np.zeros_like(active_eef_rot_mat_history)
            active_eef_rot_mat_history_new[:, :, 0] = active_eef_rot_mat_history[:, :, 1]
            active_eef_rot_mat_history_new[:, :, 1] = -active_eef_rot_mat_history[:, :, 2]
            active_eef_rot_mat_history_new[:, :, 2] = -active_eef_rot_mat_history[:, :, 0]
            active_eef_rot_mat_history = active_eef_rot_mat_history_new
            active_eef_rot_history = R.from_matrix(active_eef_rot_mat_history).as_euler('xyz')
            eef_pose_history[:, eef_idx*6+3:eef_idx*6+6] = active_eef_rot_history
        return eef_pose_history
    
    def transform_robot_eef_pose_to_uni(self, robot_eef_pose):
        for eef_idx in self.manipulate_eef_idx:
            robot_eef_pose_rot_mat = rpy_to_rot_mat(robot_eef_pose[eef_idx+3:eef_idx+6])
            robot_eef_pose_rot_mat_uni = np.zeros_like(robot_eef_pose_rot_mat)
            robot_eef_pose_rot_mat_uni[:, 0] = -robot_eef_pose_rot_mat[:, 2]
            robot_eef_pose_rot_mat_uni[:, 1] = robot_eef_pose_rot_mat[:, 0]
            robot_eef_pose_rot_mat_uni[:, 2] = -robot_eef_pose_rot_mat[:, 1]
            robot_eef_pose_rot_uni = rot_mat_to_rpy(robot_eef_pose_rot_mat_uni)
            robot_eef_pose[eef_idx+3:eef_idx+6] = robot_eef_pose_rot_uni
        return robot_eef_pose
    
    def temporal_ensemble(self, m):
        # temporal ensemble
        if self.rollout_counter < self.chunk_size:
            self.action_idx = self.rollout_counter
        else:
            self.action_idx = self.action_idx + 1
            if (self.rollout_counter - self.chunk_size) % self.infer_interval == 0:
                self.command_trajs = self.command_trajs[1:]
                self.action_idx = self.action_idx - self.infer_interval
        action = np.zeros_like(self.command)
        weight_sum = 0
        for i in range(len(self.command_trajs)):
            if self.action_idx - i * self.infer_interval >= 0:
                action = action + self.command_trajs[i][self.action_idx - i * self.infer_interval] * np.exp(-m*i)
                weight_sum = weight_sum + np.exp(-m*i)
        action = action / weight_sum
        return action

    def model_inference(self):
        # print('freq', 1 / (time.time() - self.model_time))
        # self.model_time = time.time()
        embodiment = 'locoman'
        self.get_embodiment_proprio_state(embodiment=embodiment)
        self.get_embodiment_masks(embodiment='locoman')

        if self.args.policy_type == 0:
            # mxt policy
            ### observations
            obs_dict = {}
            obs_dict['body_pose_state'] = torch.tensor(self.body_pose_proprio).to(self.device).unsqueeze(0)
            obs_dict['eef_pose_state'] = torch.tensor(self.eef_pose_proprio).to(self.device).unsqueeze(0)
            obs_dict['eef_to_body_pose_state'] = torch.tensor(self.eef_to_body_pose_proprio).to(self.device).unsqueeze(0)
            obs_dict['gripper_state'] = torch.tensor(self.gripper_angle_proprio).to(self.device).unsqueeze(0)

            if self.args.use_real_robot:
                main_image = torch.from_numpy(self.head_color_frame).float()
            else:
                main_image = torch.from_numpy(self.sim_view_frame).float()

            main_image.div_(255.0)
            main_image = torch.einsum('h w c -> c h w', main_image)
            obs_dict['main_image'] = main_image.unsqueeze(0)

            if self.args.use_wrist_camera:
                wrist_image = torch.from_numpy(self.wrist_color_frame).float()
                wrist_image.div_(255.0)
                wrist_image = torch.einsum('h w c -> c h w', wrist_image)
                obs_dict['wrist_image'] = wrist_image.unsqueeze(0)

            ### observation masks
            obs_mask_dict = {}
            obs_mask_dict['main_image'] = torch.tensor(self.img_main_mask).to(self.device).unsqueeze(0)
            if self.args.use_wrist_camera:
                obs_mask_dict['wrist_image'] = torch.tensor(self.img_wrist_mask).to(self.device).unsqueeze(0)
            obs_mask_dict['body_pose_state'] = torch.tensor(self.proprio_body_mask).to(self.device).unsqueeze(0)
            obs_mask_dict['eef_pose_state'] = torch.tensor(self.proprio_eef_mask).to(self.device).unsqueeze(0)
            obs_mask_dict['eef_to_body_pose_state'] = torch.tensor(self.proprio_eef_mask).to(self.device).unsqueeze(0)
            obs_mask_dict['gripper_state'] = torch.tensor(self.proprio_gripper_mask).to(self.device).unsqueeze(0)
            if not self.agg_modalities:
                self.act_dict = self.bc_model(obs_dict, embodiment, obs_mask_dict)
            else:
                main_image = obs_dict['main_image']
                main_image_mask = obs_mask_dict['main_image']
                _, c, h, w = main_image.shape
                main_image_left = main_image[:, :, :, :w//2]
                main_image_right = main_image[:, :, :, w//2:]
                if 'wrist_image' in obs_dict:     
                    wrist_image = obs_dict['wrist_image']
                    wrist_image_mask = obs_mask_dict['wrist_image']
                    all_images = torch.concat([main_image_left, main_image_right, wrist_image], dim=-1)
                    all_images_mask = wrist_image_mask
                else:
                    all_images = torch.concat([main_image_left, main_image_right], dim=-1)
                    all_images_mask = main_image_mask
                agg_obs_dict = {'all_images': all_images}
                agg_obs_mask_dict = {'all_images': all_images_mask}
                all_proprio = torch.concat([obs_dict[key] for key in ["body_pose_state", "eef_pose_state", "eef_to_body_pose_state", "gripper_state"]], dim=1)
                all_proprio_mask = torch.concat([obs_mask_dict[key] for key in ["body_pose_state", "eef_pose_state", "eef_to_body_pose_state", "gripper_state"]], dim=1)
                agg_obs_dict.update({'all_proprio_states': all_proprio})
                agg_obs_mask_dict.update({'all_proprio_states': all_proprio_mask})
                self.act_dict = self.bc_model(agg_obs_dict, embodiment, agg_obs_mask_dict)

            if self.agg_modalities:
                all_actions = self.act_dict['all_actions'].squeeze(0).cpu().detach().numpy()
                if self.args.action_type == 0:
                    body_pose_traj = all_actions[:, 12:18]
                    eef_pose_traj = all_actions[:, :12]
                    eef_pose_traj = self.transform_eef_pose_trajectory(eef_pose_traj)
                    gripper_angle_traj = all_actions[:, 18:20]
                elif self.args.action_type == 1:
                    delta_body_pose_traj = all_actions[:, 12:18]
                    delta_eef_pose_traj = all_actions[:, :12]
                    delta_gripper_traj = all_actions[:, 18:20]
                    body_pose_traj = np.cumsum(delta_body_pose_traj, axis=0) + self.robot_reset_pose[:6]
                    reset_eef_pose_uni = self.transform_robot_eef_pose_to_uni(self.robot_reset_pose[6:18])
                    eef_pose_traj = np.cumsum(delta_eef_pose_traj, axis=0) + reset_eef_pose_uni
                    eef_pose_traj = self.transform_eef_pose_trajectory(eef_pose_traj)
                    gripper_angle_traj = np.cumsum(delta_gripper_traj, axis=0) + self.robot_reset_pose[18:]
                else:
                    # mixed action: absolute + relative to the chunk start
                    body_pose_traj = all_actions[:, 12:18]
                    body_pose_traj[1:] = body_pose_traj[1:] + body_pose_traj[0]
                    eef_pose_traj = all_actions[:, :12]
                    eef_pose_traj[1:] = eef_pose_traj[1:] + eef_pose_traj[0]
                    eef_pose_traj = self.transform_eef_pose_trajectory(eef_pose_traj)
                    gripper_angle_traj = all_actions[:, 18:20]
                    gripper_angle_traj[1:] = gripper_angle_traj[1:] + gripper_angle_traj[0]
            else:
                if self.args.action_type == 0:
                    # absolute action
                    body_pose_traj = self.act_dict['body_pose'].squeeze(0).cpu().detach().numpy()
                    eef_pose_traj = self.act_dict['eef_pose'].squeeze(0).cpu().detach().numpy()
                    eef_pose_traj = self.transform_eef_pose_trajectory(eef_pose_traj)
                    gripper_angle_traj = self.act_dict['gripper'].squeeze(0).cpu().detach().numpy()
                elif self.args.action_type == 1:
                    # delta action
                    delta_body_pose_traj = self.act_dict['body_pose'].squeeze(0).cpu().detach().numpy()
                    delta_eef_pose_traj = self.act_dict['eef_pose'].squeeze(0).cpu().detach().numpy()
                    delta_gripper_traj = self.act_dict['gripper'].squeeze(0).cpu().detach().numpy()

                    body_pose_traj = np.cumsum(delta_body_pose_traj, axis=0) + self.robot_reset_pose[:6]
                    reset_eef_pose_uni = self.transform_robot_eef_pose_to_uni(self.robot_reset_pose[6:18])
                    eef_pose_traj = np.cumsum(delta_eef_pose_traj, axis=0) + reset_eef_pose_uni
                    eef_pose_traj = self.transform_eef_pose_trajectory(eef_pose_traj)
                    gripper_angle_traj = np.cumsum(delta_gripper_traj, axis=0) + self.robot_reset_pose[18:]
                else:
                    # mixed action: absolute + relative to the chunk start
                    body_pose_traj = self.act_dict['body_pose'].squeeze(0).cpu().detach().numpy()
                    body_pose_traj[1:] = body_pose_traj[1:] + body_pose_traj[0]
                    eef_pose_traj = self.act_dict['eef_pose'].squeeze(0).cpu().detach().numpy()
                    eef_pose_traj[1:] = eef_pose_traj[1:] + eef_pose_traj[0]
                    eef_pose_traj = self.transform_eef_pose_trajectory(eef_pose_traj)
                    gripper_angle_traj = self.act_dict['gripper'].squeeze(0).cpu().detach().numpy()
                    gripper_angle_traj[1:] = gripper_angle_traj[1:] + gripper_angle_traj[0]
                    
            self.command_traj = np.concatenate((body_pose_traj * self.act_body_mask,
                                                eef_pose_traj * self.act_eef_mask,
                                                gripper_angle_traj * self.act_gripper_mask), axis=1)
            self.command_trajs.append(self.command_traj)
            
        elif self.args.policy_type == 1 or self.args.policy_type == 2:
            proprio_state = np.concatenate([self.body_pose_proprio, self.eef_pose_proprio, self.eef_to_body_pose_proprio, self.gripper_angle_proprio])
            proprio_state = self.pre_process(proprio_state)
            proprio_state = torch.from_numpy(proprio_state).float().to(self.device).unsqueeze(0)
            curr_images = []
            if self.args.use_real_robot:
                main_image = rearrange(self.head_color_frame, 'h w c -> c h w')
            else:
                main_image = rearrange(self.sim_view_frame, 'h w c -> c h w')
            curr_images.append(main_image)
            if self.args.use_wrist_camera:
                wrist_image = rearrange(self.wrist_color_frame, 'h w c -> c h w')
                curr_images.append(wrist_image)
            with torch.no_grad():
                main_image = torch.from_numpy(main_image / 255.0).float().to(self.device).unsqueeze(0)
                main_image_left = main_image[:, :, :, :main_image.shape[3]//2]
                main_image_right = main_image[:, :, :, main_image.shape[3]//2:]
                if self.args.use_wrist_camera:
                    wrist_image = torch.from_numpy(wrist_image / 255.0).float().to(self.device).unsqueeze(0)
                    curr_image = torch.stack([main_image_left, main_image_right, wrist_image], dim=1)
                else:
                    curr_image = torch.stack([main_image_left, main_image_right], dim=1)
                actions = self.bc_model(proprio_state, curr_image).squeeze(0).cpu().detach().numpy()
            # absolute action
            actions = self.post_process(actions)
            body_pose_traj = actions[:, :6]
            eef_pose_traj = actions[:, 6:18]
            eef_pose_traj = self.transform_eef_pose_trajectory(eef_pose_traj)
            gripper_angle_traj = actions[:, 18:20]
            self.command_traj = np.concatenate((body_pose_traj * self.act_body_mask,
                                                eef_pose_traj * self.act_eef_mask,
                                                gripper_angle_traj * self.act_gripper_mask), axis=1)
            self.command_trajs.append(self.command_traj)
        else:
            # HPT
            domain = self.args.hpt_domain # change to the dataset name used in training
            proprio_state = np.concatenate([self.body_pose_proprio, self.eef_pose_proprio, self.gripper_angle_proprio, self.eef_to_body_pose_proprio])
            if self.args.use_real_robot:
                main_image = self.head_color_frame
            else:
                main_image = self.sim_view_frame
            if self.args.use_wrist_camera:
                wrist_image = self.wrist_color_frame
            else:
                wrist_image = main_image
            obs_dict = {'state': proprio_state,
                        'main_image': main_image,
                        'wrist_image': wrist_image}
            actions = self.bc_model.get_action(obs_dict, domain)
            body_pose_traj = actions[:, :6]
            eef_pose_traj = actions[:, 6:18]
            eef_pose_traj = self.transform_eef_pose_trajectory(eef_pose_traj)
            gripper_angle_traj = actions[:, 18:20]
            self.command_traj = np.concatenate((body_pose_traj * self.act_body_mask,
                                                eef_pose_traj * self.act_eef_mask,
                                                gripper_angle_traj * self.act_gripper_mask), axis=1)
            self.command_trajs.append(self.command_traj)

    def load_trajectory(self):
        path = '/home/yaru/research/locoman_learning/human2locoman/demonstrations/bimanual_test/locoman/20250106_124043/episode_1.hdf5'
        with h5py.File(path, 'a') as f:
            body_pose_history = np.array(f['actions/body'])
            delta_body_pose_history = np.array(f['actions/delta_body'])
            eef_pose_history = np.array(f['actions/eef'])
            delta_eef_pose_history = np.array(f['actions/delta_eef'])
            gripper_angle_history = np.array(f['actions/gripper'])
            delta_gripper_angle_history = np.array(f['actions/delta_gripper'])
            act_body_mask = np.array(f['masks/act_body'])
            act_eef_mask = np.array(f['masks/act_eef'])
            act_gripper_mask = np.array(f['masks/act_gripper'])
            
        # body_pose_history_d = np.cumsum(delta_body_pose_history, axis=0) + body_pose_history[0]
        # eef_pose_history_d = np.cumsum(delta_eef_pose_history, axis=0) + eef_pose_history[0]
        # gripper_angle_history_d = np.cumsum(delta_gripper_angle_history, axis=0) + gripper_angle_history[0]
        
        body_pose_history_d = np.cumsum(delta_body_pose_history, axis=0) + self.robot_reset_pose[:6]
        # robot_eef_pose_uni = self.transform_robot_eef_pose_to_uni(self.robot_reset_pose[6:18])
        # eef_pose_history_d = np.cumsum(delta_eef_pose_history, axis=0) + robot_eef_pose_uni
        eef_pose_history_d = np.cumsum(delta_eef_pose_history, axis=0) + eef_pose_history[0]
        gripper_angle_history_d = np.cumsum(delta_gripper_angle_history, axis=0) + self.robot_reset_pose[18:]
        
        eef_pose_history = self.transform_eef_pose_trajectory(eef_pose_history)
        eef_pose_history_d = self.transform_eef_pose_trajectory(eef_pose_history_d)

        if self.args.action_type == 0:
            command_history = np.concatenate((body_pose_history * act_body_mask, 
                                            eef_pose_history * act_eef_mask, 
                                            gripper_angle_history * act_gripper_mask), axis=1)
        else:
            command_history = np.concatenate((body_pose_history_d * act_body_mask, 
                                                eef_pose_history_d * act_eef_mask, 
                                                gripper_angle_history_d * act_gripper_mask), axis=1)
            
        return command_history
        
    def publish_command(self, event=None):
        # publish teleop commands at a fixed rate 
        # need to check the duration to finish one execution and if a separate thread is needed: takes 0.0002-0.0003s
        # print('freq', 1 / (time.time() - self.model_time))
        # self.model_time = time.time()

        self.update_manipulate_eef_idx()
        # reset the robot
        if self.begin_to_receive:
            if self.state_flag == 1:
                self.state_flag = 0
                self.robot_reset_publisher.publish(self.robot_reset_msg)
                self.on_reset = True
                self.reset_finished = False
                self.initial_receive = True
                self.rollout_counter = 0
                self.command_trajs = []
                print("reset robot")
        self.command[:] = 0
        
        if self.initial_receive and self.begin_to_receive and self.reset_finished:
            # initialize and start rollout
            if self.state_flag == 2:
                print('start rollout')
                if self.args.replay:
                    self.command_history = self.load_trajectory()
                self.state_flag = 0
            else:
                return
            
            self.command = self.robot_reset_pose.copy()
            self.last_command = self.command.copy()
            self.command_msg.data = self.command.tolist()
            self.command_publisher.publish(self.command_msg)
            self.initial_receive = False
        # rollout
        elif self.begin_to_receive and self.reset_finished:
            if self.state_flag == 3:
                self.state_flag = 0
                self.pause_commands = True
                print("Pause sending commands")
                return
            if self.pause_commands and self.state_flag == 2:
                self.state_flag = 0
                print("Restart sending commands")
                self.pause_commands = False
            if self.pause_commands:
                return

            if self.args.replay:
                if self.rollout_counter < self.command_history.shape[0]:
                    self.command = self.command_history[self.rollout_counter]
                    self.rollout_counter += 1
                    self.command_msg.data = self.command.tolist()
                    self.command_publisher.publish(self.command_msg)
            else:
                if self.rollout_counter % self.infer_interval == 0:
                    self.model_inference()
                if self.args.temporal_ensemble:
                    self.command = self.temporal_ensemble(m=0.1)
                else:
                    self.command = self.command_traj[self.rollout_counter % self.infer_interval]
                self.command = self.low_pass_filter(alpha=1)
                self.last_command = self.command.copy()
                self.rollout_counter += 1
                self.command_msg.data = self.command.tolist()
                self.command_publisher.publish(self.command_msg)

    def low_pass_filter(self, alpha=1.0):
        smooth_action = self.command * alpha + self.last_command * (1 - alpha)
        cur_body_rpy = self.command[3:6]
        cur_right_eef_rpy = self.command[9:12]
        cur_left_eef_rpy = self.command[15:18]
        last_body_rpy = self.last_command[3:6]
        last_right_eef_rpy = self.last_command[9:12]
        last_left_eef_rpy = self.last_command[15:18]
        smooth_body_rpy = interpolate_rpy(last_body_rpy, cur_body_rpy, alpha)
        smooth_right_eef_rpy = interpolate_rpy(last_right_eef_rpy, cur_right_eef_rpy, alpha)
        smooth_left_eef_rpy = interpolate_rpy(last_left_eef_rpy, cur_left_eef_rpy, alpha)
        smooth_action[3:6] = smooth_body_rpy
        smooth_action[9:12] = smooth_right_eef_rpy
        smooth_action[15:18] = smooth_left_eef_rpy
        return smooth_action

    def on_press_key(self, key):
        try:
            if self.begin_to_receive:
                if key.char == '1':
                    self.state_flag = 1
            if self.initial_receive and self.begin_to_receive and self.reset_finished:
                if key.char == '2':
                    self.state_flag = 2
            elif self.begin_to_receive and self.reset_finished:
                if key.char == '3':
                    self.state_flag = 3
                if key.char == '2':
                    self.state_flag = 2
        except AttributeError:
            # Handle special keys (like function keys, arrow keys, etc.)
            pass  #
                
    def keyboard_listener_thread(self):
        with keyboard.Listener(on_press=self.on_press_key) as listener:
            listener.join()
    
    def run(self):
        if self.args.use_real_robot:
            head_camera_streaming_thread = threading.Thread(target=self.head_camera_stream_thread, daemon=True)
            head_camera_streaming_thread.start()
            if self.args.use_wrist_camera:
                wrist_camera_streaming_thread = threading.Thread(target=self.wrist_camera_stream_thread, daemon=True)
                wrist_camera_streaming_thread.start()
        # model_inference_thread = threading.Thread(target=self.model_inference, daemon=True)
        # model_inference_thread.start()
        keyboard_listener_thread = threading.Thread(target=self.keyboard_listener_thread, daemon=True)
        keyboard_listener_thread.start()

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
            z = 0                # self.command = self.low_pass_filter(alpha=1)

        else:
            x = np.arctan2(sx, cx)
            y = np.arctan2(-R[2, 0], R[2, 2])
            z = np.arctan2(-R[0, 1], R[1, 1])

        return np.array([x, y, z])

    def close(self):   
        print('close rollout')

def signal_handler(sig, frame):
    print('pressed Ctrl+C! exiting...')
    rospy.signal_shutdown('Ctrl+C pressed')
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(prog="teleoperation with apple vision pro and vuer")
    # command publish rate
    # - on hand move and on cam move frequence, important
    parser.add_argument("--use_real_robot", type=str2bool, default=False, help="whether to use real robot.") #
    parser.add_argument("--operate_mode", type=int, default=1, help="1=right gripper, 2=left gripper, 3=bimanual") #
    parser.add_argument("--head_camera_type", type=int, default=1, help="0=realsense, 1=stereo rgb camera")
    parser.add_argument("--use_wrist_camera", type=str2bool, default=False, help="whether to use wrist camera for real-robot teleop.") #
    parser.add_argument("--desired_stream_fps", type=int, default=60, help="desired camera streaming fps to vuer")
    parser.add_argument("--control_freq", type=int, default=60, help="control frequency")
    parser.add_argument("--temporal_ensemble", type=str2bool, default=False, help="whether to use temporal ensemble.")
    parser.add_argument("--inference_interval", type=int, default=60, help="model inference interval of publishing commands") #
    parser.add_argument("--action_chunk_size", type=int, default=60, help="action chunk size") #
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--action_type", type=int, default=0, help="0=absolute, 1=delta, 2=mixed")
    parser.add_argument("--policy_type", type=int, default=0, help="0=mxt, 1=act, 2=hit, others=hpt") #
    parser.add_argument("--agg_modalities", action="store_true", help="whether to use aggregated modalities for the model")
    parser.add_argument("--replay", type=str2bool, default=False, help="if replay a trajectory")
    
    parser.add_argument("--embodiment_config_path", type=str, default='algos/detr/models/mxt_definitions/configs/embodiments.yaml', help="path to the embodiment config file")
    parser.add_argument("--trunk_config_path", type=str, default='algos/detr/models/mxt_definitions/configs/transformer_trunk.yaml', help="path to the trunk config file")
    parser.add_argument("--model_path", type=str, default='', help="path to the model")
    parser.add_argument("--hpt_domain", type=str, default='train_toy_collect_locoman_smaller', help="domain for hpt policy, should be the same as the dataset name used")

    args, unknown_args = parser.parse_known_args()
    
    rollout = Rollout(args)
    try:
        rollout.run()
    finally:
        rollout.close()









