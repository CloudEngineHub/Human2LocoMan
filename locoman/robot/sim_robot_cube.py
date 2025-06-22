import time
from matplotlib import pyplot as plt
from config.config import Cfg
from robot.base_robot import BaseRobot
from robot.motors import MotorCommand, MotorControlMode
from isaacgym import gymapi, gymtorch
from utilities.rotation_utils import rpy_vel_to_skew_synmetric_mat
import numpy as np
import torch
import os
import sys



class SimRobot(BaseRobot):
    def __init__(self, cfg: Cfg, sim, viewer):
        super().__init__(cfg)
        self._gym = gymapi.acquire_gym()
        self._sim = sim
        self._viewer = viewer
        self._max_reward = self._cfg.reward.max_reward
        self._sim_conf = self._cfg.get_sim_config()
        self._init_simulator()
        self._init_buffers()
        self.first_reset = True
        print("env init complete!")

    def _init_simulator(self):
        self._prepare_initial_state()
        self._create_envs()
        self._gym.prepare_sim(self._sim)
        self.realtime_img = None
    
    def _prepare_initial_state(self):
        base_init_state_list = self._cfg.init_state.pos + self._cfg.init_state.rot + self._cfg.init_state.lin_vel + self._cfg.init_state.ang_vel
        base_init_states = np.stack([base_init_state_list] * self._num_envs, axis=0)
        self._base_init_state = torch.tensor(base_init_states, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_init_pos = self._motors.init_positions.to(self._device)
        if "cuda" in self._device:
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)

    def sample_pos(self, pos_range):
        x_range = pos_range[0]
        y_range = pos_range[1]
        z_range = pos_range[2]
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        z = np.random.uniform(z_range[0], z_range[1])
        return [x, y, z]
    
    def quat_product(self, q, p):
        w = q.w*p.w - q.x*p.x - q.y*p.y - q.z*p.z
        x = q.w*p.x + q.x*p.w + q.y*p.z + q.z*p.y
        y = q.w*p.y + q.y*p.w - q.x*p.z + q.z*p.x
        z = q.w*p.z + q.z*p.w + q.x*p.y - q.y*p.x
        return w,x,y,z
    
    def _create_envs(self):
        # Load robot asset
        urdf_path = self._cfg.asset.urdf_path
        asset_root = os.path.dirname(urdf_path)
        asset_file = os.path.basename(urdf_path)
        asset_config = self._cfg.get_asset_config()
        self._robot_asset = self._gym.load_asset(self._sim, asset_root, asset_file, asset_config.asset_options)
        
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.fix_base_link = False
        cube_asset_options.density = 0.1
        self._cube_asset = self._gym.create_box(self._sim, 0.1, 0.1, 0.1, cube_asset_options)
        cube_color = gymapi.Vec3(0.8, 0.3, 0.3)
        
        target_asset_options = gymapi.AssetOptions()
        target_asset_options.fix_base_link = True
        target_asset_options.density = 0.1
        target_asset_options.disable_gravity = True
        self._target_asset = self._gym.create_box(self._sim, 0.1, 0.1, 0.1, target_asset_options)
        target_color = gymapi.Vec3(0.3, 0.8, 0.3)
        
        # cam_marker_asset_options = gymapi.AssetOptions()
        # cam_marker_asset_options.fix_base_link = True
        # cam_marker_asset_options.density = 0.1
        # cam_marker_asset_options.disable_gravity = True
        # self._cam_marker_asset = self._gym.create_capsule(self._sim, 0.01, 0.01, cam_marker_asset_options)
        # cam_marker_color = gymapi.Vec3(0.3, 0.3, 0.8)
        
        # Create envs and actors
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self._envs = []
        self._actors = []
        self._cubes = []
        self._targets = []
        self._cameras = []
        self.cube_pos_offset_range = [[0.36, 0.40], [-0.12, -0.04], [0.05001, 0.05005]]
        self.target_pos_offset_range = [[0.40, 0.48], [-0.05, 0.05], [0.05001, 0.05005]]
        for i in range(self._num_envs):
            env_handle = self._gym.create_env(self._sim, env_lower, env_upper, int(np.sqrt(self._num_envs)))
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(*self._base_init_state[i, :3])
            cube_start_pose = gymapi.Transform()
            cube_start_pose.p = gymapi.Vec3(*self.sample_pos(self.cube_pos_offset_range))
            target_start_pose = gymapi.Transform()
            target_start_pose.p = gymapi.Vec3(*self.sample_pos(self.target_pos_offset_range))
            actor_handle = self._gym.create_actor(env_handle, self._robot_asset, start_pose, "actor",
                                                  i, asset_config.self_collisions, 0)
            cube_handle = self._gym.create_actor(env_handle, self._cube_asset, cube_start_pose, "cube",
                                                 i, False, 0)
            target_handle = self._gym.create_actor(env_handle, self._target_asset, target_start_pose, "target",
                                                   i + self._num_envs, False, 0)
            self._gym.set_rigid_body_color(env_handle, cube_handle, 0, gymapi.MESH_VISUAL, cube_color)
            self._gym.set_rigid_body_color(env_handle, target_handle, 0, gymapi.MESH_VISUAL, target_color)
            self._gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self._envs.append(env_handle)
            self._actors.append(actor_handle)
            self._cubes.append(cube_handle)
            self._targets.append(target_handle)
            
            # cameras
            camera_props = gymapi.CameraProperties()
            camera_props.width = 660
            camera_props.height = 420
            camera_props.horizontal_fov = 110.0
            # camera_props.vertical_fov = 70.0
            camera_handle = self._gym.create_camera_sensor(env_handle, camera_props)
            self._cameras.append(camera_handle) 
            wrist_camera_handle = self._gym.create_camera_sensor(env_handle, camera_props)
            self._cameras.append(wrist_camera_handle)
            
            base_body_handle = self._gym.find_actor_rigid_body_handle(env_handle, actor_handle, "base")
            eef_wrist_handle = self._gym.find_actor_rigid_body_handle(env_handle, actor_handle, "1_FR_link_3")
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(0.29, 0.0, 0.11)
            local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 1.0, 0.0), np.pi / 2.8)
            wrist_transform = gymapi.Transform()
            wrist_transform.p = gymapi.Vec3(0.005, -0.03, -0.025)
            rot1 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 1.0, 0.0), np.pi / 2)
            rot2 = gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), 8 * np.pi / 9)
            wrist_transform.r = gymapi.Quat(*self.quat_product(rot2, rot1))
            # local_transform.p = gymapi.Vec3(0.15, 0., 0.45)
            # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 1.0, 0.0), np.pi / 3.0)
            self._gym.attach_camera_to_body(camera_handle, env_handle, base_body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            self._gym.attach_camera_to_body(wrist_camera_handle, env_handle, eef_wrist_handle, wrist_transform, gymapi.FOLLOW_TRANSFORM)
            
            
    def _init_buffers(self):
        # Rigid body indices within all rigid bodies
        self._torso_indices = torch.zeros(len(self._torso_names), dtype=torch.long, device=self._device, requires_grad=False)
        self._hip_indices = torch.zeros(len(self._hip_names), dtype=torch.long, device=self._device, requires_grad=False)
        self._thigh_indices = torch.zeros(len(self._thigh_names), dtype=torch.long, device=self._device, requires_grad=False)
        self._calf_indices = torch.zeros(len(self._calf_names), dtype=torch.long, device=self._device, requires_grad=False)
        self._foot_indices = torch.zeros(self._num_legs, dtype=torch.long, device=self._device, requires_grad=False)
        self._swing_foot_indices = torch.zeros(len(self._swing_foot_names), dtype=torch.long, device=self._device, requires_grad=False)
        self._stance_foot_indices = torch.zeros(len(self._stance_foot_names), dtype=torch.long, device=self._device, requires_grad=False)

        # Extract indices of different bodies
        for i in range(len(self._torso_names)):
            self._torso_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actors[0], self._torso_names[i])
        for i in range(len(self._hip_names)):
            self._hip_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actors[0], self._hip_names[i])
        for i in range(len(self._thigh_names)):
            self._thigh_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actors[0], self._thigh_names[i])
        for i in range(len(self._calf_names)):
            self._calf_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actors[0], self._calf_names[i])
        stance_idx = 0
        for i in range(self._num_legs):
            self._foot_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actors[0], self._foot_names[i])
            for swing_foot_idx in range(len(self._swing_foot_names)):
                if self._swing_foot_names[swing_foot_idx] in self._foot_names[i]:
                    self._swing_foot_indices[swing_foot_idx] = self._foot_indices[i]
                    break
                elif swing_foot_idx == len(self._swing_foot_names) - 1:
                    self._stance_foot_indices[stance_idx] = self._foot_indices[i]
                    stance_idx += 1
        
        # Get gym GPU state tensors
        actor_root_state = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        net_contact_forces = self._gym.acquire_net_contact_force_tensor(self._sim)
        rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)
        jacobians = self._gym.acquire_jacobian_tensor(self._sim, "actor")
        massmatrix = self._gym.acquire_mass_matrix_tensor(self._sim, "actor")
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_jacobian_tensors(self._sim)
        self._gym.refresh_mass_matrix_tensors(self._sim)

        # Robot state buffers
        # root state
        self.all_root_state = gymtorch.wrap_tensor(actor_root_state).view(self._num_envs, -1, 13)
        self._base_pos_sim = self.all_root_state[:, 0, 0:3]
        self._base_quat_sim2b = self.all_root_state[:, 0, 3:7]
        self._base_lin_vel_sim = self.all_root_state[:, 0, 7:10]
        self._base_ang_vel_sim = self.all_root_state[:, 0, 10:13]
        # self._base_rot_mat_w2b = quat_to_rot_mat(self._base_quat_w2b)
        # self._base_rot_mat_b2w = torch.transpose(self._base_rot_mat_w2b, 1, 2)

        # dof state
        self._num_joints = self._gym.get_asset_dof_count(self._robot_asset)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._joint_pos = self._dof_state.view(self._num_envs, self._num_joints, 2)[..., 0]
        self._joint_vel = self._dof_state.view(self._num_envs, self._num_joints, 2)[..., 1]
        # force state
        self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self._num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self._foot_contact = self._contact_forces[:, self._foot_indices, 2] > 1.0
        self._jacobian_sim = gymtorch.wrap_tensor(jacobians)
        self._mass_matrix = gymtorch.wrap_tensor(massmatrix)
        self._gravity_vec = torch.stack([torch.tensor([0., 0., 1.], dtype=torch.float, device=self._device, requires_grad=False)] * self._num_envs)
        self._projected_gravity = torch.bmm(self._base_rot_mat_b2w, self._gravity_vec[:, :, None])[:, :, 0]
        # rigid body state
        self._num_bodies = self._gym.get_asset_rigid_body_count(self._robot_asset)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self._num_envs * self._num_bodies, :].view(self._num_envs, self._num_bodies, 13)
        self._foot_pos_sim = self._rigid_body_state[:, self._foot_indices, 0:3]
        self._foot_vel_sim = self._rigid_body_state[:, self._foot_indices, 7:10]
        # self._swing_foot_pos_w = self._rigid_body_state.view(self._num_envs, self._num_bodies, 13)[:, self._swing_foot_indices, 0:3]
        # self._swing_foot_vel_w = self._rigid_body_state.view(self._num_envs, self._num_bodies, 13)[:, self._swing_foot_indices, 7:10]
        # self._stance_foot_pos_w = self._rigid_body_state.view(self._num_envs, self._num_bodies, 13)[:, self._stance_foot_indices, 0:3]
        # self._stance_foot_vel_w = self._rigid_body_state.view(self._num_envs, self._num_bodies, 13)[:, self._stance_foot_indices, 7:10]

        if self._use_gripper:
            self._eef_indices = torch.zeros(len(self._eef_names), dtype=torch.long, device=self._device, requires_grad=False)
            for i in range(len(self._eef_names)):
                self._eef_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actors[0], self._eef_names[i])
            self._eef_pos_sim = self._rigid_body_state[:, self._eef_indices, 0:3]
            self._eef_quat_sim = self._rigid_body_state[:, self._eef_indices, 3:7]

        # Other useful buffers
        self._torques = torch.zeros(self._num_envs, self._num_joints, dtype=torch.float, device=self._device, requires_grad=False)

        # print(f"Number of dofs: {self._num_joints}")
        # print(f"Number of bodies: {self._num_bodies}")
        # print(f"torso indices: {self._torso_indices}")
        # print(f"hip indices: {self._hip_indices}")
        # print(f"thigh indices: {self._thigh_indices}")
        # print(f"calf indices: {self._calf_indices}")
        # print(f"feet indices: {self._foot_indices}")
        # print('swing_feet_indices: ', self._swing_foot_indices)
        # print('stance_feet_indices: ', self._stance_foot_indices)

        # The origins for each environment
        num_cols = np.floor(np.sqrt(self._num_envs))
        num_rows = np.ceil(self._num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing='ij')
        spacing = self._sim_conf.env_spacing
        self._env_origins = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._env_origins[:, 0] = spacing * xx.to(self._device).flatten()[0:self._num_envs]
        self._env_origins[:, 1] = spacing * yy.to(self._device).flatten()[0:self._num_envs]
        self._env_origins[:, 2] = 0.

        # useful buffers
        if self._cfg.sim.show_gui:
            self.first_render = True
        self.log_one_time = True
        self.finish_reset = False

    def reset(self):
        self.finish_reset = False
        self._reset_idx(torch.arange(self._num_envs, device=self._device))
        self.finish_reset = True
        
        rgb_image = self._gym.get_camera_image(self._sim, self._envs[0], self._cameras[0], gymapi.IMAGE_COLOR)  # this has 4 channels: RGBA
        rgb_image = rgb_image.reshape(rgb_image.shape[0], -1, 4)[:,:,:3]
        wrist_rgb_image = self._gym.get_camera_image(self._sim, self._envs[0], self._cameras[1], gymapi.IMAGE_COLOR)  # this has 4 channels: RGBA
        wrist_rgb_image = wrist_rgb_image.reshape(wrist_rgb_image.shape[0], -1, 4)[:,:,:3]
        
        obs_dict = {
            'qpos': self._joint_pos[0],
            'eef': self._rigid_body_state[0, self._eef_indices[0], :], # right eef pos & quat
            'images': {
                'main': rgb_image,
                'wrist': wrist_rgb_image
            }
        }
        
        self.first_reset = False
        
        return {'observation': obs_dict,
                'reward': 0.,
                } 
    
    def _reset_idx(self, env_ids: torch.Tensor): # TODO: reset cube and target pose!
        # if len(env_ids) == 0:
        #     return
        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        # cube_env_ids = env_ids + self._num_envs
        # target_env_ids = env_ids + 2 * self._num_envs
        # actor_ids_int32 = torch.cat([env_ids, cube_env_ids, target_env_ids], dim=-1).to(dtype=torch.int32)
        # # env_ids_int32 = torch.cat([env_ids, cube_env_ids], dim=-1).to(dtype=torch.int32)
        # # env_ids_int32 = env_ids.to(dtype=torch.int32)
        # # Reset root state:
        # self.all_root_state[env_ids, 0] = self._base_init_state[env_ids]
        # self.all_root_state[env_ids, 0, :3] += self._env_origins[env_ids]
        # # cubes
        # self.all_root_state[env_ids, 1] = torch.zeros(len(env_ids), 13).to(self._device)
        # self.all_root_state[env_ids, 1, 3:7] = torch.tensor([0., 0., 0., 1.]).to(self._device)
        # self.all_root_state[env_ids, 1, :3] = torch.tensor([self.sample_pos(self.cube_pos_offset_range) for _ in range(len(env_ids))]).to(self._device)
        # self.all_root_state[env_ids, 1, :3] += self._env_origins[env_ids]
        # # self.all_root_state[env_ids, 1, 2] += 0.2
        # # targets
        # self.all_root_state[env_ids, 2] = torch.zeros(len(env_ids), 13).to(self._device)
        # self.all_root_state[env_ids, 2, 3:7] = torch.tensor([0., 0., 0., 1.]).to(self._device)
        # self.all_root_state[env_ids, 2, :3] = torch.tensor([self.sample_pos(self.target_pos_offset_range) for _ in range(len(env_ids))]).to(self._device)
        # self.all_root_state[env_ids, 2, :3] += self._env_origins[env_ids]
        
        # self._gym.set_actor_root_state_tensor_indexed(self._sim, gymtorch.unwrap_tensor(self.all_root_state.view(-1, 13)),
        #                                               gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))
        # # Reset dofs
        # self._joint_pos[env_ids] = self._joint_init_pos
        # self._joint_vel[env_ids] = 0.
        # self._gym.set_dof_state_tensor_indexed(self._sim, gymtorch.unwrap_tensor(self._dof_state),
        #                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # self._init_buffers()
        
        # # If reset all enviroments
        # if len(env_ids) == self._num_envs:
        #     # self._stablize_the_robot()
        #     self._num_step[:] = 0
        # else:
        #     self._num_step[env_ids] = 0
        
        # self._update_state(reset_estimator=True, env_ids=env_ids)
        
        # self._base_pos_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._base_quat_w2b = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self._device, requires_grad=False)
        # self._base_rot_mat_w2b = torch.zeros((self._num_envs, 3, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._base_rot_mat_b2w = torch.zeros((self._num_envs, 3, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._base_rpy_w2b = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._base_lin_vel_b = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._base_lin_vel_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._base_ang_vel_b = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._base_ang_vel_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._base_lin_acc_b = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._base_lin_acc_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._base_ang_acc_b = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._base_ang_acc_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)

        # # joint state
        # # self._motors = self._cfg.asset.motors
        # # self._num_joints = self._motors._num_motors
        # # self._dog_num_joints = 12
        # # print('num_joints: ', self._num_joints)
        # self._joint_pos = torch.zeros((self._num_envs, self._num_joints), dtype=torch.float, device=self._device, requires_grad=False)
        # self._joint_vel = torch.zeros((self._num_envs, self._num_joints), dtype=torch.float, device=self._device, requires_grad=False)

        # # robot jacobian in world frame
        # floating_base = self._cfg.sim.use_real_robot or (not self._cfg.get_asset_config().asset_options.fix_base_link)
        # num_all_links = 1 * int(floating_base) + 4*self._num_legs + 6 * int(self._use_gripper)
        # num_all_joints = self._num_joints + 6 * int(floating_base)
        # self._jacobian_w = torch.zeros((self._num_envs, num_all_links, 6, num_all_joints), dtype=torch.float, device=self._device, requires_grad=False)

        # # foot state
        # self._foot_contact = torch.ones((self._num_envs, self._num_legs), dtype=torch.bool, device=self._device, requires_grad=False)
        # self._desired_foot_contact = torch.ones((self._num_envs, self._num_legs), dtype=torch.bool, device=self._device, requires_grad=False)
        # self._foot_pos_hip = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._foot_vel_hip = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._foot_pos_b = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._foot_vel_b = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._foot_rot_b = torch.eye(3, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, self._num_legs, 1, 1)
        # self._foot_pos_w = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # print("foot pos w immedidately after reset:", self._foot_pos_w)
        # self._foot_vel_w = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._foot_rot_w = torch.eye(3, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, self._num_legs, 1, 1)
        # self._foot_T_w = torch.eye(4, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, self._num_legs, 1, 1)
        # self._swing_foot_pos_b = torch.zeros((self._num_envs, len(self._swing_foot_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._swing_foot_vel_b = torch.zeros((self._num_envs, len(self._swing_foot_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._swing_foot_pos_w = torch.zeros((self._num_envs, len(self._swing_foot_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._swing_foot_vel_w = torch.zeros((self._num_envs, len(self._swing_foot_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._stance_foot_pos_b = torch.zeros((self._num_envs, len(self._stance_foot_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._stance_foot_vel_b = torch.zeros((self._num_envs, len(self._stance_foot_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._stance_foot_pos_w = torch.zeros((self._num_envs, len(self._stance_foot_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._stance_foot_vel_w = torch.zeros((self._num_envs, len(self._stance_foot_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._origin_foot_pos_b = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)

        # # foot jacobian in body frame
        # self._foot_jacobian_b = torch.zeros((self._num_envs, self._num_legs, 3, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._swing_foot_jacobian_b = torch.zeros((self._num_envs, len(self._swing_foot_names), 3, 3), dtype=torch.float, device=self._device, requires_grad=False)
        # self._stance_foot_jacobian_b = torch.zeros((self._num_envs, len(self._stance_foot_names), 3, 3), dtype=torch.float, device=self._device, requires_grad=False)

        # # foot jacobian in world frame
        # self._foot_jacobian_w = torch.zeros((self._num_envs, self._num_legs, 3, 6+self._num_joints), dtype=torch.float, device=self._device, requires_grad=False)
        # self._swing_foot_jacobian_w = torch.zeros((self._num_envs, len(self._swing_foot_names), 3, 6+self._num_joints), dtype=torch.float, device=self._device, requires_grad=False)
        # self._stance_foot_jacobian_w = torch.zeros((self._num_envs, len(self._stance_foot_names), 3, 6+self._num_joints), dtype=torch.float, device=self._device, requires_grad=False)

        # # robot features
        # self._HIP_OFFSETS = torch.tensor(self._cfg.asset.hip_offsets, dtype=torch.float, device=self._device, requires_grad=False)
        # self._l_hip = self._cfg.asset.l_hip
        # self._l_thi = self._cfg.asset.l_thi
        # self._l_cal = self._cfg.asset.l_cal
        # self.l_hip_sign = (-1)**(self._foot_idx_of_four + 1)


        # self._dog_joint_idx = torch.tensor(self._cfg.asset.dog_joint_idx, dtype=torch.long, device=self._device, requires_grad=False)
        # self._gripper_joint_idx = torch.tensor(self._cfg.asset.gripper_joint_idx, dtype=torch.long, device=self._device, requires_grad=False)

        # if self._use_gripper:
        #     self._eef_names = self._cfg.asset.eef_names
        #     self._eef_pos_hip = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        #     self._eef_vel_hip = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        #     self._eef_pos_b = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        #     self._eef_vel_b = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        #     self._eef_pos_w = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        #     self._eef_vel_w = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        #     self._eef_jacobian_b = torch.zeros((self._num_envs, len(self._eef_names), 3, 6), dtype=torch.float, device=self._device, requires_grad=False)
        #     self._eef_jacobian_w = torch.zeros((self._num_envs, len(self._eef_names), 3, 6+self._num_joints), dtype=torch.float, device=self._device, requires_grad=False)
        #     self._eef_rpy_w = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
        #     self._eef_rot_w = torch.eye(3, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, len(self._eef_names), 1, 1)
        #     self._eef_T_foot = torch.eye(4, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, len(self._eef_names), 1, 1)
        #     self._eef_T_w = torch.eye(4, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, len(self._eef_names), 1, 1)

        #     self._close_gripper = torch.ones((self._num_envs, len(self._eef_names)), dtype=torch.bool, device=self._device, requires_grad=False)
        #     self._gripper_angles = torch.zeros((self._num_envs, len(self._eef_names)), dtype=torch.float, device=self._device, requires_grad=False)
        #     self._gripper_desired_angles = torch.zeros((self._num_envs, len(self._eef_names)), dtype=torch.float, device=self._device, requires_grad=False)
        #     # from manipulator.gripper_kinematics import GripperKinematicsTorch
        #     # self._gripper_kinematics = GripperKinematicsTorch( self._cfg, self._device)
        
        # if not self.first_reset:
        #     time.sleep(0.1)
        #     return
        
        self._init_buffers()
        
        if len(env_ids) == 0:
            return
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        cube_env_ids = env_ids + self._num_envs
        target_env_ids = env_ids + 2 * self._num_envs
        actor_ids_int32 = torch.cat([env_ids, cube_env_ids, target_env_ids], dim=-1).to(dtype=torch.int32)
        # env_ids_int32 = torch.cat([env_ids, cube_env_ids], dim=-1).to(dtype=torch.int32)
        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        # Reset root state:
        # print(self.all_root_state.device)
        # print(self._base_init_state.device)
        # print(env_ids.device)
        self.all_root_state[env_ids, 0] = self._base_init_state[env_ids]
        self.all_root_state[env_ids, 0, :3] += self._env_origins[env_ids]
        # cubes
        self.all_root_state[env_ids, 1] = torch.zeros(len(env_ids), 13).to(self._device)
        self.all_root_state[env_ids, 1, 3:7] = torch.tensor([0., 0., 0., 1.]).to(self._device)
        self.all_root_state[env_ids, 1, :3] = torch.tensor([self.sample_pos(self.cube_pos_offset_range) for _ in range(len(env_ids))]).to(self._device)
        self.all_root_state[env_ids, 1, :3] += self._env_origins[env_ids]
        # self.all_root_state[env_ids, 1, 2] += 0.2
        # targets
        self.all_root_state[env_ids, 2] = torch.zeros(len(env_ids), 13).to(self._device)
        self.all_root_state[env_ids, 2, 3:7] = torch.tensor([0., 0., 0., 1.]).to(self._device)
        self.all_root_state[env_ids, 2, :3] = torch.tensor([self.sample_pos(self.target_pos_offset_range) for _ in range(len(env_ids))]).to(self._device)
        self.all_root_state[env_ids, 2, :3] += self._env_origins[env_ids]
        
        self._gym.set_actor_root_state_tensor_indexed(self._sim, gymtorch.unwrap_tensor(self.all_root_state.view(-1, 13)),
                                                      gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))
        # Reset dofs
        self._joint_pos[env_ids] = self._joint_init_pos
        self._joint_vel[env_ids] = 0.
        self._gym.set_dof_state_tensor_indexed(self._sim, gymtorch.unwrap_tensor(self._dof_state),
                                               gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        # self._foot_pos_sim = self._rigid_body_state[:, self._foot_indices, 0:3]
        # print("foos pos sim before refresh:", self._foot_pos_sim)
        print("gonna step zero action")
        zero_action = MotorCommand(desired_position=self._joint_init_pos.repeat(self._num_envs, 1),
                                    kp=self._motors.kps.to(self._device),
                                    desired_velocity=torch.zeros_like(self._joint_init_pos),
                                    kd=self._motors.kds.to(self._device))
        self.step(zero_action, MotorControlMode.POSITION, show_img=False)
        print("zero action stepped")
        
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_jacobian_tensors(self._sim)
        self._gym.refresh_mass_matrix_tensors(self._sim)
        
        print("tensors refreshed")
        
        # If reset all enviroments
        if len(env_ids) == self._num_envs:
            # self._stablize_the_robot()
            self._num_step[:] = 0
        else:
            self._num_step[env_ids] = 0
        
        # self._foot_pos_sim = self._rigid_body_state[:, self._foot_indices, 0:3]
        # print("foot pos sim:", self._foot_pos_sim)
        
        self._update_state(reset_estimator=True, env_ids=env_ids)
        # self._foot_pos_sim = self._rigid_body_state[:, self._foot_indices, 0:3]
        # print("foot pos sim after update:", self._foot_pos_sim)
        

        
    def _stablize_the_robot(self):
        print("Ready to reset the robot!")
        zero_action = MotorCommand(desired_position=self._joint_init_pos.repeat(self._num_envs, 1),
                                    kp=self._motors.kps,
                                    desired_velocity=torch.zeros_like(self._joint_init_pos),
                                    kd=self._motors.kds)
        for _ in torch.arange(0, self._cfg.motor_controller.reset_time, self._dt):
            self.step(zero_action, MotorControlMode.POSITION)
            # self.step_without_update_state(zero_action, MotorControlMode.POSITION)
        print("Robot reset done!")
    
    def step(self, action: MotorCommand, motor_control_mode: MotorControlMode = None, gripper_cmd=True, show_img=True):
        self._log_info_now = self._log_info and self._num_step[0] % self._log_interval == 0
        self._num_step[:] += 1
        for _ in range(self._sim_conf.action_repeat):
            self._apply_action(action, motor_control_mode)
            self._gym.refresh_dof_state_tensor(self._sim)  # only need to refresh dof state for updating reference joit position and velocity
        self._update_state()
        if self._cfg.sim.show_gui:
            self._render()
        self._gym.render_all_camera_sensors(self._sim)
        rgb_image = self._gym.get_camera_image(self._sim, self._envs[0], self._cameras[0], gymapi.IMAGE_COLOR)  # this has 4 channels: RGBA
        rgb_image = rgb_image.reshape(rgb_image.shape[0], -1, 4)[:,:,:3]
        wrist_rgb_image = self._gym.get_camera_image(self._sim, self._envs[0], self._cameras[1], gymapi.IMAGE_COLOR)  # this has 4 channels: RGBA
        # depth_image = self._gym.get_camera_image(self._sim, self._env[0], self._cameras[0], gymapi.IMAGE_DEPTH)
        # rgb_image = rgb_image.reshape(rgb_image.shape[0], 4, -1).transpose(0, 2, 1)
        wrist_rgb_image = wrist_rgb_image.reshape(wrist_rgb_image.shape[0], -1, 4)[:,:,:3]
        # show the image
        if show_img:
            if self.realtime_img is None or not plt.isinteractive():
                ax = plt.subplot()
                self.realtime_img = ax.imshow(rgb_image)
                plt.ion()
            else:
                self.realtime_img.set_data(rgb_image)
                plt.pause(0.002)
            
        # return observations, current joint positions & camera    
        obs_dict = {
            'qpos': self._joint_pos[0],
            'qvel': self._joint_vel[0],
            'images': {
                'main': rgb_image,
                'wrist': wrist_rgb_image
            }
        }
        
        rew = self._compute_reward()
        
        # return torch.cat([self._joint_pos[0], self.all_root_state[0, 0, :3], self.all_root_state[0, 1, :3], self.all_root_state[0, 2, :3]], dim=-1).squeeze().to(self._device)
        return {'observation': obs_dict,
                'reward': rew,
                }
        
    def step_without_update_state(self, action: MotorCommand, motor_control_mode: MotorControlMode = None):
        self._log_info_now = self._log_info and self._num_step[0] % self._log_interval == 0
        self._num_step[:] += 1
        for _ in range(self._sim_conf.action_repeat):
            self._apply_action(action, motor_control_mode)
            self._gym.refresh_dof_state_tensor(self._sim)  # only need to refresh dof state for updating reference joit position and velocity
        self._update_sensors()
        if self._cfg.sim.show_gui:
            self._render()

    def _apply_action(self, action: MotorCommand, motor_control_mode: MotorControlMode = None):
        self._action_to_torque(action, motor_control_mode)
        self._gym.set_dof_actuation_force_tensor(self._sim, gymtorch.unwrap_tensor(self._torques))
        self._gym.simulate(self._sim)
        if self._device == "cpu":
            self._gym.fetch_results(self._sim, True)
        # if self._log_info_now:
        #     print('torques: ', self._torques[0].cpu().numpy())

    def _action_to_torque(self, action: MotorCommand, motor_control_mode: MotorControlMode = None):
        if motor_control_mode is None:
            motor_control_mode = self._motors._motor_control_mode
        if motor_control_mode == MotorControlMode.POSITION:
            self._torques[:] = action.kp * (action.desired_position - self._joint_pos) - action.kd * self._joint_vel
        elif motor_control_mode == MotorControlMode.TORQUE:
            self._torques[:] = action.desired_extra_torque
        elif motor_control_mode == MotorControlMode.HYBRID:
            self._torques[:] = action.kp * (action.desired_position - self._joint_pos) +\
                               action.kd * (action.desired_velocity - self._joint_vel) +\
                               action.desired_extra_torque
        else:
            raise ValueError('Unknown motor control mode for Go1 robot: {}.'.format(motor_control_mode))
        # if self._log_info_now:
        #     print('kp: ', action.kp[0].cpu().numpy())
        #     print('kd: ', action.kd[0].cpu().numpy())
        #     print('desired_position: ', action.desired_position[0].cpu().numpy())
        #     print('joint_pos: ', self._joint_pos[0].cpu().numpy())
        #     print('joint_vel: ', self._joint_vel[0].cpu().numpy())
        #     print('torques: ', self._torques[0].cpu().numpy())

    def _update_sensors(self):
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_dof_force_tensor(self._sim)
        self._gym.refresh_jacobian_tensors(self._sim)
        self._gym.refresh_mass_matrix_tensors(self._sim)
        # # Only need to recompute the tensors that are not from 'view' or 'slice
        # self._base_rot_mat_w2b[:] = quat_to_rot_mat(self._base_quat_w2b)
        # self._base_rot_mat_b2w[:] = torch.transpose(self._base_rot_mat_w2b, 1, 2)
        # self._projected_gravity[:] = torch.bmm(self._base_rot_mat_b2w, self._gravity_vec[:, :, None])[:, :, 0]
        # print("foot_pos_w: ", self._foot_pos_w)
        self._foot_pos_sim = self._rigid_body_state[:, self._foot_indices, 0:3]
        self._foot_vel_sim = self._rigid_body_state[:, self._foot_indices, 7:10]
        if self._use_gripper:
            self._eef_pos_sim = self._rigid_body_state[:, self._eef_indices, 0:3]
            self._eef_quat_sim = self._rigid_body_state[:, self._eef_indices, 3:7]

    def _update_foot_global_state(self):
        self._state_estimator[self._cur_fsm_state].set_foot_global_state()
        # print('foot_pos_w_after_updating: \n', self.foot_pos_w)

    def _update_foot_contact_state(self):
        self._foot_contact[:] = self._contact_forces[:, self._foot_indices, 2] > 1.0

    def _update_foot_jocabian_position_velocity(self):
        # ----------------- compute the jacobian in the world frame -----------------
        self._state_estimator[self._cur_fsm_state].compute_jacobian_w()

        # ----------------- compute the foot jacobian in the body frame -----------------
        for i in range(self._num_legs):
            self._foot_jacobian_b[:, i, :, :] = torch.bmm(self._base_rot_mat_b2w, self._jacobian_w[:, self._foot_indices[i], :3, 6:9])
        # self._foot_jacobian_b[:, 0, :, :] = torch.bmm(self._base_rot_mat_b2w, self._jacobian_w[:, 4, :3, 6:9])
        # self._foot_jacobian_b[:, 1, :, :] = torch.bmm(self._base_rot_mat_b2w, self._jacobian_w[:, 8, :3, 9:12])
        # self._foot_jacobian_b[:, 2, :, :] = torch.bmm(self._base_rot_mat_b2w, self._jacobian_w[:, 12, :3, 12:15])
        # self._foot_jacobian_b[:, 3, :, :] = torch.bmm(self._base_rot_mat_b2w, self._jacobian_w[:, 16, :3, 15:18])
        self._swing_foot_jacobian_b[:] = self._foot_jacobian_b[:, self._swing_foot_idx_of_four]
        self._stance_foot_jacobian_b[:] = self._foot_jacobian_b[:, self._stance_foot_idx_of_four]

        # # validate the foot jacobian
        # if self._log_info_now:
        #     # temp_foot_jacobian = self._compute_foot_jacobian(self._joint_pos)
        #     # print('diff: ', temp_foot_jacobian[0] - self._foot_jacobian_b[0])
        #     print('jacobian_w: \n', self._jacobian_w[0, :3, :6, :6].cpu().numpy())
        #     # print('temp_jacobian_w: \n', temp_jacobian_w[0, :3].cpu().numpy())
        #     # print('jacobian_sim: \n', self._jacobian_sim[0, :3, 0:6, 0:6])
        #     # print('reshape_1: \n', self._jacobian_sim[0, :3, 0:6, 0:6].unsqueeze(-3).reshape(1, -1, 4, 3, 3))
        #     # print('reshape_2: \n', self._jacobian_sim[0, :3, 0:6, 0:6].unsqueeze(-3).reshape(1, -1, 4, 3, 3).reshape(1, -1, 1, 6, 6).squeeze(-3))

        # ----------------- compute foot local position -----------------
        self._foot_pos_b[:] = torch.bmm(self._base_rot_mat_b2w, (self._foot_pos_w-self._base_pos_w.unsqueeze(1)).transpose(1, 2)).transpose(1, 2)
        self._foot_pos_hip[:] = self._foot_pos_b - self._HIP_OFFSETS
        self._swing_foot_pos_b[:] = self._foot_pos_b[:, self._swing_foot_idx_of_four]
        self._stance_foot_pos_b[:] = self._foot_pos_b[:, self._stance_foot_idx_of_four]

        # ----------------- compute foot local velocity -----------------
        # Vf^b = R_b^w * (Vf^w - Vb^w - [w_b^w] * R_w^b * pf^b)
        self._foot_vel_b[:] = torch.bmm(self._base_rot_mat_b2w,
                                        (self._foot_vel_w - self._base_lin_vel_w.unsqueeze(1)).transpose(-2, -1) -\
                                            torch.bmm(rpy_vel_to_skew_synmetric_mat(self._base_ang_vel_w), torch.bmm(self._base_rot_mat_b2w.transpose(-2, -1), self._foot_pos_b.transpose(-2, -1)))).transpose(1, 2)
        self._foot_vel_hip[:] = self._foot_vel_b
        self._swing_foot_vel_b[:] = self._foot_vel_b[:, self._swing_foot_idx_of_four]
        self._stance_foot_vel_b[:] = self._foot_vel_b[:, self._stance_foot_idx_of_four]

        # # validate the foot local state
        # if self._log_info_now:
        #     temp_foot_pos_b, temp_foot_vel_b = self._forward_kinematics(return_frame='body')
        #     print('pos_b_diff: ', temp_foot_pos_b[0] - self._foot_pos_b[0])
        #     print('vel_b_diff: ', temp_foot_vel_b[0] - self._foot_vel_b[0])

        # computed_joint_angle = self.compute_joint_pos_from_foot_pos_b(self._foot_pos_b)
        # print('computed_joint_angle: ', computed_joint_angle)
        # print('joint_pos: ', self._joint_pos)
            
    def _render(self, sync_frame_time=True):
        if self._viewer:
            # check for window closed
            if self._gym.query_viewer_has_closed(self._viewer):
                sys.exit()

        if self.first_render:
            direct_scale = 1.0 if self._cfg.manipulation.manipulate_leg_idx else -1.0
            mean_pos = torch.min(self._base_pos_w,
                                dim=0)[0].cpu().numpy() + np.array([.8, direct_scale*.9, .4])
            target_pos = torch.mean(self._base_pos_w, dim=0).cpu().numpy() + np.array([0.2, 0., 0.])
            cam_pos = gymapi.Vec3(*mean_pos)
            cam_target = gymapi.Vec3(*target_pos)
            self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)
            self.first_render = False

        if self._device != "cpu":
            self._gym.fetch_results(self._sim, True)

        # step graphics
        self._gym.step_graphics(self._sim)
        self._gym.draw_viewer(self._viewer, self._sim, True)
        if sync_frame_time:
            self._gym.sync_frame_time(self._sim)

    # torques
    @property
    def torques(self):
        return self._torques
    
    def _compute_reward(self):
        # compute the reward
        reward = 0
        # reward for the distance between the cube and the target
        cube_pos = self.all_root_state[0, 1, :3]
        target_pos = self.all_root_state[0, 2, :3]
        reward += torch.exp(-self._cfg.reward.reward_coeff * torch.norm(cube_pos - target_pos))
        return reward

    
    def get_camera_image(self):
        rgb_image = self._gym.get_camera_image(self._sim, self._envs[0], self._cameras[0], gymapi.IMAGE_COLOR)
        rgb_image = rgb_image.reshape(rgb_image.shape[0], -1, 4)[:,:,:3]