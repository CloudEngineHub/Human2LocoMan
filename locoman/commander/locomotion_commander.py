from commander.base_commander import BaseCommander
import numpy as np
from planner.gait_planner import GaitPlanner, LegState
from planner.raibert_swing_leg_planner import RaibertSwingLegPlanner


class LocomotionCommander(BaseCommander):
    def __init__(self, robot, env_ids=0):
        super().__init__(robot, env_ids=0)
        self._action_mode = 3
        self._desired_body_pva[0:6] = self._cfg.locomotion.desired_pose
        self._desired_body_pva[6:12] = self._cfg.locomotion.desired_velocity

        # for switching to stance
        self._going_to_stand = False
        self._switching_time = 0.5
        self._switching_steps = 0

        self._gait_generator = GaitPlanner(robot, env_ids)
        foot_landing_clearance = self._cfg.locomotion.foot_landing_clearance_real if self._cfg.sim.use_real_robot else self._cfg.locomotion.foot_landing_clearance_sim
        self._swing_leg_controller = RaibertSwingLegPlanner(robot, env_ids, self._gait_generator, self._desired_body_pva[6:8], self._desired_body_pva[11], foot_landing_clearance=foot_landing_clearance, foot_height=self._cfg.locomotion.foot_height)

    def reset(self):
        self._action_mode = 3
        self._desired_body_pva[0:6] = self._cfg.locomotion.desired_pose
        self._desired_body_pva[6:12] = self._cfg.locomotion.desired_velocity

        average_height = -np.mean(self._robot.foot_pos_b_np[self._env_ids, :, 2])
        self._swing_leg_controller._foot_height = (average_height / self._desired_body_pva[2]) * self._cfg.locomotion.foot_height
        self._desired_body_pva[2] = average_height

        self._gait_generator.reset()
        self._swing_leg_controller.reset()

        super().reset()

    def _update_joystick_command_callback(self, command_msg):
        if self._is_used:
            self._is_updating_command = True
            command_np = np.array(command_msg.data)
            self._body_pv_buffer[6:8] = command_np[0:2] * self._body_pv_scale[6:8]  # vx, vy
            self._body_pv_buffer[2:5] = command_np[2:5] * self._body_pv_scale[2:5]  # z, roll, pitch
            self._body_pv_buffer[11] = command_np[5] * self._body_pv_scale[11]  # wz
            self._eef_joint_pos_buffer[:] = command_np[6:12].reshape(2, 3) * self._eef_joint_pos_scale
            self._gripper_angles_buffer[:] = command_np[12:14] * self._gripper_angle_scale
            self._is_updating_command = False

    def _update_human_command_callback(self, command_msg):
        pass

    def compute_command_for_wbc(self):
        super().compute_command_for_wbc()
        self._gait_generator.update()
        self._swing_leg_controller.update()
        self._contact_state[:] = np.array([state in (LegState.STANCE, LegState.EARLY_CONTACT, LegState.LOSE_CONTACT) for state in self._gait_generator.leg_state])
        self._desired_footeef_pva_w[0, :] = self._swing_leg_controller.get_desired_foot_positions().flatten()

        if not self._is_updating_command:
            if self._body_p_command_delta:
                self._body_pva_cmd[3:5] += self._body_pv_buffer[3:5]
                self._body_pva_cmd[3:5] = np.clip(self._body_pva_cmd[3:5], -self._body_pv_limit[3:5], self._body_pv_limit[3:5])
            else:
                self._body_scaled_pva_cmd[3:5] = self._body_pv_buffer[3:5] * self._body_pv_limit[3:5]
                self._body_pva_cmd[3:5] = np.clip(self._body_scaled_pva_cmd[3:5], self._body_pva_cmd[3:5]-self._body_pose_scale[3:5], self._body_pva_cmd[3:5]+self._body_pose_scale[3:5])
            self._body_pva_cmd[2] += self._body_pv_buffer[2]  # for height, always use delta command
            self._body_pva_cmd[2] = np.clip(self._body_pva_cmd[2], self._locomotion_height_range[0], self._locomotion_height_range[1])
            
            if self._body_v_command_delta:
                self._body_pva_cmd[6:12] += self._body_pv_buffer[6:12]
                self._body_pva_cmd[6:12] = np.clip(self._body_pva_cmd[6:12], -self._body_pv_limit[6:12], self._body_pv_limit[6:12])
            else:
                self._body_scaled_pva_cmd[6:12] = self._body_pv_buffer[6:12] * self._body_pv_limit[6:12]
                self._body_pva_cmd[6:12] = np.clip(self._body_scaled_pva_cmd[6:12], self._body_pva_cmd[6:12]-self._body_vel_scale, self._body_pva_cmd[6:12]+self._body_vel_scale)
            self._gripper_angles_cmd[:] += self._gripper_angles_buffer
            self._gripper_angles_cmd[:] = np.clip(self._gripper_angles_cmd, self._gripper_angle_range[0], self._gripper_angle_range[1])
            
            self._body_pv_buffer[:] = 0
            self._eef_pva_cmd[:, 0:3] += self._eef_joint_pos_buffer
            self._eef_joint_pos_buffer[:] = 0
            self._gripper_angles_buffer[:] = 0

        if self._going_to_stand:
            height = self._body_pva_cmd[2]
            self._body_pva_cmd[:] = 0
            self._body_pva_cmd[2] = height
            self._switching_steps += 1

        return {"action_mode": self._action_mode,
                "contact_state": self._contact_state,
                "body_pva": self._body_pva_cmd,
                "footeef_pva": self._desired_footeef_pva_w,
                "eef_pva": self._eef_pva_cmd,
                "gripper_angles": self._gripper_angles_cmd,
                }

    def prepare_to_stand(self):
        self._going_to_stand = True

    def check_finished(self):
        if self._switching_steps * self._robot._dt > self._switching_time and np.sum(self._contact_state) == 4:
            self._going_to_stand = False
            self._switching_steps = 0
            return True
        else:
            return False







