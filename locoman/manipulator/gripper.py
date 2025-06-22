from config.config import Cfg
import numpy as np
from manipulator.base_servo import Address
from manipulator.dynamixel_client import DynamixelClient
import time
import math
import serial.tools.list_ports
import subprocess
# np.set_printoptions(edgeitems=3, infstr='inf', linewidth=200, nanstr='nan', precision=4, suppress=True, threshold=1000, formatter=None)


class Gripper:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.motor_ids = cfg.gripper.motor_ids

        self.dof_idx = cfg.gripper.dof_idx
        self.dof_motor_ids = list(np.array(self.motor_ids)[self.dof_idx])

        self.gripper_idx = cfg.gripper.gripper_idx
        self.gripper_motor_ids = list(np.array(self.motor_ids)[self.gripper_idx])

        self.arm_1_idx = cfg.gripper.arm_1_idx
        self.arm_1_motor_ids = list(np.array(self.motor_ids)[self.arm_1_idx])

        self.arm_2_idx = cfg.gripper.arm_2_idx
        self.arm_2_motor_ids = list(np.array(self.motor_ids)[self.arm_2_idx])

        self.cmd_addr = Address()

        try:
            self.dxl_client = DynamixelClient(self.motor_ids, self.dof_idx, self.gripper_idx, self.arm_1_idx, self.arm_2_idx, self.check_device_port(), cfg.gripper.baudrate)
            self.dxl_client.connect()
            print("Connected to the loco-manipulators")
        except Exception:
            print("Failed to connect to the loco-manipulators")

        self.reset_pos_sim = cfg.gripper.reset_pos_sim
        self.reset_time = cfg.gripper.reset_time

        self.s2r_scale = cfg.gripper.s2r_scale
        self.s2r_offset = cfg.gripper.s2r_offset
        self.des_pos_real = np.zeros(len(self.motor_ids))
        self.des_pos_applied = np.zeros(len(self.motor_ids))
        self.circle_offset = np.zeros(len(self.motor_ids))
        self.curr_pos_read = np.zeros(len(self.motor_ids))
        self.curr_vel_read = np.zeros(len(self.motor_ids))
        self.gripper_state_sim = np.zeros((2, len(self.motor_ids)))

        self.kP = cfg.gripper.kP
        self.kI = cfg.gripper.kI
        self.kD = cfg.gripper.kD
        self.CurrLim = np.ones(len(self.motor_ids)) * cfg.gripper.curr_lim
        self.gripper_delta_max = cfg.gripper.gripper_delta_max

        # ros topics
        self.gripper_des_pos_sim_topic = cfg.gripper.gripper_des_pos_sim_topic
        self.gripper_cur_state_sim_topic = cfg.gripper.gripper_cur_state_sim_topic

        self.reset()

    def reset(self):
        self.dxl_client.set_torque_enabled(self.motor_ids, False)
        self.set_operating_mode('current_position')
        self.set_PID_current_params()
        self.dxl_client.set_torque_enabled(self.motor_ids, True)
        for _ in range(3):
            self.update_circle_offset()
        self.move_to_target_pos(self.reset_pos_sim, 'sim', self.reset_time)
        self.get_gripper_state()

    def update_circle_offset(self):
        self.curr_pos_read[:], self.curr_vel_read[:] = self.dxl_client.read_all_pos_vel()
        if np.sum(self.curr_pos_read) < 1e-4:
            print('Failed to read the position')
            quit()
        self.circle_offset[(self.curr_pos_read / 3.14 * 180)>180] = 6.28

    def move_to_target_pos(self, target_pos, input_type='sim', duration=None):
        if input_type == 'sim':
            target_pos_applied = self.circle_offset+target_pos*self.s2r_scale+self.s2r_offset
        elif input_type == 'real':
            target_pos_applied = self.circle_offset+target_pos
        else:
            target_pos_applied = target_pos

        if duration is None:
            self.des_pos_applied = target_pos_applied
            self.dxl_client.write_desired_pos(self.motor_ids, self.des_pos_applied)
        else:
            # time_begin_0 = time.time()
            delta_t = 0.02
            for t in np.arange(delta_t, duration+delta_t, delta_t):
                time_begin = time.time()
                blend_ratio = min(t / duration, 1)
                self.des_pos_applied = blend_ratio * target_pos_applied + (1 - blend_ratio) * self.curr_pos_read
                self.dxl_client.write_desired_pos(self.motor_ids, self.des_pos_applied)
                time_cost = time.time() - time_begin
                # print('time_cost: ', time_cost)
                time.sleep(max(delta_t - time_cost, 0))
            # print('time_cost: ', (time.time() - time_begin_0))

    def set_des_pos_from_sim(self, des_pos_sim):
        des_pos_sim = des_pos_sim.clip(-3.3, 3.3)
        self.des_pos_applied = self.circle_offset+des_pos_sim*self.s2r_scale+self.s2r_offset
        # clip the gripper angles into a safe range
        self.des_pos_applied[self.gripper_idx] = np.clip(self.des_pos_applied[self.gripper_idx], self.curr_pos_read[self.gripper_idx]-self.gripper_delta_max, self.curr_pos_read[self.gripper_idx]+self.gripper_delta_max)
        self.dxl_client.write_desired_pos(self.motor_ids, self.des_pos_applied)

    def get_gripper_state(self):
        self.curr_pos_read[:], self.curr_vel_read[:] = self.dxl_client.read_all_pos_vel()
        self.gripper_state_sim[0, :] = ((self.curr_pos_read - self.circle_offset - self.s2r_offset) / self.s2r_scale)
        self.gripper_state_sim[1, :] = (self.curr_vel_read / self.s2r_scale)
        return self.gripper_state_sim

    def set_operating_mode(self, mode='current_position'):
        operating_mode = 5 if mode == 'current_position' else 3
        self.dxl_client.sync_write(self.motor_ids, np.ones(len(self.motor_ids))*operating_mode, self.cmd_addr.Operating_Mode[0], self.cmd_addr.Operating_Mode[1])

    def set_PID_current_params(self):
        self.dxl_client.sync_write(self.motor_ids, self.kP, self.cmd_addr.Position_P_GAIN[0], self.cmd_addr.Position_P_GAIN[1])
        self.dxl_client.sync_write(self.motor_ids, self.kI, self.cmd_addr.Position_I_GAIN[0], self.cmd_addr.Position_I_GAIN[1])
        self.dxl_client.sync_write(self.motor_ids, self.kD, self.cmd_addr.Position_D_GAIN[0], self.cmd_addr.Position_D_GAIN[1])
        self.dxl_client.sync_write(self.motor_ids, self.CurrLim, self.cmd_addr.GOAL_CURRENT[0], self.cmd_addr.GOAL_CURRENT[1])

    def check_device_port(self):
        # vid_pid should be in the format "0403:6014"
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if port.vid is not None and port.pid is not None:
                device_vid_pid = "{:04x}:{:04x}".format(port.vid, port.pid)
                if device_vid_pid.lower() == self.cfg.gripper.vid_pid.lower():
                    device_name = port.device.split('/')[-1]
                    set_success = set_latency_timer(device_name, 1)
                    if set_success:
                        return port.device
        print("Device not found")
        return None

def set_latency_timer(device, latency):
    # Building the path to the latency_timer file for the device
    latency_file_path = f"/sys/bus/usb-serial/devices/{device}/latency_timer"
    # Command to change the latency timer
    cmd = f"echo {latency} | sudo tee {latency_file_path}"
    try:
        # Executing the command
        subprocess.run(cmd, shell=True, check=True)
        print(f"Latency timer set to {latency} for {device}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to set latency timer for {device}: {e}")
        return False



#init the node
def main():
    gripper = Gripper(Cfg)

    gripper_actions = np.array([[math.pi, -0.05, 0., 0.02, math.pi, 0.05, 0., 0.02],
                                    [math.pi, -0.05, 0., 0.02, math.pi/2, 1.05, 0.5, 0.02],
                                    [math.pi, -0.05, 0., 0.02, math.pi, 0.05, 0., 0.02],
                                    [math.pi, -0.05, 0., 0.02, math.pi/2, 1.05, -0.5, 0.02],
                                    [math.pi, -0.05, 0., 0.02, math.pi, 0.05, 0., 0.02],
                                    ])

    while True:
        print(gripper.get_gripper_state())
        time.sleep(0.02)


if __name__ == "__main__":
    main()



# -------Notes-------

# 1. when the operating mode is set to position control mode, the read position will be reset to 0~2pi,
#    so remember to update circle_offset, see: https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/#present-position
