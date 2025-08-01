#!/usr/bin/python

import sys
import time
import numpy as np
import pickle
import unitree_legged_sdk.lib.python.amd64.robot_interface as sdk
from config.config import Cfg


if __name__ == '__main__':

    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    states = []
    udp.InitCmdData(cmd)

    step_time = Cfg.motor_controller.dt / Cfg.bimanual_trajectory.recording_fps_mul_motor_controller_fps

    motiontime = 0
    start_time = time.time()
    while time.time() - start_time < 10:
        time_start = time.time()
        udp.Recv()
        udp.GetRecv(state)

        states.append(dict(
            timestamp=time.time() - start_time,
            body_height=np.array(state.bodyHeight),
            body_position=np.array(state.position),
            base_orientation=np.array(state.imu.rpy),
            velocity=np.array(state.velocity),
            angular_velocity=np.array(state.imu.gyroscope),
            motor_position=np.array([m.q for m in state.motorState]),
            motor_vel=np.array([m.dq for m in state.motorState]),
            motor_torque=np.array([m.tauEst for m in state.motorState]),
            motor_acc=np.array([m.ddq for m in state.motorState]),
            foot_force=np.array(state.footForce),
            foot_force_est=np.array(state.footForceEst),
        ))
        # print(motiontime, state.imu.rpy[1])
        udp.Send()

        motiontime += 1
        time.sleep(max(0, step_time - (time.time() - time_start)))

    with open(Cfg.bimanual_trajectory.trajectory_path, "wb") as f:
        pickle.dump(states, f)


