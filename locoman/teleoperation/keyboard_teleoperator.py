#!/usr/bin/env python

from __future__ import print_function

import threading

import rospy
from std_msgs.msg import Float32MultiArray, Int32

import sys
from select import select

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

import numpy as np
from config.config import Cfg


# control the body movement:
# d(x, y, z, roll, pitch, yaw) for stance/manipulation mode
# d(vx, vy, z, roll, pitch, wz) for locomotion mode
bodyBindings = {
    'e': np.array([1, 0, 0, 0, 0, 0]),
    'd': np.array([-1, 0, 0, 0, 0, 0]),
    's': np.array([0, 1, 0, 0, 0, 0]),
    'f': np.array([0, -1, 0, 0, 0, 0]),
    't': np.array([0, 0, 1, 0, 0, 0]),
    'g': np.array([0, 0, -1, 0, 0, 0]),
    'l': np.array([0, 0, 0, 1, 0, 0]),
    'j': np.array([0, 0, 0, -1, 0, 0]),
    'i': np.array([0, 0, 0, 0, 1, 0]),
    'k': np.array([0, 0, 0, 0, -1, 0]),
    'y': np.array([0, 0, 0, 0, 0, 1]),
    'h': np.array([0, 0, 0, 0, 0, -1]),
    }

# control the end-effector movement:
# d(q1, q2, q3, q4, q5, q6) of the grippers' joint angles for stance/locomotion mode
# d(x, y, z) of the foot and d(q1, q2, q3) of the manipulate-gripper's joint angles for foot-based manipulation mode
# d(x, y, z, roll, pitch, yaw) of the end-effector in the world frame for eef-based manipulation mode
eefBindings = {
    'E': np.array([1, 0, 0, 0, 0, 0]),
    'D': np.array([-1, 0, 0, 0, 0, 0]),
    'S': np.array([0, 1, 0, 0, 0, 0]),
    'F': np.array([0, -1, 0, 0, 0, 0]),
    'T': np.array([0, 0, 1, 0, 0, 0]),
    'G': np.array([0, 0, -1, 0, 0, 0]),
    'L': np.array([0, 0, 0, 1, 0, 0]),
    'J': np.array([0, 0, 0, -1, 0, 0]),
    'I': np.array([0, 0, 0, 0, 1, 0]),
    'K': np.array([0, 0, 0, 0, -1, 0]),
    'Y': np.array([0, 0, 0, 0, 0, 1]),
    'H': np.array([0, 0, 0, 0, 0, -1]),
    }

FSMBindings = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
}


class PublishThread(threading.Thread):
    def __init__(self):
        super(PublishThread, self).__init__()
        self.command_publisher = rospy.Publisher(Cfg.commander.joystick_command_topic, Float32MultiArray, queue_size = 1)
        self.fsm_publisher = rospy.Publisher(Cfg.fsm_switcher.fsm_state_topic, Int32, queue_size = 1)
        self.robot_reset_publisher = rospy.Publisher(Cfg.teleoperation.robot_reset_topic, Int32, queue_size = 1)
        self.auto_mode_publisher = rospy.Publisher(Cfg.teleoperation.auto_mode_topic, Int32, queue_size = 1)
        
        self.command = np.zeros(20)
        self.command_msg = Float32MultiArray()
        self.command_msg.data = self.command.tolist()

        self.fsm_state_msg = Int32()
        self.fsm_state_msg.data = 0
        
        self.robot_reset_msg = Int32()
        self.robot_reset_msg.data = 0
        
        self.auto_mode_msg = Int32()
        self.auto_mode_msg.data = 0

        self.condition = threading.Condition()
        self.done = False
        self.rate = rospy.Rate(30)
        self.start()

    def stop(self):
        self.done = True
        self.join()

    def run(self):
        while not self.done:
            self.condition.acquire()
            self.command[:] = 0
            key = getKey(settings, timeout=0.5)
            # if key is not None:
            #     print('key: ', key)
            if key in bodyBindings.keys():
                self.command[0:6] = bodyBindings[key]
                self.command_msg.data = self.command.tolist()
                print('command: ', self.command)
                self.command_publisher.publish(self.command_msg)
            elif key in eefBindings.keys():
                self.command[6:12] = eefBindings[key]
                self.command_msg.data = self.command.tolist()
                print('command: ', self.command)
                self.command_publisher.publish(self.command_msg)
            elif key in FSMBindings.keys():
                self.fsm_state_msg.data = FSMBindings[key]
                print('fsm_state: ', self.fsm_state_msg.data)
                self.fsm_publisher.publish(self.fsm_state_msg)
            elif key == '\x03':
                self.done = True
            elif key == '\t': # Tab key
                self.robot_reset_publisher.publish(self.robot_reset_msg)
                print('reset within fsm_state: ', self.fsm_state_msg.data)
            elif key == 'a': # auto
                self.auto_mode_msg.data = 1 if self.auto_mode_msg.data == 0 else 0
                self.auto_mode_publisher.publish(self.auto_mode_msg)
            self.condition.notify()  # Notify any waiting thread that something has changed
            self.condition.release()
            self.rate.sleep()

def getKey(settings, timeout):
    tty.setraw(sys.stdin.fileno())
    # sys.stdin.read() returns a string on Linux
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def restoreTerminalSettings(old_settings):
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('teleop_twist_keyboard')

    pub_thread = PublishThread()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        pub_thread.stop()
        restoreTerminalSettings(settings)






