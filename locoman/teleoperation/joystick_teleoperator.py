import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rospy
from std_msgs.msg import Float32MultiArray, Int32
import pygame
import numpy as np
from config.config import Cfg
import time


class JoystickTeleoperator():
    def __init__(self):
        # Initialize pygame for Xbox controller
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        rospy.init_node('joystick_teleoperation')

        self.command_publisher = rospy.Publisher(Cfg.commander.joystick_command_topic, Float32MultiArray, queue_size = 1)
        self.fsm_publisher = rospy.Publisher(Cfg.fsm_switcher.fsm_state_topic, Int32, queue_size = 1)
        self.teleop_mode_publisher = rospy.Publisher(Cfg.teleoperation.human_teleoperator.mode_updata_topic, Int32, queue_size = 1)
        self.fsm_to_teleop_mode_mapping = Cfg.teleoperation.human_teleoperator.fsm_to_teleop_mode_mapping

        self.command = np.zeros(14)
        self.command_msg = Float32MultiArray()
        self.command_msg.data = self.command.tolist()
        self.last_command_switch = 0.
        self.command_inpute_for_body = True

        self.fsm_state_msg = Int32()
        self.fsm_state_msg.data = 0
        self.teleop_mode_msg = Int32()
        self.teleop_mode_msg.data = 0
        self.last_stance_bimanual_time = time.time()
        self.stance_bimanual_transition_time = 1.0
        self.rate = rospy.Rate(30)

    def run(self):
        while not rospy.is_shutdown():
            self.construct_command()
            self.rate.sleep()

    def construct_command(self):
        pygame.event.pump()

        # buttons: 0:A, 1:B, 2:X, 3:Y, 4:LB, 5:RB, 6:Select-, 7:+Start, 8:Home
        # FSM_Buttons: A: stance, -:lf-manipulate +: rf-manipulate X: le-manipulate, B: re-manipulate, Y: locomotion

        command_switch_signal = self.joystick.get_button(8)
        if command_switch_signal == 1 and self.last_command_switch != 1:
            self.command_inpute_for_body = not self.command_inpute_for_body
            self.last_command_switch = 1
        else:
            self.last_command_switch = command_switch_signal

        self.command[:] = 0
        self.command[12:14] = np.array(self.joystick.get_hat(0))
        x = -self.joystick.get_axis(1)
        y = -self.joystick.get_axis(0)
        z = (-self.joystick.get_axis(2)-2) /2.0 + (self.joystick.get_axis(5)+2) /2.0
        roll  = -self.joystick.get_button(4) + self.joystick.get_button(5)
        pitch = self.joystick.get_axis(4)
        yaw = -self.joystick.get_axis(3)
        if self.command_inpute_for_body:
            self.command[0:6] = np.array([x, y, z, roll, pitch, yaw])
        else:
            self.command[6:12] = np.array([x, y, z, roll, pitch, yaw])
        self.command[abs(self.command) < 0.1] = 0
        self.command_msg.data = self.command.tolist()
        self.command_publisher.publish(self.command_msg)

        fsm_buttons = [self.joystick.get_button(0), self.joystick.get_button(7), self.joystick.get_button(6), self.joystick.get_button(1), self.joystick.get_button(2), self.joystick.get_button(3)]
        fsm_buttons = list(map(bool, fsm_buttons))
        if 1 in fsm_buttons:
            true_index = fsm_buttons.index(True)
            # fsm: 0:stance, 1:rf-manipulate, 2:lf-manipulate, 3:re-manipulate, 4:le-manipulate, 5:locomotion, 6:loco-manipulation, 7:bi-manipulation
            if self.fsm_state_msg.data != [0, 1, 2, 3, 4, 5][true_index] and time.time() - self.last_stance_bimanual_time > self.stance_bimanual_transition_time:
                if self.fsm_state_msg.data !=0:
                    self.fsm_state_msg.data = 0
                    self.command_inpute_for_body = True
                    self.last_stance_bimanual_time = time.time()
                else:
                    self.fsm_state_msg.data = [0, 1, 2, 3, 4, 5][true_index]
                    self.command_inpute_for_body = False if self.fsm_state_msg.data!=5 else True
                print('fsm_state: ', self.fsm_state_msg.data)
                self.fsm_publisher.publish(self.fsm_state_msg)
            # mask bimanual manipulation or loco-manipulation
            elif self.fsm_state_msg.data == 0 and time.time() - self.last_stance_bimanual_time > self.stance_bimanual_transition_time:
                self.fsm_state_msg.data = 7  # bi-manipulation
                # self.fsm_state_msg.data = 6  # loco-manipulation
                self.last_stance_bimanual_time = time.time()
                print('fsm_state: ', self.fsm_state_msg.data)
                self.fsm_publisher.publish(self.fsm_state_msg)
            self.teleop_mode_msg.data = self.fsm_to_teleop_mode_mapping[self.fsm_state_msg.data]
            self.teleop_mode_publisher.publish(self.teleop_mode_msg)



if __name__ == '__main__':
    joystick_teleoperator = JoystickTeleoperator()
    try:
        joystick_teleoperator.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        pygame.quit()
