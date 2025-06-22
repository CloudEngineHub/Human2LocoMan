from config.config import Cfg
from config.go1_config import config_go1
from config.loco_manipulation_config import loco_manipulation_config
import argparse
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from manipulator.gripper import Gripper
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class GripperROS:
    def __init__(self, cfg: Cfg):
        rospy.init_node('gripper_ros')
        self.gripper = Gripper(cfg)
        self.rate = rospy.Rate(cfg.gripper.update_gripper_freq)

        self.cur_state_sim_pub = rospy.Publisher(cfg.gripper.gripper_cur_state_sim_topic, JointState, queue_size=100)
        self.cur_state_sim_msg = JointState()
        self.cur_state_sim_msg.name = ['1_FR_joint_1', '1_FR_joint_2', '1_FR_joint_3', '1_FR_eef', '2_FL_joint_1', '2_FL_joint_2', '2_FL_joint_3', '2_FL_eef']

        self.des_pos_sim_sub = rospy.Subscriber(cfg.gripper.gripper_des_pos_sim_topic, JointState, self.des_pos_sim_callback)

        self.during_getting_state = False
        print('------- Ready to control gripper -------')

    def des_pos_sim_callback(self, joint_msg: JointState):
        if self.during_getting_state:
            return

        self.gripper.set_des_pos_from_sim(np.array(joint_msg.position))
        if joint_msg.name[0] == 'update_state':
            self.during_getting_state = True
            self.publish_cur_state()
            self.during_getting_state = False

    def publish_cur_state(self):
        self.gripper.get_gripper_state()
        self.cur_state_sim_msg.header.stamp = rospy.Time.now()
        self.cur_state_sim_msg.position = list(self.gripper.gripper_state_sim[0])
        self.cur_state_sim_msg.velocity = list(self.gripper.gripper_state_sim[1])
        self.cur_state_sim_pub.publish(self.cur_state_sim_msg)


    def run(self):
        rospy.spin()

        # while not rospy.is_shutdown():
        #     time_begin = time.time()
        #     self.gripper.get_gripper_state()
        #     self.gripper.dxl_client.write_desired_pos(self.gripper.motor_ids, self.gripper.des_pos_applied)
        #     time_cost = time.time() - time_begin
        #     print('time_cost: ', time_cost)
        #     self.rate.sleep()


def main(args):
    config_go1(Cfg)

    if vars(args)['loco_manipulation']:
        loco_manipulation_config(Cfg)
        
    gripper_ros = GripperROS(Cfg)
    gripper_ros.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Gripper")
    parser.add_argument("--loco_manipulation", type=str2bool, default=False, help="set as True to perform loco-manipulation")
    args = parser.parse_args()
    main(args)


