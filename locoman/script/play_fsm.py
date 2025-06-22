from multiprocessing import shared_memory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.exists("config/config.py"):
    print(1)
    import config
from config.config import Cfg
from config.go1_config import config_go1
from config.narrow_space_config import narrow_space_config
from config.loco_manipulation_config import loco_manipulation_config
import argparse
import rospy
import isaacgym


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def fsm(args):
    for key, value in vars(args).items():
        setattr(Cfg.sim, key, value)
        # print(f"cfg.sim.{key} = {getattr(Cfg.sim, key)}")
    Cfg.update_parms()
    config_go1(Cfg)
    if vars(args)['narrow_space']:
        narrow_space_config(Cfg)
    if vars(args)['loco_manipulation']:
        loco_manipulation_config(Cfg)

    from runner.fsm_runner import FSMRunner
    runner = FSMRunner(Cfg)
    
    while not rospy.is_shutdown():
        try:
            runner.step()
        except KeyboardInterrupt:
            # shm = shared_memory.SharedMemory(name=Cfg.teleoperation.shm_name)
            # shm.unlink()
            break

    quit()


if __name__ == '__main__':
    rospy.init_node('fsm_ros')
    parser = argparse.ArgumentParser(prog="Loco-Manipulation")
    parser.add_argument("--use_real_robot", type=str2bool, default=False, help="whether to use real robot.")
    parser.add_argument("--use_gpu", type=str2bool, default=True, help="whether to use GPU")
    parser.add_argument("--show_gui", type=str2bool, default=True, help="set as True to show GUI")
    parser.add_argument("--sim_device", default='cuda:0', help="the gpu to use")
    parser.add_argument("--use_gripper", type=str2bool, default=True, help="set as True to use gripper")
    parser.add_argument("--narrow_space", type=str2bool, default=False, help="set as True to manipulate in narrow space")
    parser.add_argument("--loco_manipulation", type=str2bool, default=False, help="set as True to perform loco-manipulation")
    args = parser.parse_args()

    fsm(args)

