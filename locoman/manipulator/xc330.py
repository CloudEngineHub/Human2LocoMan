from manipulator.base_servo import BaseServo


class XC330(BaseServo):
    def __init__(self, servo_ids):
        super().__init__('xc330', servo_ids)
    
        self.address.TORQUE_ENABLE = 64
        self.address.GOAL_POSITION = 116
        self.address.PRESENT_POS_VEL_CUR = 126