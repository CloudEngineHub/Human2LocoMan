import enum


class FSM_State(enum.Enum):
    STANCE = 1
    MANIPULATION = 2
    LOCOMOTION = 3
    LOCOMANIPULATION = 4
    BIMANIPULATION = 5

class Manipulate_Mode(enum.Enum):
    LEFT_FOOT = 1
    RIGHT_FOOT = 2
    LEFT_EEF = 3
    RIGHT_EEF = 4


class FSM_OperatingMode(enum.Enum):
    NORMAL = 1
    TRANSITION = 2
    RESET = 3

# class FSM_Command(enum.Enum):
#     STANCE = 0
#     MANIPULATION_LEFT_EEF = 1
#     MANIPULATION_RIGHT_EEF = 2
#     MANIPULATION_LEFT_FOOT = 3
#     MANIPULATION_RIGHT_FOOT = 4
#     LOCOMOTION = 5
#     LOCOMANIPULATION = 6
#     BIMANIPULATION = 7
    
def fsm_command_to_fsm_state_and_manipulate_mode(command, with_gripper=True):
    fsm_state =  [FSM_State.STANCE,
            FSM_State.MANIPULATION,
            FSM_State.MANIPULATION,
            FSM_State.MANIPULATION,
            FSM_State.MANIPULATION,
            FSM_State.LOCOMOTION,
            FSM_State.LOCOMANIPULATION,
            FSM_State.BIMANIPULATION][command]
    if with_gripper:
        manipulate_mode = [None, Manipulate_Mode.RIGHT_FOOT, Manipulate_Mode.LEFT_FOOT, Manipulate_Mode.RIGHT_EEF, Manipulate_Mode.LEFT_EEF, None, None, None][command]
    else:
        manipulate_mode = [None, Manipulate_Mode.RIGHT_FOOT, Manipulate_Mode.LEFT_FOOT, Manipulate_Mode.RIGHT_FOOT, Manipulate_Mode.LEFT_FOOT, None, None, None][command]

    return fsm_state, manipulate_mode


# stance-0, foot_manipulation-1, eef_manipulation-2, locomotion-3, loco-manipulation-4
def fsm_state_and_manipulate_mode_to_action_mode(fsm_state: FSM_State, manipulate_mode: Manipulate_Mode):
    if fsm_state == FSM_State.STANCE:
        return 0
    elif fsm_state == FSM_State.MANIPULATION:
        if manipulate_mode in [Manipulate_Mode.LEFT_FOOT, Manipulate_Mode.RIGHT_FOOT]:
            return 1
        else:
            return 2
    elif fsm_state == FSM_State.LOCOMOTION:
        return 3
    else:
        return 4













