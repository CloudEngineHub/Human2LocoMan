from robot.base_robot import BaseRobot
from commander.stance_commander import StanceCommander
from commander.foot_manipulate_commander import FootManipulateCommander
from commander.eef_manipulate_commander import EEFManipulateCommander
from commander.locomotion_commander import LocomotionCommander
from commander.loco_manipulation_commander import LocoManipulationCommander
from commander.bi_manipulation_commander import BiManipulationCommander
from fsm.finite_state_machine import FSM_State, Manipulate_Mode


class FSMCommander:
    def __init__(self, runner, env_ids=0):
        self._runner = runner
        self._robot: BaseRobot = runner._robot
        self._env_ids = env_ids
        self._cfg = self._robot._cfg

        self._stance_commander = StanceCommander(self._robot, self._env_ids)
        self._foot_manipulate_commander = FootManipulateCommander(self._robot, self._env_ids)
        self._eef_manipulate_commander = EEFManipulateCommander(self._robot, self._env_ids)
        self._locomotion_commander = LocomotionCommander(self._robot, self._env_ids)
        if self._cfg.loco_manipulation.loco_manipulation_commander:
            self._locomanipulation_commander = LocoManipulationCommander(self._robot, self._env_ids)
        self._bi_manipulation_commander = BiManipulationCommander(self._robot, self._env_ids)


    def get_command_generator(self):
        if self._robot._cur_fsm_state == FSM_State.STANCE:
            return self._stance_commander
        elif self._robot._cur_fsm_state == FSM_State.MANIPULATION and self._robot._cur_manipulate_mode in [Manipulate_Mode.LEFT_FOOT, Manipulate_Mode.RIGHT_FOOT]:
            return self._foot_manipulate_commander
        elif self._robot._cur_fsm_state == FSM_State.MANIPULATION and self._robot._cur_manipulate_mode in [Manipulate_Mode.LEFT_EEF, Manipulate_Mode.RIGHT_EEF]:
            return self._eef_manipulate_commander
        elif self._robot._cur_fsm_state == FSM_State.LOCOMOTION:
            return self._locomotion_commander
        elif self._robot._cur_fsm_state == FSM_State.LOCOMANIPULATION:
            return self._locomanipulation_commander
        elif self._robot._cur_fsm_state == FSM_State.BIMANIPULATION:
            return self._bi_manipulation_commander











