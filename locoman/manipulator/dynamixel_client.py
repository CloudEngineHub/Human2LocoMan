"""Communication using the DynamixelSDK."""
##This is based off of the dynamixel SDK
import atexit
import logging
import time
from typing import Optional, Sequence, Union, Tuple

import numpy as np

PROTOCOL_VERSION = 2.0

# The following addresses assume XH motors.
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POS_VEL_CUR = 126
ADDR_PRESENT_POS_VEL = 126

ADDR_Position_D_GAIN = 80
ADDR_Position_I_GAIN = 82
ADDR_Position_P_GAIN = 84
ADDR_Position_PID = 80

# Data Byte Length
LEN_PRESENT_POSITION = 4
LEN_PRESENT_VELOCITY = 4
LEN_PRESENT_CURRENT = 2
LEN_PRESENT_POS_VEL_CUR = 10
LEN_PRESENT_POS_VEL = 8
LEN_GOAL_POSITION = 4

LEN_Position_D_GAIN = 2
LEN_Position_I_GAIN = 2
LEN_Position_P_GAIN = 2
LEN_Position_PID = 6

DEFAULT_POS_SCALE = 2.0 * np.pi / 4096  # 0.001534 rad (0.088 degrees)
# See http://emanual.robotis.com/docs/en/dxl/x/xh430-v210/#goal-velocity
DEFAULT_VEL_SCALE = 0.229 * 2.0 * np.pi / 60.0  # 0.229 rpm
DEFAULT_CUR_SCALE = 1.34

COMM_SUCCESS = 0

def dynamixel_cleanup_handler():
    """Cleanup function to ensure Dynamixels are disconnected properly."""
    open_clients = list(DynamixelClient.OPEN_CLIENTS)
    for open_client in open_clients:
        if open_client.port_handler.is_using:
            logging.warning('Forcing client to close.')
        open_client.port_handler.is_using = False
        open_client.disconnect()


def signed_to_unsigned(value: int, size: int) -> int:
    """Converts the given value to its unsigned representation."""
    if value < 0:
        bit_size = 8 * size
        max_value = (1 << bit_size) - 1
        value = max_value + value
    return value


def unsigned_to_signed(value: int, size: int) -> int:
    """Converts the given value from its unsigned representation."""
    bit_size = 8 * size
    if (value & (1 << (bit_size - 1))) != 0:
        value = -((1 << bit_size) - value)
    return value


class DynamixelClient:
    """Client for communicating with Dynamixel motors.

    NOTE: This only supports Protocol 2.
    """

    # The currently open clients.
    OPEN_CLIENTS = set()

    def __init__(self,
                 motor_ids: Sequence[int],
                 dof_idx: Sequence[int],
                 gripper_idx: Sequence[int],
                 arm_1_idx: Sequence[int],
                 arm_2_idx: Sequence[int],
                 port: str = '/dev/ttyUSB1',
                 baudrate: int = 1000000,
                 lazy_connect: bool = False,
                 pos_scale: Optional[float] = None,
                 vel_scale: Optional[float] = None,
                 cur_scale: Optional[float] = None):
        """Initializes a new client.

        Args:
            motor_ids: All motor IDs being used by the client.
            port: The Dynamixel device to talk to. e.g.
                - Linux: /dev/ttyUSB0
                - Mac: /dev/tty.usbserial-*
                - Windows: COM1
            baudrate: The Dynamixel baudrate to communicate with.
            lazy_connect: If True, automatically connects when calling a method
                that requires a connection, if not already connected.
            pos_scale: The scaling factor for the positions. This is
                motor-dependent. If not provided, uses the default scale.
            vel_scale: The scaling factor for the velocities. This is
                motor-dependent. If not provided uses the default scale.
            cur_scale: The scaling factor for the currents. This is
                motor-dependent. If not provided uses the default scale.
        """
        import dynamixel_sdk
        self.dxl = dynamixel_sdk

        self.motor_ids = list(motor_ids)

        self.dof_idx = list(dof_idx)
        self.dof_motor_ids = list(np.array(self.motor_ids)[self.dof_idx])
        self.gripper_idx = list(gripper_idx)
        self.gripper_motor_ids = list(np.array(self.motor_ids)[self.gripper_idx])
        self.arm_1_idx = list(arm_1_idx)
        self.arm_1_motor_ids = list(np.array(self.motor_ids)[self.arm_1_idx])
        self.arm_2_idx = list(arm_2_idx)
        self.arm_2_motor_ids = list(np.array(self.motor_ids)[self.arm_2_idx])

        self.port_name = port
        self.baudrate = baudrate
        self.lazy_connect = lazy_connect

        self.port_handler = self.dxl.PortHandler(port)
        self.packet_handler = self.dxl.PacketHandler(PROTOCOL_VERSION)

        self._all_pos_reader = dynamixel_sdk.GroupSyncRead(self.port_handler, self.packet_handler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        self._all_vel_reader = dynamixel_sdk.GroupSyncRead(self.port_handler, self.packet_handler, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
        
        self._dof_pos_vel_reader = dynamixel_sdk.GroupSyncRead(self.port_handler, self.packet_handler, ADDR_PRESENT_VELOCITY, LEN_PRESENT_POS_VEL)
        self._gripper_pos_vel_reader = dynamixel_sdk.GroupSyncRead(self.port_handler, self.packet_handler, ADDR_PRESENT_VELOCITY, LEN_PRESENT_POS_VEL)

        self._arm_1_pos_vel_reader = dynamixel_sdk.GroupSyncRead(self.port_handler, self.packet_handler, ADDR_PRESENT_VELOCITY, LEN_PRESENT_POS_VEL)
        self._arm_2_pos_vel_reader = dynamixel_sdk.GroupSyncRead(self.port_handler, self.packet_handler, ADDR_PRESENT_VELOCITY, LEN_PRESENT_POS_VEL)

        for motor_id in self.motor_ids:
            pos_addparam_result = self._all_pos_reader.addParam(motor_id)
            vel_addparam_result = self._all_vel_reader.addParam(motor_id)
            if not (pos_addparam_result and vel_addparam_result):
                print("[ID:%03d] groupSyncRead addparam failed" % motor_id)
                quit()
            if motor_id in self.dof_motor_ids:
                pos_addparam_result = self._dof_pos_vel_reader.addParam(motor_id)
                if not pos_addparam_result:
                    print("[ID:%03d] groupSyncRead addparam failed" % motor_id)
                    quit()
            if motor_id in self.gripper_motor_ids:
                pos_addparam_result = self._gripper_pos_vel_reader.addParam(motor_id)
                if not pos_addparam_result:
                    print("[ID:%03d] groupSyncRead addparam failed" % motor_id)
                    quit()
            if motor_id in self.arm_1_motor_ids:
                pos_addparam_result = self._arm_1_pos_vel_reader.addParam(motor_id)
                if not pos_addparam_result:
                    print("[ID:%03d] groupSyncRead addparam failed" % motor_id)
                    quit()
            if motor_id in self.arm_2_motor_ids:
                pos_addparam_result = self._arm_2_pos_vel_reader.addParam(motor_id)
                if not pos_addparam_result:
                    print("[ID:%03d] groupSyncRead addparam failed" % motor_id)
                    quit()
                    
        self.pos_scale = pos_scale if pos_scale is not None else DEFAULT_POS_SCALE
        self.vel_scale = vel_scale if vel_scale is not None else DEFAULT_VEL_SCALE
        self.current_pos = np.zeros(len(motor_ids))
        self.current_vel = np.zeros(len(motor_ids))

        self._sync_writers = {}

        self.OPEN_CLIENTS.add(self)

    @property
    def is_connected(self) -> bool:
        return self.port_handler.is_open

    def connect(self):
        """Connects to the Dynamixel motors.

        NOTE: This should be called after all DynamixelClients on the same
            process are created.
        """
        assert not self.is_connected, 'Client is already connected.'

        if self.port_handler.openPort():
            logging.info('Succeeded to open port: %s', self.port_name)
        else:
            raise OSError(
                ('Failed to open port at {} (Check that the device is powered '
                 'on and connected to your computer).').format(self.port_name))

        if self.port_handler.setBaudRate(self.baudrate):
            logging.info('Succeeded to set baudrate to %d', self.baudrate)
        else:
            raise OSError(
                ('Failed to set the baudrate to {} (Ensure that the device was '
                 'configured for this baudrate).').format(self.baudrate))

        # Start with all motors enabled.  NO, I want to set settings before enabled
        #self.set_torque_enabled(self.motor_ids, True)

    def disconnect(self):
        """Disconnects from the Dynamixel device."""
        if not self.is_connected:
            return
        if self.port_handler.is_using:
            logging.error('Port handler in use; cannot disconnect.')
            return
        # Ensure motors are disabled at the end.
        self.set_torque_enabled(self.motor_ids, False, retries=0)
        self.port_handler.closePort()
        if self in self.OPEN_CLIENTS:
            self.OPEN_CLIENTS.remove(self)

    # def read_all_pos_vel(self):
    #     self.read_arm_1_pos_vel()
    #     self.read_arm_2_pos_vel()
    #     return self.current_pos, self.current_vel

    def read_all_pos_vel(self):
        time_start = time.time()
        dxl_comm_result = self._all_pos_reader.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
            print("Failed to get present position and velocity of the arms!!!!!!!!!!")
        else:
            for i, motor_id in enumerate(self.motor_ids):
                pos_data = self._all_pos_reader.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                vel_data = self._all_pos_reader.getData(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
                self.current_pos[i] = float(unsigned_to_signed(pos_data, LEN_PRESENT_POSITION)) * self.pos_scale
                self.current_vel[i] = float(unsigned_to_signed(vel_data, LEN_PRESENT_VELOCITY)) * self.vel_scale
        # print('read_all_pos_vel time: ', time.time() - time_start)
        return self.current_pos, self.current_vel

    # def read_dof_pos_vel(self):
    #     dxl_comm_result = self._dof_pos_vel_reader.txRxPacket()
    #     if dxl_comm_result != COMM_SUCCESS:
    #         print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
    #         print("Failed to get present position and velocity of the dofs!!!!!!!!!!")
    #     else:
    #         for i, motor_id in enumerate(self.dof_motor_ids):
    #             pos_data = self._dof_pos_vel_reader.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    #             vel_data = self._dof_pos_vel_reader.getData(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
    #             self.current_pos[self.dof_idx[i]] = float(unsigned_to_signed(pos_data, LEN_PRESENT_POSITION)) * self.pos_scale
    #             self.current_vel[self.dof_idx[i]] = float(unsigned_to_signed(vel_data, LEN_PRESENT_VELOCITY)) * self.vel_scale
    #     return self.current_pos, self.current_vel

    # def read_gripper_pos_vel(self):
    #     dxl_comm_result = self._gripper_pos_vel_reader.txRxPacket()
    #     if dxl_comm_result != COMM_SUCCESS:
    #         print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
    #         print("Failed to get present position and velocity of the gripper!!!!!!!!!!")
    #     else:
    #         for i, motor_id in enumerate(self.gripper_motor_ids):
    #             pos_data = self._gripper_pos_vel_reader.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    #             vel_data = self._gripper_pos_vel_reader.getData(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
    #             self.current_pos[self.gripper_idx[i]] = float(unsigned_to_signed(pos_data, LEN_PRESENT_POSITION)) * self.pos_scale
    #             self.current_vel[self.gripper_idx[i]] = float(unsigned_to_signed(vel_data, LEN_PRESENT_VELOCITY)) * self.vel_scale
    #     return self.current_pos, self.current_vel

    def read_arm_1_pos_vel(self):
        dxl_comm_result = self._arm_1_pos_vel_reader.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
            print("Failed to get present position and velocity of the arm_1!!!!!!!!!!")
        else:
            for i, motor_id in enumerate(self.arm_1_motor_ids):
                pos_data = self._arm_1_pos_vel_reader.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                vel_data = self._arm_1_pos_vel_reader.getData(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
                self.current_pos[self.arm_1_idx[i]] = float(unsigned_to_signed(pos_data, LEN_PRESENT_POSITION)) * self.pos_scale
                self.current_vel[self.arm_1_idx[i]] = float(unsigned_to_signed(vel_data, LEN_PRESENT_VELOCITY)) * self.vel_scale
        return self.current_pos, self.current_vel

    def read_arm_2_pos_vel(self):
        dxl_comm_result = self._arm_2_pos_vel_reader.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
            print("Failed to get present position and velocity of the arm_2!!!!!!!!!!")
        else:
            for i, motor_id in enumerate(self.arm_2_motor_ids):
                pos_data = self._arm_2_pos_vel_reader.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                vel_data = self._arm_2_pos_vel_reader.getData(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
                self.current_pos[self.arm_2_idx[i]] = float(unsigned_to_signed(pos_data, LEN_PRESENT_POSITION)) * self.pos_scale
                self.current_vel[self.arm_2_idx[i]] = float(unsigned_to_signed(vel_data, LEN_PRESENT_VELOCITY)) * self.vel_scale
        return self.current_pos, self.current_vel

    def set_torque_enabled(self,
                           motor_ids: Sequence[int],
                           enabled: bool,
                           retries: int = -1,
                           retry_interval: float = 0.25):
        """Sets whether torque is enabled for the motors.

        Args:
            motor_ids: The motor IDs to configure.
            enabled: Whether to engage or disengage the motors.
            retries: The number of times to retry. If this is <0, will retry
                forever.
            retry_interval: The number of seconds to wait between retries.
        """
        remaining_ids = list(motor_ids)
        while remaining_ids:
            remaining_ids = self.write_byte(
                remaining_ids,
                int(enabled),
                ADDR_TORQUE_ENABLE,
            )
            if remaining_ids:
                logging.error('Could not set torque %s for IDs: %s',
                              'enabled' if enabled else 'disabled',
                              str(remaining_ids))
            if retries == 0:
                break
            time.sleep(retry_interval)
            retries -= 1

    def write_desired_pos(self, motor_ids: Sequence[int],
                          positions: np.ndarray):
        """Writes the given desired positions.

        Args:
            motor_ids: The motor IDs to write to.
            positions: The joint angles in radians to write.
        """
        assert len(motor_ids) == len(positions)

        # Convert to Dynamixel position space.
        positions = positions / self.pos_scale
        self.sync_write(motor_ids, positions, ADDR_GOAL_POSITION,
                        LEN_GOAL_POSITION)

    def write_byte(
            self,
            motor_ids: Sequence[int],
            value: int,
            address: int,
    ) -> Sequence[int]:
        """Writes a value to the motors.

        Args:
            motor_ids: The motor IDs to write to.
            value: The value to write to the control table.
            address: The control table address to write to.

        Returns:
            A list of IDs that were unsuccessful.
        """
        self.check_connected()
        errored_ids = []
        for motor_id in motor_ids:
            comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, address, value)
            success = self.handle_packet_result(
                comm_result, dxl_error, motor_id, context='write_byte')
            if not success:
                errored_ids.append(motor_id)
        return errored_ids

    def sync_write(self, motor_ids: Sequence[int],
                   values: Sequence[Union[int, float]], address: int,
                   size: int):
        """Writes values to a group of motors.

        Args:
            motor_ids: The motor IDs to write to.
            values: The values to write.
            address: The control table address to write to.
            size: The size of the control table value being written to.
        """
        self.check_connected()
        key = (address, size)
        if key not in self._sync_writers:
            self._sync_writers[key] = self.dxl.GroupSyncWrite(
                self.port_handler, self.packet_handler, address, size)
        sync_writer = self._sync_writers[key]

        errored_ids = []
        for motor_id, desired_pos in zip(motor_ids, values):
            value = signed_to_unsigned(int(desired_pos), size=size)
            value = value.to_bytes(size, byteorder='little')
            success = sync_writer.addParam(motor_id, value)
            if not success:
                errored_ids.append(motor_id)

        if errored_ids:
            logging.error('Sync write failed for: %s', str(errored_ids))

        comm_result = sync_writer.txPacket()
        self.handle_packet_result(comm_result, context='sync_write')

        sync_writer.clearParam()

    def check_connected(self):
        """Ensures the robot is connected."""
        if self.lazy_connect and not self.is_connected:
            self.connect()
        if not self.is_connected:
            raise OSError('Must call connect() first.')

    def handle_packet_result(self,
                             comm_result: int,
                             dxl_error: Optional[int] = None,
                             dxl_id: Optional[int] = None,
                             context: Optional[str] = None):
        """Handles the result from a communication request."""
        error_message = None
        if comm_result != self.dxl.COMM_SUCCESS:
            error_message = self.packet_handler.getTxRxResult(comm_result)
        elif dxl_error is not None:
            error_message = self.packet_handler.getRxPacketError(dxl_error)
        if error_message:
            if dxl_id is not None:
                error_message = '[Motor ID: {}] {}'.format(
                    dxl_id, error_message)
            if context is not None:
                error_message = '> {}: {}'.format(context, error_message)
            logging.error(error_message)
            return False
        return True

    def convert_to_unsigned(self, value: int, size: int) -> int:
        """Converts the given value to its unsigned representation."""
        if value < 0:
            max_value = (1 << (8 * size)) - 1
            value = max_value + value
        return value

    def __enter__(self):
        """Enables use as a context manager."""
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, *args):
        """Enables use as a context manager."""
        self.disconnect()

    def __del__(self):
        """Automatically disconnect on destruction."""
        self.disconnect()


# Register global cleanup function.
atexit.register(dynamixel_cleanup_handler)

if __name__ == '__main__':
    import argparse
    import itertools

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--motors',
        required=True,
        help='Comma-separated list of motor IDs.')
    parser.add_argument(
        '-d',
        '--device',
        default='/dev/ttyUSB0',
        help='The Dynamixel device to connect to.')
    parser.add_argument(
        '-b', '--baud', default=1000000, help='The baudrate to connect with.')
    parsed_args = parser.parse_args()

    motors = [int(motor) for motor in parsed_args.motors.split(',')]

    way_points = [np.zeros(len(motors)), np.full(len(motors), np.pi)]

    with DynamixelClient(motors, parsed_args.device,
                         parsed_args.baud) as dxl_client:
        for step in itertools.count():
            if step > 0 and step % 50 == 0:
                way_point = way_points[(step // 100) % len(way_points)]
                print('Writing: {}'.format(way_point.tolist()))
                dxl_client.write_desired_pos(motors, way_point)
            read_start = time.time()
            pos_now, vel_now, cur_now = dxl_client.read_pos_vel_cur()
            if step % 5 == 0:
                print('[{}] Frequency: {:.2f} Hz'.format(
                    step, 1.0 / (time.time() - read_start)))
                print('> Pos: {}'.format(pos_now.tolist()))
                print('> Vel: {}'.format(vel_now.tolist()))
                print('> Cur: {}'.format(cur_now.tolist()))
