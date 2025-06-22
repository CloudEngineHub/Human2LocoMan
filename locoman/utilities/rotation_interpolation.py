from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def interpolate_rpy(rpy_start, rpy_end, t):
    """
    Interpolate between two RPY rotations.
    
    Parameters:
    rpy_start (array-like): Start rotation as [roll, pitch, yaw]
    rpy_end (array-like): End rotation as [roll, pitch, yaw]
    t (float): Interpolation parameter, 0 <= t <= 1
    
    Returns:
    np.array: Interpolated rotation as [roll, pitch, yaw]
    """
    # Convert RPY to quaternions
    r_start = R.from_euler('xyz', rpy_start)
    r_end = R.from_euler('xyz', rpy_end)

    key_times = [0, 1]
    slerp = Slerp(key_times, R.concatenate([r_start, r_end]))
    
    # Perform slerp
    slerp_rot = slerp(t)
    
    # Convert back to RPY
    interpolated_rpy = slerp_rot.as_euler('xyz')
    
    return interpolated_rpy