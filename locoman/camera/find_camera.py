import subprocess
import re
import cv2

def list_video_devices():
    # Run the v4l2-ctl command to list devices
    result = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    
    # Parse the output to map camera names to device paths
    device_map = {}
    lines = output.split('\n')
    current_device_name = ""
    
    for line in lines:
        if line.strip() == "":
            continue
        if not line.startswith('\t'):
            # This is a device name line
            current_device_name = line.strip()
        else:
            # This is a device path line
            device_path = line.strip()
            if current_device_name in device_map:
                device_map[current_device_name].append(device_path)
            else:
                device_map[current_device_name] = [device_path]
    
    return device_map

def find_device_path_by_name(device_map, name):
    for device_name, device_paths in device_map.items():
        if name in device_name:
            return device_paths[0]  # Return the first device path found
    return None

# List all video devices
device_map = list_video_devices()

# Print the devices for inspection
for name, paths in device_map.items():
    print(f"Device: {name}, Paths: {paths}")

# Find the device path by the camera name (replace 'Integrated Webcam' with your camera name)
camera_name = "3D USB Camera"
device_path = find_device_path_by_name(device_map, camera_name)

if device_path:
    print(f"Opening camera: {camera_name} at {device_path}")
    
    # Open the camera using OpenCV
    camera = cv2.VideoCapture(device_path)
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
    else:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

        while camera.isOpened():
            ret, frame = camera.read()
            if ret:
                # Write the frame to the output file
                out.write(frame)

                # Display the frame
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished
        camera.release()
        out.release()
        cv2.destroyAllWindows()
else:
    print(f"Camera '{camera_name}' not found.")