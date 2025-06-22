import cv2
import time

# Function to list available cameras
def list_cameras():
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

available_cameras = list_cameras()
print("Available cameras:", available_cameras)

# Check if at least two cameras are available
if len(available_cameras) < 2:
    print("Error: Less than two cameras found.")
    exit()

# Open the cameras using the detected indices
cap1 = cv2.VideoCapture(available_cameras[0])
cap2 = cv2.VideoCapture(available_cameras[1])
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one of the cameras.")
    exit()
    
# Set the desired frame rate
desired_fps = 60
cap1.set(cv2.CAP_PROP_FPS, desired_fps)
cap2.set(cv2.CAP_PROP_FPS, desired_fps)
actual_fps1 = cap1.get(cv2.CAP_PROP_FPS)
actual_fps2 = cap2.get(cv2.CAP_PROP_FPS)
print(f"Attempted to set FPS to {desired_fps}, actual FPS of camera 1 is {actual_fps1}")
print(f"Attempted to set FPS to {desired_fps}, actual FPS of camera 2 is {actual_fps2}")

# cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
# cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

actual_width1 = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height1 = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_width2 = cap2.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height2 = cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f'camera 1 resolution: [{actual_width1}, {actual_height1}]')
print(f'camera 2 resolution: [{actual_width2}, {actual_height2}]')

frame_duration = 1 / desired_fps

while True:
    start_time = time.time()
    # Capture frame-by-frame from the first camera
    ret1, frame1 = cap1.read()
    # Capture frame-by-frame from the second camera
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Could not read frame from one of the cameras.")
        break

    # Display the resulting frames
    
    # frame1 = cv2.resize(frame1, (1280, 480))
    
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Wait to maintain the frame rate
    elapsed_time = time.time() - start_time
    sleep_time = frame_duration - elapsed_time
    if sleep_time > 0:
        time.sleep(sleep_time)

    freq = 1 / (time.time() - start_time)
    # print('freq', freq)

# Release the camera resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()