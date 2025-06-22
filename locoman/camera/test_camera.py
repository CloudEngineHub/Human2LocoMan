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
if len(available_cameras) < 1:
    print("Error: no camera found.")
    exit()

# Open the cameras using the detected indices
cap1 = cv2.VideoCapture(available_cameras[0])
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap1.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Set the desired frame rate
desired_fps = 60
cap1.set(cv2.CAP_PROP_FPS, desired_fps)
actual_fps = cap1.get(cv2.CAP_PROP_FPS)
print(f"Attempted to set FPS to {desired_fps}, actual FPS is {actual_fps}")

frame_duration = 1 / desired_fps

while True:
    start_time = time.time()
    # Capture frame-by-frame from the first camera
    ret1, frame1 = cap1.read()

    if not ret1:
        print("Error: Could not read frame.")
        break

    # Display the resulting frames
    cv2.imshow('Camera 1', frame1)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Wait to maintain the frame rate
    elapsed_time = time.time() - start_time
    sleep_time = frame_duration - elapsed_time
    if sleep_time > 0:
        time.sleep(sleep_time)

# Release the camera resources
cap1.release()
cv2.destroyAllWindows()