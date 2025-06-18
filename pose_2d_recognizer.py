import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# Initialize OpenCV VideoCapture
cap = cv2.VideoCapture(0)

def angle_to_letter(right_angle, left_angle):
    """
    converts two angles to the corresponding letter according to the semaphore alphabet
    
    Args:
        right_angle: float representing the angle of the right arm expected to be (0, 45, 90, 135, 180, 225, 270, 315)
        left_angle: float representing the angle of the left arm expected to be (0, 45, 90, 135, 180, 225, 270, 315)
    
    Returns:
        string: Capital Letter if a matching letter has been found or '-' if no letter was found
    """
    semaphore_letters = {
        'A': (45, 0),
        'B': (90, 0),
        'C': (135, 0),
        'D': (180, 0),
        'E': (0, 225),
        'F': (0, 270),
        'G': (0, 315),
        'H': (90, 45),
        'I': (135, 45),
        'J': (180, 270),
        'K': (45, 180),
        'L': (45, 225),
        'M': (45, 270),
        'N': (45, 315),
        'O': (90, 135),
        'P': (90, 180),
        'Q': (90, 225),
        'R': (90, 270),
        'S': (90, 315),
        'T': (135, 180),
        'U': (135, 225),
        'V': (180, 315),
        'W': (225, 270),
        'X': (225, 315),
        'Y': (135, 270),
        'Z': (315, 270)
    }
    for letter, angles in semaphore_letters.items():
        if right_angle == angles[0] and left_angle == angles[1]:
            return letter
    return '-'

def calculate_angle(vector):
    """
    Calculate the angle from a vertical vector to vector in degrees, in the range [0, 360).
    
    Args:
        vector: NumPy array representing the first vector.
    
    Returns:
        float: Angle in degrees, in [0, 360).
    """
    vertical_vector = [0, 1]
    # Compute the angle using arctan2 for vector1 relative to vector2
    angle_rad = np.arctan2(vector[1], vector[0]) - np.arctan2(vertical_vector[1], vertical_vector[0])
    # Normalize angle to [0, 2π)
    angle_rad = (angle_rad + 2 * np.pi) % (2 * np.pi)
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def snap_angle(angle):
    """
    Calculate the closest angle equal to a multiple of 45.
    
    Args:
        angle: float representing na angle between 0 and 360.
    
    Returns:
        float: Snapped Angle in degrees, in [0, 360).
    """
    # snap angle to closest multiple of 45
    snapped_angle =  45 * round(angle/45)
    # convert angle of 360 to 0
    if snapped_angle == 360:
        snapped_angle = 0
    return snapped_angle

def show_window():
    """
    Shows the window and scales its contents without stretching
    """
    # Get window size
    win_x, win_y, win_w, win_h = cv2.getWindowImageRect('Semaphore Recognition')

    # Original frame size
    frame_h, frame_w = frame.shape[:2]
    scale = min(win_w / frame_w, win_h / frame_h)
    resized_w, resized_h = int(frame_w * scale), int(frame_h * scale)

    # Resize frame with aspect ratio preserved
    resized_frame = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # Create black background
    display_frame = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    x_offset = (win_w - resized_w) // 2
    y_offset = (win_h - resized_h) // 2

    # Paste resized frame onto center
    display_frame[y_offset:y_offset + resized_h, x_offset:x_offset + resized_w] = resized_frame

    # Show padded frame
    cv2.imshow('Semaphore Recognition', display_frame)

cv2.namedWindow('Semaphore Recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Semaphore Recognition', 1280, 720)
while cv2.getWindowProperty('Semaphore Recognition', cv2.WND_PROP_VISIBLE) >= 1:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Get image dimensions
        height, width, _ = frame.shape

        # Function to get pixel coordinates from normalized landmarks
        def get_coords(landmark):
            return np.array([int(landmark.x * width), int(landmark.y * height)])

        # Define landmark indices
        landmarks = {
            "left_shoulder":    get_coords(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]),
            "left_wrist":       get_coords(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]),
            "right_shoulder":   get_coords(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]),
            "right_wrist":      get_coords(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        }

        # Draw connections
        cv2.line(frame, tuple(landmarks["left_shoulder"]), tuple(landmarks["left_wrist"]), (0, 0, 255), 2)
        cv2.line(frame, tuple(landmarks["right_shoulder"]), tuple(landmarks["right_wrist"]), (0, 0, 255), 2)

        # Draw landmarks
        for name, coords in landmarks.items():
            cv2.circle(frame, tuple(coords), 5, (255, 255, 255), -1)
        
        # Get vectors
        left_shoulder_wrist = landmarks["left_wrist"] - landmarks["left_shoulder"]
        right_shoulder_wrist = landmarks["right_wrist"] - landmarks["right_shoulder"]

        # Calculate angle
        left_angle = calculate_angle(left_shoulder_wrist)
        right_angle = calculate_angle(right_shoulder_wrist)

        # get closest snapped angle
        left_angle_snapped = snap_angle(left_angle)
        right_angle_snapped = snap_angle(right_angle)

        # get matching letter
        letter = angle_to_letter(right_angle_snapped, left_angle_snapped)

        # display angle
        cv2.putText(frame, f"{left_angle:.1f} {left_angle_snapped}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{right_angle:.1f} {right_angle_snapped}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # display letter
        cv2.putText(frame, f"{letter}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    show_window()

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()