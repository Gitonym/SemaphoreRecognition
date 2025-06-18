import cv2
import mediapipe as mp
import math
import os
import argparse
import time

# call main.py for graphic semaphore detection
# call main.py --save-data for saving the detected images in corresponding folder, saving only starts after letter was held for one second
# call main.py --save-data --save-letter X for saving all images in the X folder, saving only starts after X was held for one second

def vector_from_points(a, b):
    return [b.x - a.x, b.y - a.y, b.z - a.z]


def calculate_angle(v1, v2):
    x1, y1 = v1[0], v1[1]
    x2, y2 = v2[0], v2[1]
    angle1 = math.atan2(y1, x1)
    angle2 = math.atan2(y2, x2)
    angle = angle1 - angle2
    return math.degrees(angle) % 360


def angle_to_semaphore_angle(angle):
    return (round(angle / 45) * 45) % 360


def angle_to_letter(right_angle, left_angle):
    semaphore_letters = {
        'A': (45, 0), 'B': (90, 0), 'C': (135, 0), 'D': (180, 0),
        'E': (0, 225), 'F': (0, 270), 'G': (0, 315),
        'H': (90, 45), 'I': (135, 45), 'J': (180, 270),
        'K': (45, 180), 'L': (45, 225), 'M': (45, 270), 'N': (45, 315),
        'O': (90, 135), 'P': (90, 180), 'Q': (90, 225),
        'R': (90, 270), 'S': (90, 315), 'T': (135, 180),
        'U': (135, 225), 'V': (180, 315), 'W': (225, 270),
        'X': (225, 315), 'Y': (135, 270), 'Z': (315, 270)
    }
    for letter, angles in semaphore_letters.items():
        if right_angle == angles[0] and left_angle == angles[1]:
            return letter
    return '-'


def draw_vector(image, start, end, color=(0, 255, 0), thickness=3):
    h, w, _ = image.shape
    start_pt = (int(start.x * w), int(start.y * h))
    end_pt = (int(end.x * w), int(end.y * h))
    cv2.arrowedLine(image, start_pt, end_pt, color, thickness, tipLength=0.2)


def process_frame(image, pose, mp_pose):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    letter = '-'
    r_angle_rounded = l_angle_rounded = 0

    if results.pose_world_landmarks:
        lm3d = results.pose_world_landmarks.landmark

        r_shoulder = lm3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_wrist = lm3d[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_shoulder = lm3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_wrist = lm3d[mp_pose.PoseLandmark.LEFT_WRIST.value]

        r_vec = vector_from_points(r_shoulder, r_wrist)
        l_vec = vector_from_points(l_shoulder, l_wrist)
        r_vert = vector_from_points(r_shoulder, type('P', (), {'x': r_shoulder.x, 'y': r_shoulder.y + 10, 'z': r_shoulder.z}))
        l_vert = vector_from_points(l_shoulder, type('P', (), {'x': l_shoulder.x, 'y': l_shoulder.y + 10, 'z': l_shoulder.z}))

        r_angle = calculate_angle(r_vec, r_vert)
        l_angle = calculate_angle(l_vec, l_vert)
        r_angle_rounded = angle_to_semaphore_angle(r_angle)
        l_angle_rounded = angle_to_semaphore_angle(l_angle)
        letter = angle_to_letter(r_angle_rounded, l_angle_rounded)

    return letter, r_angle_rounded, l_angle_rounded, results.pose_landmarks


def save_frame(image, letter, save_dir="data"):
    folder = os.path.join(save_dir, letter)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"frame_{int(time.time() * 1000)}.jpg")
    cv2.imwrite(filename, image)


def update_letter_timer(current_letter, last_letter, letter_start_time, current_time):
    if current_letter == last_letter:
        elapsed = current_time - letter_start_time
    else:
        last_letter = current_letter
        letter_start_time = current_time
        elapsed = 0
    return last_letter, letter_start_time, elapsed


def check_activation(letter, save_letter, activated, elapsed, hold_time):
    if activated:
        return True
    if letter == save_letter and elapsed > hold_time:
        return True
    return False


def draw_hold_time(frame, elapsed, hold_time, save_letter=None, activated=False):
    if save_letter and not activated:
        countdown = min(elapsed, hold_time)
        text = f'Hold {save_letter}: {countdown:.2f}s'
    else:
        countdown = min(elapsed, hold_time)
        text = f'Hold Time: {countdown:.2f}s'
    cv2.putText(frame, text, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def run(save_data=False, save_letter=None):
    HOLD_TIME = 1
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=2, enable_segmentation=False)
    cap = cv2.VideoCapture(0)

    last_letter = '-'
    letter_start_time = 0
    activated = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        original_frame = frame.copy()
        letter, r_angle, l_angle, landmarks = process_frame(frame, pose, mp_pose)
        current_time = time.time()

        last_letter, letter_start_time, elapsed = update_letter_timer(letter, last_letter, letter_start_time, current_time)

        if save_letter:
            activated = check_activation(letter, save_letter, activated, elapsed, HOLD_TIME)
            if activated and save_data:
                save_frame(original_frame, save_letter)
        else:
            if letter != '-' and elapsed > HOLD_TIME and save_data:
                save_frame(original_frame, letter)

        # Draw vectors
        if landmarks:
            draw_vector(frame, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value], (255, 0, 0))
            draw_vector(frame, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value], (255, 0, 0))

        # Draw info text
        cv2.putText(frame, f'Letter: {letter}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Right Angle: {r_angle}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Left Angle: {l_angle}', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        draw_hold_time(frame, elapsed, HOLD_TIME, save_letter, activated)

        cv2.imshow('3D Pose Semaphore', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-data', action='store_true', help='Save frames to dataset folders')
    parser.add_argument('--save-letter', type=str, help='Letter folder to save images into regardless of detected letter after activation')
    args = parser.parse_args()
    run(save_data=args.save_data, save_letter=args.save_letter)

