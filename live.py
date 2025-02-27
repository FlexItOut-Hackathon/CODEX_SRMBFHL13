import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1  
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

screen_width = 1920  
screen_height = 1080  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {frame_width}x{frame_height}")

is_portrait = frame_height > frame_width  
rotation_angle = 90 if is_portrait else 0 
cv2.namedWindow('FLEX-IT-OUT - Fitness Detection', cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty('FLEX-IT-OUT - Fitness Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Variables for activity detection
activity_counts = {'squats': 0, 'crunches': 0, 'pushups': 0}
activity_states = {'squats': 'standing', 'crunches': 'up', 'pushups': 'up'}  

def calculate_angle(a, b, c):
    """Calculate angle between three points (in degrees)."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def detect_activity(landmarks):
    """Detect squats, crunches, and push-ups based on keypoints."""
    global activity_counts, activity_states

    landmark_list = landmarks.landmark
    
    # Common landmarks
    left_hip = landmark_list[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmark_list[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmark_list[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_hip = landmark_list[mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = landmark_list[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = landmark_list[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_shoulder = landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmark_list[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmark_list[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_shoulder = landmark_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmark_list[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmark_list[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    # Detect all activities simultaneously
    feedback_squats, count_squats = detect_squat(landmark_list)
    feedback_crunches, count_crunches = detect_crunch(landmark_list)
    feedback_pushups, count_pushups = detect_pushup(landmark_list)

    return {
        'squats': (feedback_squats, count_squats),
        'crunches': (feedback_crunches, count_crunches),
        'pushups': (feedback_pushups, count_pushups)
    }

def detect_squat(landmarks):
    """Detect squats based on keypoints."""
    global activity_counts, activity_states

    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    hip_to_knee_dist = abs(left_hip.y - left_knee.y)
    squat_threshold = 0.2

    feedback = ""
    current_state = activity_states['squats']

    if left_knee_angle < 120 and right_knee_angle < 120 and hip_to_knee_dist > squat_threshold:
        feedback = "Good form! Keep going."
        current_state = 'squatting'
    elif hip_to_knee_dist < squat_threshold / 2:
        feedback = "Stand up fully."
        current_state = 'standing'
    else:
        feedback = "Bend your knees more."

    if activity_states['squats'] == 'standing' and current_state == 'squatting':
        activity_states['squats'] = 'squatting'
    elif activity_states['squats'] == 'squatting' and current_state == 'standing':
        activity_counts['squats'] += 1
        activity_states['squats'] = 'standing'
        print(f"Squats: {activity_counts['squats']}")

    return feedback, activity_counts['squats']

def detect_crunch(landmarks):
    """Detect crunches based on keypoints."""
    global activity_counts, activity_states

    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    torso_angle = calculate_angle(left_hip, left_shoulder, left_hip)
    shoulder_to_hip_dist = abs(left_shoulder.y - left_hip.y)
    crunch_threshold = 0.15

    feedback = ""
    current_state = activity_states['crunches']

    if torso_angle < 45 and shoulder_to_hip_dist < crunch_threshold:
        feedback = "Great crunch! Keep going."
        current_state = 'down'
    elif shoulder_to_hip_dist > crunch_threshold * 2:
        feedback = "Sit up fully."
        current_state = 'up'
    else:
        feedback = "Curl your torso more."

    if activity_states['crunches'] == 'up' and current_state == 'down':
        activity_states['crunches'] = 'down'
    elif activity_states['crunches'] == 'down' and current_state == 'up':
        activity_counts['crunches'] += 1
        activity_states['crunches'] = 'up'
        print(f"Crunches: {activity_counts['crunches']}")

    return feedback, activity_counts['crunches']

def detect_pushup(landmarks):
    """Detect push-ups based on keypoints."""
    global activity_counts, activity_states

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    pushup_threshold = 90  # Elbow angle for down position

    feedback = ""
    current_state = activity_states['pushups']

    if left_elbow_angle < pushup_threshold and right_elbow_angle < pushup_threshold:
        feedback = "Good push-up! Lower more."
        current_state = 'down'
    elif left_elbow_angle > 160 and right_elbow_angle > 160:
        feedback = "Push up fully."
        current_state = 'up'
    else:
        feedback = "Bend your elbows more."

    if activity_states['pushups'] == 'up' and current_state == 'down':
        activity_states['pushups'] = 'down'
    elif activity_states['pushups'] == 'down' and current_state == 'up':
        activity_counts['pushups'] += 1
        activity_states['pushups'] = 'up'
        print(f"Push-ups: {activity_counts['pushups']}")

    return feedback, activity_counts['pushups']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Rotate frame if in portrait mode
    if is_portrait:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # Swap width and height for full-screen in portrait
        temp_width, temp_height = screen_width, screen_height
        screen_width, screen_height = temp_height, temp_width

    # Resize frame to match screen resolution (rotated if necessary)
    frame = cv2.resize(frame, (screen_width, screen_height))

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Draw landmarks and process activity detection
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # Detect all activities
        results = detect_activity(results.pose_landmarks)

        # Display counts and feedback for all activities
        y_offset = 30
        for activity, (feedback, count) in results.items():
            cv2.putText(frame, f"{activity.capitalize()}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, feedback, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30

    
    cv2.imshow('FLEX-IT-OUT - Fitness Detection', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
pose.close()