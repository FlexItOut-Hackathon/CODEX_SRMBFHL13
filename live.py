import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1  
)
mp_drawing = mp.solutions.drawing_utils

class ExerciseTracker:
    def __init__(self):
        self.exercise_type = None
        self.squat_tracker = Squat()
        self.pushup_tracker = PushUp()
        self.hammer_curl_tracker = HammerCurl()
    
    def set_exercise(self, exercise):
        self.exercise_type = exercise
    
    def track(self, landmark_list):
        if self.exercise_type == "Squat":
            return self.squat_tracker.track_squat(landmark_list)
        elif self.exercise_type == "PushUp":
            return self.pushup_tracker.track_push_up(landmark_list)
        elif self.exercise_type == "HammerCurl":
            return self.hammer_curl_tracker.track_hammer_curl(landmark_list)
        return None

class Squat:
    def __init__(self):
        self.counter = 0
        self.stage = None
    
    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle < 180 else 360 - angle
    
    def track_squat(self, landmark_list):
        hip = landmark_list[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmark_list[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmark_list[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        angle = self.calculate_angle(hip, knee, ankle)
        if angle > 170:
            self.stage = "Up"
        elif angle < 90 and self.stage == "Up":
            self.stage = "Down"
            self.counter += 1
        return self.counter, angle, self.stage

class PushUp:
    def __init__(self, up_tolerance=5, down_tolerance=5, min_time_between_counts=1):
        self.counter = 0
        self.stage = "Initial"
        self.last_counter_update = time.time()
        self.up_tolerance = up_tolerance
        self.down_tolerance = down_tolerance
        self.min_time_between_counts = min_time_between_counts

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle < 180 else 360 - angle

    def track_push_up(self, landmark_list):
        shoulder = landmark_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        elbow = landmark_list[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
        wrist = landmark_list[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        up_threshold = 150 - self.up_tolerance
        down_threshold = 70 + self.down_tolerance
        
        if angle > up_threshold:
            self.stage = "Up"
        elif angle < down_threshold and self.stage == "Up":
            self.stage = "Down"
            if time.time() - self.last_counter_update > self.min_time_between_counts:
                self.counter += 1
                self.last_counter_update = time.time()
        
        return self.counter, angle, self.stage

class HammerCurl:
    def __init__(self):
        self.counter_right = 0
        self.counter_left = 0
        self.stage_right = "Up"
        self.stage_left = "Up"
    
    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle < 180 else 360 - angle
    
    def track_hammer_curl(self, landmark_list):
        shoulder = landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        elbow = landmark_list[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        wrist = landmark_list[mp_pose.PoseLandmark.LEFT_WRIST.value]
        angle_left = self.calculate_angle(shoulder, elbow, wrist)
        
        shoulder = landmark_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmark_list[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmark_list[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        angle_right = self.calculate_angle(shoulder, elbow, wrist)
        
        if angle_right > 150:
            self.stage_right = "Up"
        elif angle_right < 50 and self.stage_right == "Up":
            self.stage_right = "Down"
            self.counter_right += 1
        
        if angle_left > 150:
            self.stage_left = "Up"
        elif angle_left < 50 and self.stage_left == "Up":
            self.stage_left = "Down"
            self.counter_left += 1
        
        return self.counter_right, self.stage_right, self.counter_left, self.stage_left

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow('FLEX-IT-OUT - Fitness Detection', cv2.WINDOW_NORMAL)
cv2.namedWindow('Exercise Stats', cv2.WINDOW_NORMAL)

exercise_tracker = ExerciseTracker()
selected_exercise = None

while selected_exercise not in ["Squat", "PushUp", "HammerCurl"]:
    selected_exercise = input("Select exercise (Squat, PushUp, HammerCurl): ")
exercise_tracker.set_exercise(selected_exercise)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    stats_frame = np.ones((400, 500, 3), dtype=np.uint8) * 50
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        tracking_results = exercise_tracker.track(results.pose_landmarks.landmark)
        if tracking_results:
            if selected_exercise == "HammerCurl":
                right_count, right_stage, left_count, left_stage = tracking_results
                cv2.putText(stats_frame, f"Hammer Curl Right: {right_count} ({right_stage})", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(stats_frame, f"Hammer Curl Left: {left_count} ({left_stage})", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                count, _, stage = tracking_results
                cv2.putText(stats_frame, f"{selected_exercise}: {count} ({stage})", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imshow('FLEX-IT-OUT - Fitness Detection', frame)
    cv2.imshow('Exercise Stats', stats_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()