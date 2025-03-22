import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation
import math

def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def get_frame_landmarks(frame, pose):
    """Extract landmarks from a frame"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        return landmarks
    return None

def extract_angles(landmarks, frame_shape):
    """Extract relevant angles from landmarks"""
    if not landmarks:
        return None
    
    h, w = frame_shape[:2]
    
    # Convert landmarks to pixel coordinates
    def get_point(landmark):
        return [int(landmark.x * w), int(landmark.y * h)]
    
    # Get key points
    left_shoulder = get_point(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER])
    right_shoulder = get_point(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER])
    left_hip = get_point(landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP])
    right_hip = get_point(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP])
    left_knee = get_point(landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE])
    right_knee = get_point(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE])
    left_ankle = get_point(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE])
    right_ankle = get_point(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE])
    
    # Calculate angles
    # Shoulder angles (relative to torso)
    left_shoulder_angle = calculate_angle(left_hip, left_shoulder, [left_shoulder[0] + 100, left_shoulder[1]])
    right_shoulder_angle = calculate_angle(right_hip, right_shoulder, [right_shoulder[0] + 100, right_shoulder[1]])
    
    # Hip angles
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    
    # Knee angles
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    # Ankle angles (dorsiflexion)
    left_ankle_angle = calculate_angle(left_knee, left_ankle, [left_ankle[0], left_ankle[1] + 100])
    right_ankle_angle = calculate_angle(right_knee, right_ankle, [right_ankle[0], right_ankle[1] + 100])
    
    return {
        'left_shoulder': left_shoulder_angle,
        'right_shoulder': right_shoulder_angle,
        'left_hip': left_hip_angle,
        'right_hip': right_hip_angle,
        'left_knee': left_knee_angle,
        'right_knee': right_knee_angle,
        'left_ankle': left_ankle_angle,
        'right_ankle': right_ankle_angle
    }

def process_video():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Open the input video
    cap = cv2.VideoCapture('User Input/UserInput.mov')
    
    if not cap.isOpened():
        print("Error: Could not open input video")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer object with Mac-compatible codec
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(
        'UserInputLabeled.mp4', 
        fourcc, 
        fps, 
        (width, height),
        isColor=True
    )
    
    if not out.isOpened():
        print("Error: Could not create output video file")
        cap.release()
        return

    print("Processing video...")
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Create black background
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Process the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw the pose landmarks
            mp_drawing.draw_landmarks(
                black_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            # Get landmarks for coloring
            landmarks = results.pose_landmarks.landmark
            
            # Define key points with their colors and labels
            key_points = {
                'Left Shoulder': (mp_pose.PoseLandmark.LEFT_SHOULDER, (0, 255, 0)),      # Green
                'Right Shoulder': (mp_pose.PoseLandmark.RIGHT_SHOULDER, (0, 255, 255)),  # Yellow
                'Left Hip': (mp_pose.PoseLandmark.LEFT_HIP, (255, 0, 0)),               # Blue
                'Right Hip': (mp_pose.PoseLandmark.RIGHT_HIP, (255, 0, 255)),           # Magenta
                'Left Knee': (mp_pose.PoseLandmark.LEFT_KNEE, (0, 0, 255)),             # Red
                'Right Knee': (mp_pose.PoseLandmark.RIGHT_KNEE, (255, 165, 0)),         # Orange
                'Left Ankle': (mp_pose.PoseLandmark.LEFT_ANKLE, (255, 255, 0)),         # Cyan
                'Right Ankle': (mp_pose.PoseLandmark.RIGHT_ANKLE, (128, 0, 128))        # Purple
            }
            
            # Draw colored circles and labels for key points
            for name, (landmark, color) in key_points.items():
                point = landmarks[landmark.value]
                x = int(point.x * width)
                y = int(point.y * height)
                cv2.circle(black_frame, (x, y), 8, color, -1)  # Larger circles
                cv2.putText(black_frame, name, (x + 10, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame
        out.write(black_frame)
        frame_count += 1

    print(f"Processed {frame_count} frames")
    print(f"Created: UserInputLabeled.mp4")
    
    # Release everything
    cap.release()
    out.release()
    pose.close()
    print("Done")

if __name__ == "__main__":
    process_video() 