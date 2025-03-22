import cv2
import mediapipe as mp
import json
import numpy as np

def generate_model_data():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Use your model video path
    video = cv2.VideoCapture('Model/SquatMasterSide.mov')
    landmarks_data = []
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Convert landmarks to list format
            frame_landmarks = []
            for landmark in results.pose_landmarks.landmark:
                frame_landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            landmarks_data.append(frame_landmarks)
    
    video.release()
    pose.close()
    
    # Save to JSON file
    with open('model_landmarks.json', 'w') as f:
        json.dump(landmarks_data, f)

if __name__ == "__main__":
    generate_model_data() 