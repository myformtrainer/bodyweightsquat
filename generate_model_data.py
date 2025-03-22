import cv2
import mediapipe as mp
import json
import numpy as np
import os

def generate_model_data():
    print("Starting model data generation...")
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Get absolute path to the model video
    current_dir = os.getcwd()
    video_path = os.path.join(current_dir, 'Model', 'SquatMasterSide.mov')
    
    if not os.path.exists(video_path):
        print(f"Error: Model video not found at {video_path}")
        return
    
    print(f"Loading video from: {video_path}")
    video = cv2.VideoCapture(video_path)
    landmarks_data = []
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
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
    
    print(f"Processed {frame_count} frames")
    print(f"Captured landmarks for {len(landmarks_data)} frames")
    
    # Save to JSON file
    output_path = os.path.join(current_dir, 'model_landmarks.json')
    with open(output_path, 'w') as f:
        json.dump(landmarks_data, f)
    
    print(f"Saved model data to: {output_path}")

if __name__ == "__main__":
    generate_model_data() 