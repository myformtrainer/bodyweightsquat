import cv2
import mediapipe as mp
import numpy as np

def process_video():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Open the input video with correct path
    input_path = 'Model/SquatMasterSide.mov'  # Updated path
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open input video '{input_path}'")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer object with Mac-compatible codec
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(
        'SquatLabeledSide.mp4', 
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
    print(f"Reading from: {input_path}")
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
            
            # Get specific landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Define key points we want to highlight with unique colors
            key_points = {
                'Left Shoulder': (mp_pose.PoseLandmark.LEFT_SHOULDER, (0, 255, 0)),      # Green
                'Right Shoulder': (mp_pose.PoseLandmark.RIGHT_SHOULDER, (0, 255, 255)),  # Yellow
                'Left Hip': (mp_pose.PoseLandmark.LEFT_HIP, (255, 0, 0)),                # Blue
                'Right Hip': (mp_pose.PoseLandmark.RIGHT_HIP, (255, 0, 255)),           # Magenta
                'Left Knee': (mp_pose.PoseLandmark.LEFT_KNEE, (0, 0, 255)),             # Red
                'Right Knee': (mp_pose.PoseLandmark.RIGHT_KNEE, (255, 165, 0)),         # Orange
                'Left Ankle': (mp_pose.PoseLandmark.LEFT_ANKLE, (255, 255, 0)),         # Cyan
                'Right Ankle': (mp_pose.PoseLandmark.RIGHT_ANKLE, (128, 0, 128))        # Purple
            }
            
            # Draw circles and labels for key points
            for name, (landmark, color) in key_points.items():
                point = landmarks[landmark]
                h, w, _ = black_frame.shape
                x, y = int(point.x * w), int(point.y * h)
                cv2.circle(black_frame, (x, y), 5, color, -1)
                cv2.putText(black_frame, name, (x + 10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame
        out.write(black_frame)
        frame_count += 1

    print(f"Processed {frame_count} frames")
    print(f"Created: SquatLabeledSide.mp4")
    
    # Release everything
    cap.release()
    out.release()
    pose.close()
    print("Done")

if __name__ == "__main__":
    process_video() 