import cv2
import mediapipe as mp
import numpy as np
import os

def calculate_angle(a, b, c):
    """Calculate angle between three points.
    
    Args:
        a: First point coordinates [x, y]
        b: Middle point coordinates [x, y] (vertex)
        c: Last point coordinates [x, y]
    
    Returns:
        float: Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def get_landmarks(frame, pose):
    """Extract pose landmarks from a frame.
    
    Args:
        frame: Input video frame
        pose: MediaPipe pose detector
    
    Returns:
        pose_landmarks or None if no pose detected
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    # Add debug information
    if results.pose_landmarks is None:
        return None
    return results.pose_landmarks

def extract_angles(landmarks, frame_shape, mp_pose):
    """Calculate angles for all key joints.
    
    Args:
        landmarks: MediaPipe pose landmarks
        frame_shape: Shape of the video frame
        mp_pose: MediaPipe pose solution
    
    Returns:
        dict: Dictionary containing angles for all joints
    """
    if not landmarks:
        return None
        
    h, w = frame_shape[:2]
    
    def get_point(landmark_idx):
        landmark = landmarks.landmark[landmark_idx]
        return [int(landmark.x * w), int(landmark.y * h)]
    
    # Get key points
    left_shoulder = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    left_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP.value)
    right_hip = get_point(mp_pose.PoseLandmark.RIGHT_HIP.value)
    left_knee = get_point(mp_pose.PoseLandmark.LEFT_KNEE.value)
    right_knee = get_point(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    left_ankle = get_point(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    right_ankle = get_point(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    
    angles = {
        'shoulders': {
            'left': calculate_angle(left_hip, left_shoulder, [left_shoulder[0] + 100, left_shoulder[1]]),
            'right': calculate_angle(right_hip, right_shoulder, [right_shoulder[0] + 100, right_shoulder[1]])
        },
        'hips': {
            'left': calculate_angle(left_shoulder, left_hip, left_knee),
            'right': calculate_angle(right_shoulder, right_hip, right_knee)
        },
        'knees': {
            'left': calculate_angle(left_hip, left_knee, left_ankle),
            'right': calculate_angle(right_hip, right_knee, right_ankle)
        },
        'ankles': {
            'left': calculate_angle(left_knee, left_ankle, [left_ankle[0], left_ankle[1] + 100]),
            'right': calculate_angle(right_knee, right_ankle, [right_ankle[0], right_ankle[1] + 100])
        }
    }
    return angles

def analyze_angles(stored_data, mp_pose):
    """Analyze angle ranges and calculate scores."""
    # Initialize dictionaries to store angle ranges
    model_angles = {
        'shoulders': {'left': {'min': float('inf'), 'max': float('-inf')},
                     'right': {'min': float('inf'), 'max': float('-inf')}},
        'hips': {'left': {'min': float('inf'), 'max': float('-inf')},
                 'right': {'min': float('inf'), 'max': float('-inf')}},
        'knees': {'left': {'min': float('inf'), 'max': float('-inf')},
                 'right': {'min': float('inf'), 'max': float('-inf')}},
        'ankles': {'left': {'min': float('inf'), 'max': float('-inf')},
                  'right': {'min': float('inf'), 'max': float('-inf')}}
    }
    
    user_angles = {
        'shoulders': {'left': {'min': float('inf'), 'max': float('-inf')},
                     'right': {'min': float('inf'), 'max': float('-inf')}},
        'hips': {'left': {'min': float('inf'), 'max': float('-inf')},
                 'right': {'min': float('inf'), 'max': float('-inf')}},
        'knees': {'left': {'min': float('inf'), 'max': float('-inf')},
                 'right': {'min': float('inf'), 'max': float('-inf')}},
        'ankles': {'left': {'min': float('inf'), 'max': float('-inf')},
                  'right': {'min': float('inf'), 'max': float('-inf')}}
    }
    
    # Process all frames to find min and max angles
    for i in range(len(stored_data['model_landmarks'])):
        model_frame_angles = extract_angles(stored_data['model_landmarks'][i],
                                         stored_data['model_shapes'][i],
                                         mp_pose)
        user_frame_angles = extract_angles(stored_data['user_landmarks'][i],
                                        stored_data['user_shapes'][i],
                                        mp_pose)
        
        if model_frame_angles and user_frame_angles:
            # Update model angles
            for joint in model_angles:
                for side in ['left', 'right']:
                    angle = model_frame_angles[joint][side]
                    model_angles[joint][side]['min'] = min(model_angles[joint][side]['min'], angle)
                    model_angles[joint][side]['max'] = max(model_angles[joint][side]['max'], angle)
            
            # Update user angles
            for joint in user_angles:
                for side in ['left', 'right']:
                    angle = user_frame_angles[joint][side]
                    user_angles[joint][side]['min'] = min(user_angles[joint][side]['min'], angle)
                    user_angles[joint][side]['max'] = max(user_angles[joint][side]['max'], angle)
    
    # Calculate scores based on angle ranges
    scores = {}
    total_score = 0
    
    print("\n=== Detailed Analysis ===\n")
    
    for joint in model_angles:
        joint_score = 0
        print(f"{joint.title()} Analysis:")
        
        for side in ['left', 'right']:
            # Get ranges
            model_range = model_angles[joint][side]['max'] - model_angles[joint][side]['min']
            user_range = user_angles[joint][side]['max'] - user_angles[joint][side]['min']
            
            # Get midpoints
            model_mid = (model_angles[joint][side]['max'] + model_angles[joint][side]['min']) / 2
            user_mid = (user_angles[joint][side]['max'] + user_angles[joint][side]['min']) / 2
            
            # Calculate differences
            range_diff = abs(model_range - user_range)
            mid_diff = abs(model_mid - user_mid)
            
            # Calculate side score (out of 12.5 - half of 25 points per joint)
            range_score = max(0, 6.25 - (range_diff / 45) * 6.25)
            mid_score = max(0, 6.25 - (mid_diff / 45) * 6.25)
            side_score = range_score + mid_score
            
            joint_score += side_score
            
            print(f"\n{side.title()} Side:")
            print(f"Model Range: {model_angles[joint][side]['min']:.1f}° - {model_angles[joint][side]['max']:.1f}° (Range: {model_range:.1f}°)")
            print(f"User Range: {user_angles[joint][side]['min']:.1f}° - {user_angles[joint][side]['max']:.1f}° (Range: {user_range:.1f}°)")
            print(f"Range Difference: {range_diff:.1f}°")
            print(f"Midpoint Difference: {mid_diff:.1f}°")
            print(f"Side Score: {side_score:.1f}/12.5")
        
        scores[joint] = joint_score
        print(f"\nTotal {joint} score: {joint_score:.1f}/25")
        print("-" * 50)
        total_score += joint_score
    
    print("\n=== Final Score ===")
    print(f"Total Score: {total_score:.1f}/100")
    
    return scores, total_score

def compare_videos():
    print("\nChecking video files...")
    
    # Initialize MediaPipe Pose with lower confidence thresholds
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=1
    )
    
    # Get absolute paths to videos
    current_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current working directory: {current_dir}")
    print(f"Script directory: {base_dir}")
    
    # Define video paths (using known correct paths)
    video_paths = {
        'model_original': '/Users/joshkim/Desktop/squat-technique/Model/SquatMasterSide.mov',  # Using the full path you provided
        'user_original': os.path.join(current_dir, 'User Input', 'UserInput.mov'),
        'model_labeled': os.path.join(current_dir, 'SquatLabeledSide.mp4'),
        'user_labeled': os.path.join(current_dir, 'UserInputLabeled.mp4')
    }
    
    # Check each file
    print("\nChecking for required videos:")
    missing_files = []
    for name, path in video_paths.items():
        if os.path.exists(path):
            print(f"✓ Found {name}: {path}")
        else:
            print(f"✗ Missing {name}: {path}")
            missing_files.append(name)
    
    if missing_files:
        print("\nError: The following files are missing:")
        for file in missing_files:
            print(f"- {file}")
        return
        
    try:
        # Open all videos
        caps = {
            'model_original': cv2.VideoCapture(video_paths['model_original']),
            'user_original': cv2.VideoCapture(video_paths['user_original']),
            'model_labeled': cv2.VideoCapture(video_paths['model_labeled']),
            'user_labeled': cv2.VideoCapture(video_paths['user_labeled'])
        }
        
        # Verify all videos opened successfully
        for name, cap in caps.items():
            if not cap.isOpened():
                print(f"❌ Error: Could not open {name} video")
                return
        
        # Get video properties
        model_frames = int(caps['model_original'].get(cv2.CAP_PROP_FRAME_COUNT))
        user_frames = int(caps['user_original'].get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Information:")
        print(f"Model video frames: {model_frames}")
        print(f"User video frames: {user_frames}")
        
        stored_data = {
            'model_landmarks': [],
            'user_landmarks': [],
            'model_shapes': [],
            'user_shapes': []
        }
        
        print("\nStarting video comparison...")
        print("Press 'q' to quit video playback")
        
        cv2.namedWindow('Comparison', cv2.WINDOW_NORMAL)
        
        frame_number = 0
        landmarks_detected = 0
        model_detections = 0
        user_detections = 0
        
        while True:
            # Read frames from all videos
            frames = {}
            all_frames_read = True
            
            for name, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    all_frames_read = False
                    break
                frames[name] = frame
            
            if not all_frames_read:
                break
            
            frame_number += 1
            
            # Process original frames for pose detection
            landmarks_model = get_landmarks(frames['model_original'], pose)
            landmarks_user = get_landmarks(frames['user_original'], pose)
            
            # Track individual detections
            if landmarks_model:
                model_detections += 1
            if landmarks_user:
                user_detections += 1
            
            if landmarks_model and landmarks_user:
                landmarks_detected += 1
                stored_data['model_landmarks'].append(landmarks_model)
                stored_data['user_landmarks'].append(landmarks_user)
                stored_data['model_shapes'].append(frames['model_original'].shape)
                stored_data['user_shapes'].append(frames['user_original'].shape)
            
            # Display labeled frames with detection status
            height = 480
            model_display = cv2.resize(frames['model_labeled'], 
                                     (int(height * frames['model_labeled'].shape[1] / frames['model_labeled'].shape[0]), height))
            user_display = cv2.resize(frames['user_labeled'], 
                                    (int(height * frames['user_labeled'].shape[1] / frames['user_labeled'].shape[0]), height))
            
            # Add detection status
            cv2.putText(model_display, 
                       f"Pose {'Detected' if landmarks_model else 'Not Detected'}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0) if landmarks_model else (0, 0, 255), 2)
            
            cv2.putText(user_display, 
                       f"Pose {'Detected' if landmarks_user else 'Not Detected'}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0) if landmarks_user else (0, 0, 255), 2)
            
            combined_frame = np.hstack((model_display, user_display))
            
            # Add frame counter and detection stats
            cv2.putText(combined_frame, 
                       f'Frame: {frame_number} | Model: {model_detections} | User: {user_detections} | Sync: {landmarks_detected}', 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            cv2.imshow('Comparison', combined_frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                print("\nVideo playback stopped by user")
                break
        
        print(f"\nAnalysis Summary:")
        print(f"Total frames processed: {frame_number}")
        print(f"Model pose detections: {model_detections}")
        print(f"User pose detections: {user_detections}")
        print(f"Synchronized pose detections: {landmarks_detected}")
        
        if landmarks_detected == 0:
            print("\n❌ Error: No synchronized poses were detected.")
            print("\nPossible issues:")
            if model_detections == 0:
                print("- No poses detected in model video")
            if user_detections == 0:
                print("- No poses detected in user video")
            print("\nTry:")
            print("1. Ensure person is clearly visible in both videos")
            print("2. Check video quality and lighting")
            print("3. Verify that videos show full body poses")
            return
        
        # Continue with angle analysis if poses were detected
        if landmarks_detected > 0:
            print("\nCalculating scores...")
            scores, total_score = analyze_angles(stored_data, mp_pose)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            for cap in caps.values():
                cap.release()
            cv2.destroyAllWindows()
            pose.close()
        except:
            pass
        print("Done")

if __name__ == "__main__":
    compare_videos() 