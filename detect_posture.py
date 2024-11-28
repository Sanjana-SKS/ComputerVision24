import cv2
import mediapipe as mp
import math as m

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Video Capture Test
# Replace 0 with string for video file ex: video.mp4
capture = cv2.VideoCapture(0)

# Distance Formula
def distanceFormula(x1, y1, x2, y2):
    return m.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Calculate Confidence using landmark positions
def calculate_confidence(landmarks):
    lmPose = mp_pose.PoseLandmark

    # Get positions
    # Left Shoulder
    left_shoulder_x = landmarks.landmark[lmPose.LEFT_SHOULDER].x
    left_shoulder_y = landmarks.landmark[lmPose.LEFT_SHOULDER].y
    # Right Shoulder
    right_shoulder_x = landmarks.landmark[lmPose.RIGHT_SHOULDER].x
    right_shoulder_y = landmarks.landmark[lmPose.RIGHT_SHOULDER].y
    # Left Hip
    left_hip_x = landmarks.landmark[lmPose.LEFT_HIP].x
    left_hip_y = landmarks.landmark[lmPose.LEFT_HIP].y
    # Right Hip
    right_hip_x = landmarks.landmark[lmPose.RIGHT_HIP].x
    right_hip_y = landmarks.landmark[lmPose.RIGHT_HIP].y

    # Calculate the distance for the shoulders and hips
    shoulder_distance = distanceFormula(left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y)
    hip_distance = distanceFormula(left_hip_x, left_hip_y, right_hip_x, right_hip_y)

    # Check confidence
    if shoulder_distance < hip_distance:
        # Poor posture/slouching
        confidence = 0
    else:
        # Confident
        confidence = 1

    return confidence

# Process Video Frames
while capture.isOpened():
    success, frame = capture.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Recolor to RGB because MediaPipe works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the pose landmarks
    results = pose.process(rgb_frame)

    # Draw landmarks on the original frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Calculate confidence
        confidence = calculate_confidence(results.pose_landmarks)
    else:
        # No posture was detected
        confidence = -1

    # Display confidence on frame
    cv2.putText(frame, f'Posture Confidence: {confidence}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Display the output frame
    cv2.imshow("Posture Detection", frame)

    # Exit if users presses 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the capture object andn close all windows
capture.release()
cv2.destroyAllWindows()
