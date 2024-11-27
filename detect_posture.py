import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Video Capture Test
# Replace 0 with string for video file ex: video.mp4
capture = cv2.VideoCapture(0)

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

    # Display the output frame
    cv2.imshow("Posture Detection", frame)

    # Exit if users presses 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the capture object andn close all windows
capture.release()
cv2.destroyAllWindows()
