#Version: 1.0
#Python 3.11.0
#Authors: Matthew, Sanjana, Judy
#Date: 12/2/2024
#Description: Main script for real time presentation feedback system
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #For the annoying TensorFlow warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import cv2
import mediapipe as mp
import time
import numpy as np
#from volume import VolumeReader
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark

#Init MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#Init VolumeReader
#volume_reader = VolumeReader(port='COM6', baud_rate=9600)

#Data Collection
posture_data = []
volume_data = []
analysis_interval = 5  # seconds
last_analysis_time = time.time()

#Capture that video
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('video.mp4')

#Might calculate the angle 
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.degrees(radians)
    angle = angle if angle >= 0 else angle + 360
    return angle

#Might calculate the distance
def calculate_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    return np.linalg.norm(a - b)

#loop for stuff
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            #RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            #Make the magic happen
            results = pose.process(image)

            #Is the image writeable?
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #Get the landmarks
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                #Draw the body landmarks
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                #Get those landmarks
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                #Midpoint between shoulders and hips
                mid_shoulder = NormalizedLandmark(
                    x=(left_shoulder.x + right_shoulder.x) / 2,
                    y=(left_shoulder.y + right_shoulder.y) / 2
                )
                #Midpoint between hips
                mid_hip = NormalizedLandmark(
                    x=(left_hip.x + right_hip.x) / 2,
                    y=(left_hip.y + right_hip.y) / 2
                )

                #Vertical point above the mid-shoulder
                vertical_point = NormalizedLandmark(
                    x=mid_shoulder.x,
                    y=mid_shoulder.y - 0.1  # Adjust as needed
                )

                #How level is the spine
                spine_angle = calculate_angle(mid_hip, mid_shoulder, vertical_point)

                #How level are the shoulders
                shoulder_level_diff = abs(left_shoulder.y - right_shoulder.y)

                #Are we moving our arms enough
                left_wrist_hip_dist = calculate_distance(left_wrist, left_hip)
                right_wrist_hip_dist = calculate_distance(right_wrist, right_hip)

                #Collect data
                current_time = time.time()
                posture_data.append({
                    'time': current_time,
                    'spine_angle': spine_angle,
                    'shoulder_level_diff': shoulder_level_diff,
                    'left_wrist_hip_dist': left_wrist_hip_dist,
                    'right_wrist_hip_dist': right_wrist_hip_dist
                })
                #Volume data
                '''
                volume = volume_reader.get_volume()
                volume_data.append({
                    'time': current_time,
                    'volume': volume
                }) '''
                
                #Change stuff
                if current_time - last_analysis_time >= analysis_interval:
                    #Posture Analysis
                    recent_posture_data = [d for d in posture_data if d['time'] >= current_time - analysis_interval]
                    avg_spine_angle = np.mean([d['spine_angle'] for d in recent_posture_data])
                    avg_shoulder_level_diff = np.mean([d['shoulder_level_diff'] for d in recent_posture_data])

                    #Arm variance
                    left_arm_variance = np.var([d['left_wrist_hip_dist'] for d in recent_posture_data])
                    right_arm_variance = np.var([d['right_wrist_hip_dist'] for d in recent_posture_data])

                    #Volume Analysis
                    #recent_volume_data = [d for d in volume_data if d['time'] >= current_time - analysis_interval]
                    #avg_volume = np.mean([d['volume'] for d in recent_volume_data])

                    #Array to store metrics for display lol
                    metrics_display = []

                    #Average Spine Angle Feedback
                    if 178 <= avg_spine_angle <= 182:
                        spine_feedback = "Spine alignment is good"
                        spine_status = "good"
                    else:
                        spine_feedback = "You need to stand up straighter"
                        spine_status = "bad"

                    metrics_display.append({
                        'name': 'Avg Spine Angle',
                        'value': f"{avg_spine_angle:.2f}",
                        'feedback': spine_feedback,
                        'status': spine_status
                    })

                    #Feedback for Volume
                    '''
                    if avg_volume >= 55:
                        volume_feedback = "Volume level is good"
                        volume_status = "good"
                    else:
                        volume_feedback = "You need to speak louder"
                        volume_status = "bad"

                    metrics_display.append({
                        'name': 'Avg Volume',
                        'value': f"{avg_volume:.2f} dB",
                        'feedback': volume_feedback,
                        'status': volume_status
                    })
                    '''

                    #Left arm good?
                    if left_arm_variance >= 0.0010:
                        left_arm_feedback = "Left arm movement is good"
                        left_arm_status = "good"
                    else:
                        left_arm_feedback = "Move your left arm more"
                        left_arm_status = "bad"

                    metrics_display.append({
                        'name': 'Left Arm Variance',
                        'value': f"{left_arm_variance:.4f}",
                        'feedback': left_arm_feedback,
                        'status': left_arm_status
                    })

                    #Right arm good?
                    if right_arm_variance >= 0.0010:
                        right_arm_feedback = "Right arm movement is good"
                        right_arm_status = "good"
                    else:
                        right_arm_feedback = "Move your right arm more"
                        right_arm_status = "bad"

                    metrics_display.append({
                        'name': 'Right Arm Variance',
                        'value': f"{right_arm_variance:.4f}",
                        'feedback': right_arm_feedback,
                        'status': right_arm_status
                    })

                    #shoulder good?
                    if avg_shoulder_level_diff <= 0.02:  # Adjust threshold as appropriate
                        shoulder_feedback = "Shoulders are level"
                        shoulder_status = "good"
                    else:
                        shoulder_feedback = "Keep your shoulders level"
                        shoulder_status = "bad"

                    metrics_display.append({
                        'name': 'Shoulder Level Diff',
                        'value': f"{avg_shoulder_level_diff:.4f}",
                        'feedback': shoulder_feedback,
                        'status': shoulder_status
                    })

                    #Reset the data
                    last_analysis_time = current_time
                else:
                    #Display the image with the metrics
                    metrics_display = metrics_display if 'metrics_display' in locals() else []

                #Add border around the image
                border_color = (0, 0, 0)  # Black border
                bordered_image = cv2.copyMakeBorder(
                    image,
                    top=2, bottom=2, left=2, right=2,
                    borderType=cv2.BORDER_CONSTANT,
                    value=border_color
                )

                #Display metrics on a dashboard
                dashboard_height = 360  
                dashboard = np.full((dashboard_height, bordered_image.shape[1], 3), 255, dtype=np.uint8)  

                #COLORS
                good_color = (0, 200, 0)  #Green
                bad_color = (0, 0, 255)   #Red 
                text_color = (0, 0, 0)    #Black

                #Start
                y0, dy = 30, 60  

                #Display metrics
                for i, metric in enumerate(metrics_display):
                    y = y0 + i * dy
                    # Use text color based on status
                    color = good_color if metric['status'] == 'good' else bad_color
                    # Draw the metric name and value
                    cv2.putText(dashboard, f"{metric['name']}: {metric['value']}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    # Draw the feedback message
                    cv2.putText(dashboard, metric['feedback'], (10, y + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

                #Display current volume
                '''
                current_volume = volume_reader.get_volume()
                cv2.putText(dashboard, f"Current Volume: {current_volume} dB",
                            (10, dashboard_height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 2)
                '''

                #Concatenate image and dashboard
                combined_image = np.vstack((bordered_image, dashboard))

                #Display the image
                cv2.imshow('Posture and Volume Analysis', combined_image)
            else:
                cv2.putText(image, "No pose detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Posture and Volume Analysis', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    #Catch the error
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        #volume_reader.stop()
