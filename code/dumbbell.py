import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from tkinter import *
import cv2
from PIL import Image, ImageTk


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return float(angle)


## Setup mediapipe instance
def do_dumbbell(lbl_1, panel, camera, root):
    
    # Curl counter variables
    counter = 0
    stage = None
    arm_max_angle = -1
    arm_min_angle = 999
    idx = 0
    agle = None
    dif = 0
    notification = None
    message = ''
    msg = ""
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
            ret, frame = camera.read()
            frame =  cv2.flip(frame,1)

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]


                # print(r_shoulder)
                # Calculate angle
                hip_angle = calculate_angle(r_elbow, r_shoulder, r_hip)
                angle = calculate_angle(shoulder, elbow, wrist)
                r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                # print(r_angle)
                
                # Visualize angle
                cv2.putText(image, str(r_angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                
                # save angle
                if idx == 0:
                    agle = float(r_angle)
                idx += 1
                # print(idx)
                
                # determine the stage
                if idx % 10 == 0:
                    dif = r_angle - agle
                    agle = r_angle

                    if abs(dif) < 2:
                        continue
                    """
                    if hip_angle > 30 :
                        print("wrong")
                        message = "Wrong!!"
                        cv2.rectangle(image, (0, 0), (225, 73), (0, 0, 255), -1)

                        cv2.putText(image, str(message),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)


                        continue
                    """
                    
                    # print(dif)
                    # down stage
                    if dif > 0:
                        # if stage == None:
                        #     stage = "down"
                        # elif stage == "up":
                        #     stage = "down"
                        stage = "down"
                    # down stage
                    elif dif < 0:
                        if stage == "down":
                            # stage = "up"
                            counter += 1
                        elif stage == None:
                            # stage = "up"
                            counter += 1
                        stage = "up"
                
                # print(stage)
                if stage == 'up':
                    # reset max angle of this iteration
                    if arm_max_angle != -1:
                        arm_max_angle = -1
                    # find min angle
                    if r_angle <= arm_min_angle:
                        arm_min_angle = r_angle
                    # message
                    if arm_min_angle >= 30:
                        # print(arm_min_angle)
                        message = 'up'
                        # print('keep going up')
                    elif arm_min_angle < 30:
                        message = 'good'
                    
            
                elif stage == 'down':
                    # reset min angle
                    if arm_min_angle != 999:
                        arm_min_angle = 999
                    # find max angle
                    if r_angle >= arm_max_angle:
                        arm_max_angle = r_angle
                    # message
                    if arm_max_angle <= 150:    
                        # print('keep going down')
                        message = 'down'
                    else:
                        message = 'good'
                
                if hip_angle > 15 :
                        print("wrong")
                        message = "Wrong!!"
                

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (0, 0, 255), -1)

            cv2.putText(image, str(message),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image, stage,
            #             (60, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                    )

            # cv2.imshow('Output Feed', image)
            cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)  # 轉換顏色從BGR到RGBA
            current_image = Image.fromarray(image)  # 將圖像轉換成Image對象
            imgtk = ImageTk.PhotoImage(current_image)
            panel.imgtk = imgtk
            panel.config(image=imgtk)
            lbl_1.config(text = message)
            root.update()

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # cap.release()
    # cv2.destroyAllWindows()