import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from tkinter import *
import cv2
from PIL import Image, ImageTk
import threading


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# old 站立, new 蹲下
def lim_squat_2(old_blen, shoulder, hip):
    new_blen = get_body_len(shoulder, hip)
    # print(new_blen)
    # print(old_blen)
    if new_blen < old_blen*0.75:
        return 0 # 駝背
    return 1 # correct


# 計算軀幹與水平面的角度
def lim_squat_3(a, b):
    a = np.array(a)  # shoulder
    b = np.array(b)  # hip
    c = np.array([b[0]+0.5,b[1]]) # horizon
    angle = calculate_angle(a, b, -c)
    # print("軀幹與水平面: "angle)
    if angle < 35 or angle > 145: # 朝地
        return 0
    return 1 #朝前

def get_body_len(shoulder, hip):
    s = np.array(shoulder)
    h = np.array(hip)
    return ((s[0] - h[0])**2 + (s[1] - h[1])**2)**0.5 


# Curl counter variables
def do_squat(lbl_1, panel, camera, root):
    counter = 0
    stage = None
    body_len = 1
    msg = ""
    stage = "up"
    
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
            ret, frame = camera.read()
            frame = cv2.flip(frame,1)

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


                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]


                # Calculate angle
                #angle = calculate_angle(l_shoulder, l_elbow, wrist)
                l_angle = calculate_angle(l_hip, l_knee, l_ankle)
                body_angle = calculate_angle(l_shoulder, l_hip, l_ankle)

                # print(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x)
                # Visualize angle
                '''
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                '''


                # Curl counter logic
                # l:(hip, knee, ankle), body:(shoulder, hip, ankle)
                if l_angle > 160 and body_angle > 160: 
                    body_len = get_body_len(l_shoulder, l_hip)
                    stage = "up"
                if l_angle < 90 and stage == "up":  #蹲下時增加次數
                    stage = "down"
                    counter += 1
                    print(counter)
                    msg = ""
                    if lim_squat_2(body_len, l_shoulder, l_hip) == 0:
                            msg = "背一定要打直。"
                        # # 判斷身體跟水平面的夾角
                    if lim_squat_3(l_shoulder, l_hip) == 0 or lim_squat_3(r_shoulder, r_hip) == 0:
                        msg = msg + "胸朝前方，不要朝地板。"
                    if msg == "":
                        msg = "做得不錯!"

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (0, 0, 255), -1)

            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                        )

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)  # 轉換顏色從BGR到RGBA
            current_image = Image.fromarray(image)  # 將圖像轉換成Image對象
            imgtk = ImageTk.PhotoImage(current_image)
            panel.imgtk = imgtk
            panel.config(image=imgtk)
            lbl_1.config(text = msg)
            root.update()
        # camera.release()
        # cv2.destroyAllWindows()
