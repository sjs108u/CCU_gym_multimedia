from tkinter import *
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

import squat
import dumbbell


camera = cv2.VideoCapture(0)  # 攝像頭
x = "initial"
global button_click
button_click = False

def video_loop():
    success, img = camera.read()  # 從攝像頭讀取照片
    if success and button_click == False:
        img = cv2.flip(img,1)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 轉換顏色從BGR到RGBA
        current_image = Image.fromarray(cv2image)  # 將圖像轉換成Image對象
        imgtk = ImageTk.PhotoImage(current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
    elif success and button_click == "squat":
        squat.do_squat(lbl_1, panel, camera, root)
    elif success and button_click == "dumbbell":
        dumbbell.do_dumbbell(lbl_1, panel, camera, root)
    root.after(1, video_loop)



def prepare_squat():
    global button_click
    button_click = "squat"


def prepare_dumbbell():
    global button_click
    button_click = "dumbbell"

global root
root = Tk()
root.title("opencv + tkinter")
#root.protocol('WM_DELETE_WINDOW', detector)

panel = Label(root)  # initialize image panel
panel.pack(padx=10, pady=10)
root.config(cursor="arrow")
btn = Button(root, text="舉啞鈴(dumbbell)", command=prepare_dumbbell)
btn.pack(fill="both", expand=True, padx=10, pady=10)
btn2 = Button(root, text="深蹲(squat)", command=prepare_squat)
btn2.pack(fill="both", expand=True, padx=10, pady=10)
global lbl_1
lbl_1 = Label(root, text = x)
lbl_1.pack(padx=10, pady=10)


video_loop()

root.mainloop()
# 當一切都完成後，關閉攝像頭並釋放所佔資源
camera.release()
cv2.destroyAllWindows()