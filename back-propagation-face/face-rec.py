import tkinter
import os
import numpy as np
import cv2
import random
import pickle
import time
window  = tkinter.Tk()
window.title("Face Recognition")
window.geometry('500x500')


def Add():
    roll = roll_no.get()
    Create(roll)
roll_no = tkinter.StringVar()




def Create(roll):
    path = os.getcwd()
    path = os.path.join(path, "DataSeT")
    if os.path.isdir(path) == False:
        os.mkdir(path)
    roll_no = str(roll)

    filefolder = os.path.join(path, roll_no)
    try:
        os.mkdir(filefolder, 0o755)
    except OSError:
        print("Unique id has  already been taken")
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    for i in range(0, 25):
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi_gray, (200, 200))
            cv2.imwrite(os.path.join(filefolder, 'image_%i.jpg' % i), roi)
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
    if len(os.listdir(filefolder)) == 0:
        os.rmdir(filefolder)
    cv2.destroyAllWindows()

def Update():
    path = os.getcwd()
    category = []
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer1 = cv2.face.FisherFaceRecognizer_create()
    Data_Dir = os.path.join(path, 'DataSet')
    training = []
    for file in os.listdir(Data_Dir):
        img_dir = os.path.join(Data_Dir, file)
        category.append(file)
        for images in os.listdir(img_dir):
            img = cv2.imread(os.path.join(img_dir, images), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (150, 150))
            training.append([img, category.index(file)])
    random.shuffle(training)
    x = []
    y = []
    for features, labels in training:
        x.append(features)
        y.append(labels)
    x = np.array(x).reshape(-1, 150, 150, 1)
    pickle_out = open("training_img.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()
    pickle_out = open("training_roll_no.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
    recognizer.train(x, np.array(y))
    recognizer.save("trainer.yml")
    recognizer1.train(x, np.array(y))
    recognizer1.save("trainer1.yml")

def Recognise():
    path = os.getcwd()
    path = os.path.join(path, "DataSeT")
    Categories = []
    for files in os.listdir(path):
        Categories.append(files)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer1 = cv2.face.FisherFaceRecognizer_create()
    pickle_in = open("training_img.pickle", "rb")
    x = pickle.load(pickle_in)
    pickle_in = open("training_roll_no.pickle", "rb")
    y = pickle.load(pickle_in)
    recognizer.read("trainer.yml")
    recognizer1.read("trainer1.yml")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while (cv2.waitKey(1) & 0xFF == ord('q')) == False:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (150, 150))
            y1, conf = recognizer.predict(roi_gray)
            y2, conf1 = recognizer1.predict(roi_gray)
            if y1 == y2:
                print(Categories[y1])
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame, Categories[y1], (x,y), font, 1, (0,0,255), 1, lineType=cv2.LINE_AA)
        cv2.imshow('Image',frame)
    cv2.destroyAllWindows()

label = tkinter.Label(window,text = "NEW_ID:").grid(row = 0,column = 0)
entry = tkinter.Entry(window, textvariable=roll_no).grid(row=0, column=1)
button = tkinter.Button(window,text = "Add_New ID",command= Add).grid(row = 1 , column = 1)
button1 = tkinter.Button(window,text = "UpDate_DataSet",command = Update).grid(row = 2 , column = 0)
button2 = tkinter.Button(window,text = "Run_Recogniser",command = Recognise).grid(row = 3 , column = 0)


window.mainloop()
