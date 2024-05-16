from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import os
import csv
import time
from datetime import datetime

facedetect = cv2.CascadeClassifier("Data/haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

with open('Data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('Data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(FACES, LABELS)
col_names = ['NAME', 'TIME']

imgBgrnd = cv2.imread("background.jpg")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame, 1)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        OP = KNN.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x - 3, y - 45), (x + w + 3, y), (161, 97, 37), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (161, 97, 37), 5)
        if OP not in LABELS:
            cv2.putText(frame, "Unknown Face", (x + 10, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
        else:
            cv2.putText(frame, str(OP[0]), (x + 10, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
        attendance = [str(OP[0]), str(timestamp)]
    frame_height, frame_width, channels = frame.shape
    resized_bg = cv2.resize(imgBgrnd, (frame_width, frame_height))
    desired_height, desired_width = 1700, 3160
    resized_frame = cv2.resize(frame, (desired_width, desired_height))
    imgBgrnd[1000:1000 + desired_height, 1430:1430 + desired_width] = resized_frame
    cv2.imshow('Face Recognition Software', imgBgrnd)

    if cv2.waitKey(1) == ord('o'):
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(col_names)
                writer.writerow(attendance)
            csvfile.close()
    if cv2.waitKey(1) == ord("c"):
        break
video.release()
cv2.destroyAllWindows()
