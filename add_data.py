import cv2
import pickle
import numpy as np
import os

facedetect = cv2.CascadeClassifier("Data/haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

faces_data = []
i = 0
name = input("Enter your name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame, 1)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 75), cv2.FONT_HERSHEY_DUPLEX, 2, (161, 97, 37), 3)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (161, 97, 37), 5)
    cv2.imshow('Add New Face', frame)
    if cv2.waitKey(10) == ord("c") or len(faces_data) == 100:
        break
video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

if 'names.pkl' not in os.listdir('Data/'):
    names = [name] * 100
    with open('Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('Data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name]
    with open('Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('Data/'):
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('Data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
