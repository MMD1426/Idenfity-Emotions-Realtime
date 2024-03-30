import cv2
import keras
from keras.preprocessing.image import img_to_array
import numpy as np


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


model = keras.models.load_model('Identify-Emotions\model-idenfity-emotion.h5')

face_cascade = cv2.CascadeClassifier('Identify-Emotions\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

list_emotion = ['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']

while True:
    ret , frame = cap.read()

    frame = cv2.resize(frame,(800,600))

    num_face = face_cascade.detectMultiScale(image=frame)

    for x,y,w,h in num_face:
        draw_border(frame,(x,y),(x+w,y+h),(0,255,0),2,10,15)
        face_detecting = frame[y:y+h,x:x+w]
        face_size = cv2.resize(face_detecting,(48,48))

        if np.sum([face_size]) != 0:
            face_preprocessing = face_size.astype('float') / 255
            array_face_detecting = img_to_array(face_size)
            face_model = np.expand_dims(array_face_detecting,axis=0)

            prediction = model.predict(face_model)[0]
            lable = list_emotion[prediction.argmax()]

            cv2.putText(frame,lable,(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255))

        else:
            cv2.putText(frame,'Not Found Face',(50,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255))


    cv2.imshow('Idenfity Emotions',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()