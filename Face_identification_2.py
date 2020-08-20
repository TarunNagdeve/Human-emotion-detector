
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
model = model_from_json(open("face.json", "r").read())
model.load_weights('face.h5')
facecascade=cv2.CascadeClassifier(r'C:\Users\TARUN\.PyCharmCE2019.1\config\scratches\haarcascades\haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
while True:
    ret,img=cap.read()
    if not ret:
        continue

    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    facce_detected=facecascade.detectMultiScale(gray_img,1.15,5)
    for (x,y,w,h) in facce_detected:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
        cropped_img=gray_img[y:y+w,x:x+h]
        cropped_img=cv2.resize(cropped_img,(48,48))
        img_pixels=image.img_to_array(cropped_img)
        img_pixels=np.expand_dims(img_pixels,axis=0)
        img_pixels/=255

        pred=model.predict(img_pixels)
        max_index = np.argmax(pred[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img=cv2.resize(img,(1000,800))
    cv2.imshow('emotion_analysis',resized_img)

    if(cv2.waitKey(10)==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
