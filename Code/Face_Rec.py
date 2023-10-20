import os
import cv2
import h5py
import keras
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model

from Name import *


def recognize_faces():
    # Load pre-trained model
    model = load_model('./logs/modal1.h5')
    img_rows = 300 
    img_cols = 300

    # OpenCV capture from default camera
    cap = cv2.VideoCapture(0)

    # Create Tkinter window
    MainFrame = Tk()
    MainFrame.title("Real-time Face Recognition")
    MainFrame.geometry('800x600')
    label_img = ttk.Label(MainFrame)
    label_img.pack()

    # Define the face detection model
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #Note the change

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Preprocess input image
        img = cv2.resize(frame, (img_rows, img_cols))
        img = img_to_array(img)
        img = preprocess_input(np.array(img).reshape(-1, img_cols, img_rows, 3))

        # Predict class probabilities
        predictions = model.predict(img)
        pre_name = np.argmax(predictions, axis=1)

        # Get face label and accuracy
        for i in pre_name:
            name = Name.get(i)
            acc = np.max(predictions) * 100
            print("Predicted face: %s, Accuracy: %.2f%%" % (name, acc))

            # Draw label and accuracy on image
            cv2.putText(frame, "{} ({:.2f}%)".format(name, acc),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        #cv2.imshow('Real-time Face Recognition', frame)

        # Update the Tkinter window
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        label_img.configure(image=img)
        label_img.image = img
        MainFrame.update()

        # Stop video capture by pressing 'q' key
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # When everything done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    MainFrame.destroy()


if __name__ == '__main__':
    recognize_faces()