import os
import cv2
import numpy as np

def getImageAndLabels(path_list):
    face_images = []
    ids = []
    names = {}  # Từ điển ánh xạ ID thành tên.
    face_detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
    for image_file in os.listdir(path_list):
        if not image_file.endswith('.png'):
            continue
        image_path = os.path.join(path_list, image_file)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray)
        name = os.path.splitext(image_file)[0]  # Dò tên từ tên tệp
        for x, y, w, h in faces:
            face_images.append(gray[y:y+h, x:x+w])
            if name not in names:
                ids.append(len(names))  # Sử dụng độ dài của danh sách tên như là ID.
                names[name] = len(names)  # Ánh xạ tên với ID trong từ điển
            else:
                ids.append(names[name])  # Sử dụng ID hiện có cho tên
    return face_images, ids, names

faces, ids, names = getImageAndLabels('D:\\HocMay_CK\\BaiCuoiKy\\data')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(ids))
recognizer.write('D:\\HocMay_CK\\BaiCuoiKy\\trains\\trains.yml')

# Lưu bảng ánh xạ từ ID sang tên vào tập tin
with open('D:\\HocMay_CK\\BaiCuoiKy\\trains\\names.txt', 'w') as f:
    for name, id in names.items():
        f.write(f'{name}:{id}\n')