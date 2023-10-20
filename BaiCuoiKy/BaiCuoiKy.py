import os
import cv2 as cv


def detect_faces(image, face_detector, recognizer):
    # Chuyển đổi ảnh sang dạng xám
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong ảnh xám.
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=3)

    # Tải bản ánh xạ của ID sang tên từ một tệp.
    names = {}
    with open('D:\\HocMay_CK\\BaiCuoiKy\\trains\\names.txt', 'r') as f:
        for line in f:
            name, id = line.strip().split(':')
            names[int(id)] = name

    # Vẽ hình chữ nhật xung quanh khuôn mặt đã được phát hiện và đánh dấu tên dự đoán cho chúng.
    for x, y, w, h in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_id, confidence = recognizer.predict(face_roi)
        if confidence < 80:
            cv.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv.putText(image, f"{names[face_id]}", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)



def capture_and_detect(face_detector, recognizer):
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        detect_faces(image, face_detector, recognizer)

        cv.imshow('Face Detection', image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng đối tượng video capture.
    cap.release()



if __name__ == '__main__':
    # Tải mô hình nhận diện và nhận dạng khuôn mặt.
    face_detector = cv.CascadeClassifier(os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_default.xml'))
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read('D:\\HocMay_CK\\BaiCuoiKy\\trains\\trains.yml')

    # Bắt đầu webcam và phát hiện khuôn mặt trong từng khung hình.
    capture_and_detect(face_detector, recognizer)

    # Đóng tất cả các cửa sổ OpenCV.
    cv.destroyAllWindows()
