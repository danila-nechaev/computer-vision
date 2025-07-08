import cv2
import os
from datetime import datetime

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            
            # Рамка
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, faceBoxes

# Пути к модели
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

faceNet = cv2.dnn.readNet(faceModel, faceProto)

video = cv2.VideoCapture(0)

# Сохраняем на рабочий стол
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# Флаг: было ли уже сохранено лицо
face_saved = False

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)

    if not faceBoxes:
        print("Лица не распознаны")
    else:
        if not face_saved:
            # Берем первое лицо
            x1, y1, x2, y2 = faceBoxes[0]
            face = frame[y1:y2, x1:x2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_filename = os.path.join(desktop_path, f"first_face_{timestamp}.jpg")
            cv2.imwrite(face_filename, face)
            print(f"Сохранено первое лицо: {face_filename}")
            face_saved = True  # Больше не сохраняем

    cv2.imshow("Camera", resultImg)
