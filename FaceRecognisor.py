import cv2
import face_recognition
import numpy as np
import os

# Load dataset
path = "FaceImages"
images = []
names = []
for file in os.listdir(path):
    img = cv2.imread(f"{path}/{file}")
    images.append(img)
    names.append(os.path.splitext(file)[0])  # remove .jpg

# Encode faces
def encode_faces(imgs):
    encodings = []
    for img in imgs:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img_rgb)[0]
        encodings.append(enc)
    return encodings

known_encodings = encode_faces(images)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img_small = cv2.resize(frame, (0, 0), None, 0.25, 0.25)  # speed up
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(img_small)
    encodings_cur = face_recognition.face_encodings(img_small, faces)

    for encodeFace, faceLoc in zip(encodings_cur, faces):
        matches = face_recognition.compare_faces(known_encodings, encodeFace)
        dist = face_recognition.face_distance(known_encodings, encodeFace)
        matchIndex = np.argmin(dist)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1+6, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()