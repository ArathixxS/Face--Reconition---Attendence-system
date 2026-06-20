import cv2
import os
import numpy as np
import csv
from datetime import datetime

# -----------------------------
# Paths
# -----------------------------
DATASET_DIR = "datasets"
ARCFACE_MODEL = "arcface.onnx"
HAAR_MODEL = "haarcascade_frontalface_default.xml"
ATTENDANCE_FILE = "attendance.csv"

# -----------------------------
# Parameters
# -----------------------------
THRESHOLD = 0.45
MIN_CONFIRMATIONS = 5   # number of consistent recognitions required

# -----------------------------
# Load Haar cascade
# -----------------------------
face_cascade = cv2.CascadeClassifier(HAAR_MODEL)

# -----------------------------
# Load ArcFace recognizer
# -----------------------------
recognizer = cv2.FaceRecognizerSF.create(ARCFACE_MODEL, "")

# -----------------------------
# Recognition counters
# -----------------------------
recognition_counts = {}  # {name: count}

# -----------------------------
# Get current period
# -----------------------------
def get_current_period():
    now = datetime.now()
    minutes = now.hour * 60 + now.minute

    periods = {
        1: (9*60 + 30, 10*60 + 30),
        2: (10*60 + 30, 11*60 + 30),
        3: (11*60 + 30, 12*60 + 30),
        4: (13*60 + 30, 14*60 + 30),
        5: (14*60 + 30, 15*60 + 30),
        6: (15*60 + 30, 16*60 + 30),
        7: (14*60 + 45, 15*60 + 54),
    }

    for period, (start, end) in periods.items():
        if start <= minutes < end:
            if minutes <= start + 10:
                return period
            else:
                return None
    return None

# -----------------------------
# Attendance marking
# -----------------------------
def mark_attendance(name):
    period = get_current_period()
    if period is None:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Period", "Time"])

    with open(ATTENDANCE_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] == name and row[1] == today and int(row[2]) == period:
                return  # already marked

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, period, time_now])

    print(f"[INFO] Attendance marked: {name} | Period {period}")

# -----------------------------
# Extract face embedding
# -----------------------------
def get_face_embedding(img, face_box):
    x, y, w, h = face_box
    face = img[y:y+h, x:x+w]

    if face.size == 0:
        return None

    face = cv2.resize(face, (112, 112))
    return recognizer.feature(face)

# -----------------------------
# Build embedding database
# -----------------------------
print("Building ArcFace embedding database...")
face_db = {}

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_path):
        img = cv2.imread(os.path.join(person_path, img_name))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
        )

        if len(faces) == 0:
            continue

        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        emb = get_face_embedding(img, faces[0])
        if emb is not None:
            embeddings.append(emb)

    if embeddings:
        face_db[person] = np.mean(embeddings, axis=0)
        print(f"{person}: {len(embeddings)} images used")

print("Database ready:", list(face_db.keys()))

# -----------------------------
# Real-time recognition
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
    )

    detected_names = set()

    for (x, y, w, h) in faces:
        emb = get_face_embedding(frame, (x, y, w, h))
        if emb is None:
            continue

        name = "Unknown"
        best_score = 0.0

        for person, db_emb in face_db.items():
            score = recognizer.match(
                emb, db_emb, cv2.FaceRecognizerSF_FR_COSINE
            )
            if score > best_score and score > THRESHOLD:
                best_score = score
                name = person

        if name != "Unknown":
            detected_names.add(name)
            recognition_counts[name] = recognition_counts.get(name, 0) + 1

            if recognition_counts[name] >= MIN_CONFIRMATIONS:
                mark_attendance(name)
                recognition_counts[name] = 0

        display_text = name

        if name != "Unknown":
            count = recognition_counts.get(name, 0)
        
            # Convert similarity score to percentage
            accuracy = int(best_score * 100)
        
            display_text = f"{name} | {accuracy}% | ({count}/{MIN_CONFIRMATIONS})"
        else:
            # Show low confidence for unknown faces too
            accuracy = int(best_score * 100)
            display_text = f"Unknown | {accuracy}%"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, display_text, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

    # Clear stale counters
    for key in list(recognition_counts.keys()):
        if key not in detected_names:
            recognition_counts[key] = 0

    cv2.imshow("Face Recognition Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
