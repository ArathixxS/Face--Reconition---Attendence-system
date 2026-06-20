from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import csv
import os
import threading
import cv2

app = Flask(__name__)
app.secret_key = "secret123"

CORS(app, supports_credentials=True)

DATASET_DIR = "datasets"
ATTENDANCE_FILE = "attendance.csv"
USERS_FILE = "users.csv"

haar_file = "haarcascade_frontalface_default.xml"
IMG_SIZE = (130, 100)

# Admin credentials
ADMIN_EMAIL = "admin@gmail.com"
ADMIN_PASSWORD = "pass"
ADMIN_CODE = "123456"

os.makedirs(DATASET_DIR, exist_ok=True)

# ----------------------------------
# USER MANAGEMENT
# ----------------------------------

def load_users():
    users = {}

    if not os.path.exists(USERS_FILE):
        return users

    with open(USERS_FILE, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            users[row["email"].strip().lower()] = {
                "name": row["name"].strip(),
                "password": row["password"]
            }

    return users


def save_user(name, email, password):

    file_exists = os.path.exists(USERS_FILE)

    with open(USERS_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["name", "email", "password"])

        writer.writerow([name, email, password])


# ----------------------------------
# REGISTER
# ----------------------------------

@app.route("/register", methods=["POST"])
def register():

    data = request.json

    name = data["name"].strip()
    email = data["email"].strip().lower()
    password = data["password"]

    users = load_users()

    if email in users:
        return jsonify({"error": "User already exists"}), 400

    save_user(name, email, password)

    return jsonify({"message": "User registered successfully"})


# ----------------------------------
# USER LOGIN
# ----------------------------------

@app.route("/login", methods=["POST"])
def login():

    data = request.json

    email = data["email"].strip().lower()
    password = data["password"]

    users = load_users()

    if email not in users or users[email]["password"] != password:
        return jsonify({"error": "Invalid email or password"}), 401

    session["user"] = users[email]

    return jsonify({
        "name": users[email]["name"]
    })


# ----------------------------------
# USER PROFILE / ATTENDANCE
# ----------------------------------

@app.route("/profile")
def profile():

    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    name = session["user"]["name"].strip().lower()

    records = []

    if os.path.exists(ATTENDANCE_FILE):

        with open(ATTENDANCE_FILE, newline="") as f:

            reader = csv.DictReader(f)

            for row in reader:

                csv_name = row["Name"].strip().lower()

                if csv_name == name:
                    records.append(row)

    return jsonify({
        "attendance": records,
        "name": session["user"]["name"]
    })


# ----------------------------------
# ADMIN LOGIN
# ----------------------------------

@app.route("/admin/login", methods=["POST"])
def admin_login():

    data = request.json

    email = data.get("email")
    password = data.get("password")
    code = data.get("code")

    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD and code == ADMIN_CODE:
        session["admin"] = True
        return jsonify({"message": "Admin login successful"})

    return jsonify({"error": "Invalid admin credentials"}), 401


# ----------------------------------
# ADMIN ATTENDANCE VIEW
# ----------------------------------

@app.route("/admin/attendance")
def admin_attendance():

    if not session.get("admin"):
        return jsonify({"error": "Unauthorized"}), 403

    records = []

    if os.path.exists(ATTENDANCE_FILE):

        with open(ATTENDANCE_FILE, newline="") as f:

            reader = csv.DictReader(f)

            for row in reader:
                records.append(row)

    return jsonify(records)


# ----------------------------------
# FACE CAPTURE
# ----------------------------------

def capture_faces(username):

    path = os.path.join(DATASET_DIR, username)
    os.makedirs(path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(haar_file)
    cam = cv2.VideoCapture(0)

    count = 1

    while True:

        ret, frame = cam.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, IMG_SIZE)

            cv2.imwrite(f"{path}/{count}.png", face)
            count += 1

        cv2.imshow("Face Capture", frame)

        if cv2.waitKey(1) == 27 or count > 30:
            break

    cam.release()
    cv2.destroyAllWindows()


@app.route("/start-camera", methods=["POST"])
def start_camera():

    name = request.json["username"].strip()

    thread = threading.Thread(target=capture_faces, args=(name,))
    thread.start()

    return jsonify({"message": "Camera started"})


# ----------------------------------
# SERVE FRONTEND
# ----------------------------------

@app.route("/")
def index():
    return send_from_directory(os.getcwd(), "index.html")


# ----------------------------------
# RUN SERVER
# ----------------------------------

if __name__ == "__main__":
    app.run(debug=True)