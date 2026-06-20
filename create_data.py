from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import os
import threading

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML file (different origin)

@app.route('/')
def serve_index():
    """Serve the main index.html file."""
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'index.html')

haar_file  = 'haarcascade_frontalface_default.xml'
datasets   = 'datasets'
IMG_SIZE   = (130, 100)


def capture_faces(username: str):
    """Run in a background thread so Flask can respond immediately."""
    path = os.path.join(datasets, username)
    os.makedirs(path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)

    count = 1
    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face        = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, IMG_SIZE)
            cv2.imwrite(f"{path}/{count}.png", face_resize)
            print(f"Saved image {count}")
            count += 1

        cv2.imshow(f"Capturing – {username}  (press ESC to cancel)", frame)

        key = cv2.waitKey(100) & 0xFF
        if key == 27 or count > 30:
            break

    webcam.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {count - 1} images for '{username}'.")


@app.route('/start-camera', methods=['POST'])
def start_camera():
    data = request.get_json(force=True) or {}
    username = data.get('username', '').strip()

    if not username:
        return jsonify({'error': 'Username is required'}), 400

    # Sanitise: remove characters that are unsafe in folder names
    safe_name = "".join(c for c in username if c.isalnum() or c in (' ', '_', '-')).strip()
    if not safe_name:
        return jsonify({'error': 'Invalid username'}), 400

    # Run camera capture in a separate thread so we can return JSON immediately
    thread = threading.Thread(target=capture_faces, args=(safe_name,), daemon=True)
    thread.start()

    return jsonify({'message': f"Camera started for '{safe_name}'. Capturing 30 images…"}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000) 