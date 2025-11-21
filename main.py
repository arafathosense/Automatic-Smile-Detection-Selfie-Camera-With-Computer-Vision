import cv2
import datetime
import time
import os

# -------------------------
#  Load Haar Cascades safely
# -------------------------
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

if face_cascade.empty() or smile_cascade.empty():
    print("ERROR: Haarcascade XML files not found!")
    exit()

# -------------------------
#  Open Camera
# -------------------------
cap = cv2.VideoCapture(0)   # 0 = built-in webcam | 1 = USB webcam

if not cap.isOpened():
    print("ERROR: Camera not found!")
    exit()

# Cooldown timer (prevent continuous saving)
last_saved_time = 0
save_interval = 2    # seconds between each photo

# -------------------------
#  Main Loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read camera frame!")
        break

    original_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        face_roi = frame[y:y + h, x:x + w]
        gray_roi = gray[y:y + h, x:x + w]

        smiles = smile_cascade.detectMultiScale(gray_roi, 1.3, 25)

        for x1, y1, w1, h1 in smiles:
            cv2.rectangle(face_roi, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

            # --- Save only once per interval ---
            if time.time() - last_saved_time > save_interval:
                time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                file_name = f"selfie-{time_stamp}.png"

                # Save inside a folder
                save_path = os.path.join("saved_photos", file_name)
                os.makedirs("saved_photos", exist_ok=True)

                cv2.imwrite(save_path, original_frame)
                print("📸 Saved:", save_path)

                last_saved_time = time.time()

    cv2.imshow("cam star", frame)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
