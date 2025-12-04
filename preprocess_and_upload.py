import os
import cv2
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ.get("DATABASE_URL")
DATA_DIR = "data"   # your employee subfolders (data/12, data/14 etc)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# DB connection pool
postgres_pool = pool.SimpleConnectionPool(1, 5, DB_URL, sslmode='require')

def get_conn():
    return postgres_pool.getconn()

def release_conn(conn):
    postgres_pool.putconn(conn)


def clean_and_upload():
    conn = get_conn()
    cursor = conn.cursor()

    for folder in os.listdir(DATA_DIR):
        emp_folder = os.path.join(DATA_DIR, folder)

        if not os.path.isdir(emp_folder):
            continue

        try:
            employee_id = int(folder)
        except:
            print(f"❌ Folder '{folder}' is not a valid employee ID — Skipping.")
            continue

        print(f"\n➡ Processing employee: {employee_id}")

        for file in os.listdir(emp_folder):
            img_path = os.path.join(emp_folder, file)

            if not os.path.isfile(img_path):
                continue

            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠ Could not read: {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect face
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                print(f"⚠ No face detected in: {file} — Skipped.")
                continue

            # Use first detected face
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]

            # Resize like admin panel
            face_img = cv2.resize(face_img, (200, 200))

            # Encode to JPEG
            ret, buffer = cv2.imencode(".jpg", face_img)
            img_bytes = buffer.tobytes()

            # Insert into DB
            try:
                cursor.execute(
                    "INSERT INTO employee_images (employee_id, image) VALUES (%s, %s)",
                    (employee_id, psycopg2.Binary(img_bytes))
                )
                print(f"   ✔ Uploaded: {file}")
            except Exception as e:
                print(f"   ❌ Error uploading {file}: {e}")

    conn.commit()
    release_conn(conn)
    print("\n🎉 All faces processed and uploaded successfully!")


if __name__ == "__main__":
    clean_and_upload()
