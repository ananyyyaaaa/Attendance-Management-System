import os
import datetime
import numpy as np
import base64
import io
import sqlite3
import psycopg2
from psycopg2 import pool
import pytz
import requests
import PIL.Image
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

load_dotenv()
app = Flask(__name__)
CORS(app)

DB_URL = os.environ.get('DATABASE_URL')

# LBPH Face Recognizer Configuration
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
LBPH_THRESHOLD = 60  # Lower = stricter matching

# Face images storage directory
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- Database Pool ----------------
_postgres_pool = None

def init_connection_pool():
    global _postgres_pool
    if DB_URL and not _postgres_pool:
        try:
            if 'localhost' in DB_URL:
                _postgres_pool = pool.SimpleConnectionPool(1, 10, DB_URL)
            else:
                _postgres_pool = pool.SimpleConnectionPool(1, 10, DB_URL, sslmode='require')
            print("✅ Connection pool initialized")
        except Exception as e:
            print(f"⚠️ Connection pool failed: {e}")

def get_db_connection():
    global _postgres_pool
    if DB_URL:
        if _postgres_pool is None:
            init_connection_pool()
        if _postgres_pool:
            try:
                return _postgres_pool.getconn()
            except Exception as e:
                print(f"Postgres Connection Failed: {e}")
                return None
        else:
            try:
                if 'localhost' in DB_URL:
                    return psycopg2.connect(DB_URL)
                else:
                    return psycopg2.connect(DB_URL, sslmode='require')
            except Exception as e:
                print(f"Postgres Connection Failed: {e}")
                return None
    else:
        conn = sqlite3.connect('attendance.db', check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

def return_db_connection(conn):
    if DB_URL and _postgres_pool and conn:
        try:
            _postgres_pool.putconn(conn)
        except Exception:
            pass

# ---------------- DB Initialization ----------------
def init_db():
    if DB_URL:
        try:
            if 'localhost' in DB_URL:
                conn = psycopg2.connect(DB_URL)
            else:
                conn = psycopg2.connect(DB_URL, sslmode='require')
        except Exception as e:
            print(f"Postgres Connection Failed: {e}")
            return
    else:
        conn = sqlite3.connect('attendance.db', check_same_thread=False)
        conn.row_factory = sqlite3.Row

    c = conn.cursor()
    if DB_URL:
        # Only store employee ID, name, and email - face images stored in folders
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id SERIAL PRIMARY KEY, name TEXT, email TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id SERIAL PRIMARY KEY, name TEXT, timestamp TIMESTAMP, type TEXT)''')
        try:
            c.execute('''CREATE INDEX IF NOT EXISTS idx_logs_name_date ON logs(name, DATE(timestamp))''')
        except:
            pass
    else:
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, timestamp DATETIME, type TEXT)''')
        try:
            c.execute('''CREATE INDEX IF NOT EXISTS idx_logs_name_date ON logs(name, DATE(timestamp))''')
        except:
            pass

    if DB_URL:
        c.execute('''CREATE TABLE IF NOT EXISTS clients
                     (id SERIAL PRIMARY KEY, name TEXT, start_date TEXT, end_date TEXT, cost TEXT)''')
    else:
        c.execute('''CREATE TABLE IF NOT EXISTS clients
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, start_date TEXT, end_date TEXT, cost TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ---------------- OpenCV Face Detection & Recognition ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

_lbph_recognizer = None
_employee_ids = {}  # Maps employee ID to name
_id_to_employee = {}  # Maps name to employee ID

def get_lbph_recognizer():
    """Initialize or return existing LBPH Face Recognizer"""
    global _lbph_recognizer
    if _lbph_recognizer is None:
        print("Initializing LBPH recognizer...")
        train_recognizer()  # Train first, which will create the recognizer
        # If training didn't create a recognizer (no faces), create an empty one
        if _lbph_recognizer is None:
            _lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=LBPH_RADIUS,
                neighbors=LBPH_NEIGHBORS,
                grid_x=LBPH_GRID_X,
                grid_y=LBPH_GRID_Y
            )
            print("⚠️ Created empty LBPH recognizer (no training data)")
    return _lbph_recognizer

def get_employee_folder(employee_id):
    """Get folder path for employee face images"""
    folder_path = os.path.join(DATA_DIR, str(employee_id))
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_face_image(employee_id, face_image, image_index=0):
    """Save face image to employee folder"""
    folder_path = get_employee_folder(employee_id)
    image_path = os.path.join(folder_path, f"face_{image_index:04d}.jpg")
    success = cv2.imwrite(image_path, face_image)
    if success:
        print(f"✅ Saved face image to {image_path}")
    else:
        print(f"⚠️ Failed to save face image to {image_path}")
    return image_path

def load_face_images_from_folder(employee_id):
    """Load all face images from employee folder"""
    folder_path = get_employee_folder(employee_id)
    face_images = []
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder does not exist: {folder_path}")
        return face_images
    
    # Load all face images from folder
    try:
        files = os.listdir(folder_path)
        print(f"📁 Found {len(files)} files in {folder_path}")
    except Exception as e:
        print(f"⚠️ Error listing folder {folder_path}: {e}")
        return face_images
    
    for filename in sorted(files):
        if filename.startswith('face_') and filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            try:
                face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if face_img is not None and face_img.size > 0:
                    # Ensure consistent size (200x200)
                    face_img = cv2.resize(face_img, (200, 200))
                    face_images.append(face_img)
                    print(f"  ✅ Loaded {filename} ({face_img.shape})")
                else:
                    print(f"  ⚠️ Failed to load {filename} (image is None or empty)")
            except Exception as e:
                print(f"  ⚠️ Error loading {image_path}: {e}")
    
    return face_images

def train_recognizer():
    """Train LBPH recognizer with all registered faces from folders"""
    global _lbph_recognizer, _employee_ids, _id_to_employee
    
    conn = get_db_connection()
    if conn is None:
        print("⚠️ Cannot train: Database connection failed")
        return
    
    c = conn.cursor()
    c.execute("SELECT id, name FROM employees ORDER BY id")
    rows = c.fetchall()
    
    if DB_URL:
        return_db_connection(conn)
    else:
        conn.close()
    
    if not rows:
        print("⚠️ No employees found in database for training")
        _employee_ids = {}
        _id_to_employee = {}
        return
    
    faces = []
    labels = []
    _employee_ids = {}
    _id_to_employee = {}
    
    print(f"Found {len(rows)} employees in database, loading face images...")
    
    for row in rows:
        emp_id = row[0] if DB_URL else row['id']
        name = row[1] if DB_URL else row['name']
        
        # Load face images from folder
        face_images = load_face_images_from_folder(emp_id)
        
        if not face_images:
            print(f"⚠️ No face images found for employee {emp_id} ({name}) in folder {os.path.join(DATA_DIR, str(emp_id))}")
            continue
        
        # Add all face images for this employee
        for face_img in face_images:
            faces.append(face_img)
            labels.append(emp_id)
        
        _employee_ids[emp_id] = name
        _id_to_employee[name] = emp_id
        print(f"✅ Loaded {len(face_images)} face images for {name} (ID: {emp_id})")
    
    if faces:
        # Create or reuse recognizer
        if _lbph_recognizer is None:
            _lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=LBPH_RADIUS,
                neighbors=LBPH_NEIGHBORS,
                grid_x=LBPH_GRID_X,
                grid_y=LBPH_GRID_Y
            )
        _lbph_recognizer.train(faces, np.array(labels))
        print(f"✅ LBPH trained with {len(faces)} face images from {len(_employee_ids)} employees")
        print(f"✅ Employee IDs mapped: {_employee_ids}")
    else:
        print("⚠️ No face images found for training - recognizer not trained")
        _lbph_recognizer = None

# ---------------- Image Processing ----------------
def decode_image(base64_string):
    """Decode base64 image to numpy array"""
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split(",")[1]
        img_data = base64.b64decode(base64_string)
        pil_image = PIL.Image.open(io.BytesIO(img_data))
        rgb_image = pil_image.convert("RGB")
        return np.array(rgb_image)
    except Exception as e:
        print(f"Image Decode Error: {e}")
        return None

def detect_face(img_array):
    """Detect face in image, return face ROI and grayscale face"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, None
    
    # Use largest face
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face
    
    face_roi = gray[y:y+h, x:x+w]
    face_roi_resized = cv2.resize(face_roi, (200, 200))
    
    return face_roi_resized, (x, y, w, h)

def detect_eyes(face_gray):
    """Detect eyes in face region for liveness check"""
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)
    return len(eyes) >= 2  # At least 2 eyes detected

# ---------------- Liveness Detection ----------------
def check_liveness(images):
    """
    Basic liveness detection: Check for movement between frames and eye presence.
    Returns True if liveness detected, False if likely spoofed.
    """
    if len(images) < 2:
        return False
    
    try:
        grays = []
        for img_array in images:
            if img_array is None:
                continue
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            grays.append(gray)
        
        if len(grays) < 2:
            return False
        
        # Check 1: Movement detection
        movements = []
        for i in range(len(grays) - 1):
            diff = cv2.absdiff(grays[i], grays[i+1])
            movement = np.mean(diff)
            movements.append(movement)
        
        avg_movement = np.mean(movements)
        
        # Check 2: Eye detection in last frame
        eyes_detected = False
        if len(grays) > 0:
            last_gray = grays[-1]
            # Resize for eye detection if needed
            if last_gray.shape[0] < 100:
                last_gray = cv2.resize(last_gray, (200, 200))
            eyes_detected = detect_eyes(last_gray)
        
        # Thresholds: Real faces have movement and eyes
        if avg_movement < 3.0:
            print(f"Liveness check failed: Low movement ({avg_movement:.2f})")
            return False
        
        if not eyes_detected:
            print("Liveness check failed: Eyes not detected")
            return False
        
        print(f"Liveness check passed: Movement={avg_movement:.2f}, Eyes={eyes_detected}")
        return True
        
    except Exception as e:
        print(f"Liveness check error: {e}")
        return False

# ---------------- Attendance helpers ----------------
def get_last_log_type(name, conn=None):
    """Return the last log type (LOGIN / LOGOUT) for *today* (IST) for a given employee"""
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    if conn is None:
        return None

    c = conn.cursor()
    try:
        _ist_timezone = pytz.timezone('Asia/Kolkata')
        # Get today's date in IST
        today_ist = datetime.datetime.now(_ist_timezone).date()
        
        if DB_URL:
            # PostgreSQL: Timestamps stored as UTC, convert to IST for date comparison
            today_str = today_ist.strftime('%Y-%m-%d')
            c.execute(
                "SELECT type FROM logs WHERE name = %s AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata') = %s "
                "ORDER BY id DESC LIMIT 1",
                (name, today_str),
            )
        else:
            # SQLite: Compare with IST date (timestamps stored as IST)
            today_str = today_ist.strftime('%Y-%m-%d')
            c.execute(
                "SELECT type FROM logs WHERE name = ? AND DATE(timestamp) = ? "
                "ORDER BY id DESC LIMIT 1",
                (name, today_str),
            )
        row = c.fetchone()
    finally:
        if should_close:
            if DB_URL:
                return_db_connection(conn)
            else:
                conn.close()

    if not row:
        return None

    return row[0] if DB_URL else row["type"]

# ---------------- Routes ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/api/clients', methods=['GET'])
def get_clients():
    """Return all client packages for the admin client table"""
    conn = get_db_connection()
    if conn is None:
        return jsonify([])
    c = conn.cursor()
    c.execute("SELECT * FROM clients ORDER BY id DESC")
    rows = c.fetchall()
    if DB_URL:
        return_db_connection(conn)
    else:
        conn.close()

    clients = []
    for row in rows:
        if DB_URL:
            cid, name, start_date, end_date, cost = row[0], row[1], row[2], row[3], row[4]
        else:
            cid, name, start_date, end_date, cost = row["id"], row["name"], row["start_date"], row["end_date"], row["cost"]

        try:
            if start_date:
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime("%d-%m-%Y")
            if end_date:
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%Y")
        except Exception:
            pass

        clients.append({
            "id": cid,
            "name": name,
            "start_date": start_date,
            "end_date": end_date,
            "cost": cost,
        })

    return jsonify(clients)

@app.route('/api/clients/add', methods=['POST'])
def add_client():
    """Add a new client package from the admin panel"""
    try:
        data = request.json or {}
        name = data.get("name")
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        cost = data.get("cost")

        if not name or not start_date or not end_date or not cost:
            return jsonify({"status": "error", "message": "Missing fields"}), 400

        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "DB unavailable"}), 500
        c = conn.cursor()

        if DB_URL:
            c.execute(
                "INSERT INTO clients (name, start_date, end_date, cost) VALUES (%s, %s, %s, %s)",
                (name, start_date, end_date, cost),
            )
        else:
            c.execute(
                "INSERT INTO clients (name, start_date, end_date, cost) VALUES (?, ?, ?, ?)",
                (name, start_date, end_date, cost),
            )
        conn.commit()
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()

        return jsonify({"status": "success", "message": "Client Added"})
    except Exception as e:
        print(f"ADD CLIENT ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/register', methods=['POST'])
def register():
    """Register a new employee and save face images to folder"""
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        image_data = data.get('image')
        
        if not name or not image_data:
            return jsonify({"status": "error", "message": "Missing Data"}), 400
        
        image = decode_image(image_data)
        if image is None:
            return jsonify({"status": "error", "message": "Invalid Image Format"}), 400
        
        # Detect and extract face
        face_gray, face_rect = detect_face(image)
        if face_gray is None:
            return jsonify({"status": "error", "message": "No face detected. Please ensure your face is clearly visible"}), 400
        
        # Check for eyes (liveness)
        if not detect_eyes(face_gray):
            return jsonify({"status": "error", "message": "Eyes not detected. Please look directly at the camera"}), 400
        
        # Insert employee into database first to get ID
        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        c = conn.cursor()
        if DB_URL:
            c.execute("INSERT INTO employees (name, email) VALUES (%s, %s) RETURNING id",
                      (name, email))
            employee_id = c.fetchone()[0]
        else:
            c.execute("INSERT INTO employees (name, email) VALUES (?, ?)",
                      (name, email))
            employee_id = c.lastrowid
        
        conn.commit()
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        
        # Save face image to folder
        save_face_image(employee_id, face_gray, image_index=0)
        
        # Retrain recognizer with new face
        train_recognizer()
        
        return jsonify({"status": "success", "message": f"Registered {name}!"})
    except Exception as e:
        print(f"REGISTER ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """Manually trigger LBPH recognizer training"""
    try:
        train_recognizer()
        return jsonify({
            "status": "success", 
            "message": "Recognizer trained successfully",
            "employee_count": len(_employee_ids),
            "employee_ids": _employee_ids
        })
    except Exception as e:
        print(f"TRAIN ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check recognizer status"""
    return jsonify({
        "recognizer_initialized": _lbph_recognizer is not None,
        "employee_count": len(_employee_ids),
        "employee_ids": _employee_ids,
        "data_dir": DATA_DIR,
        "data_dir_exists": os.path.exists(DATA_DIR)
    })

@app.route('/scan', methods=['POST'])
def scan():
    """Scan face and log attendance using LBPH recognizer"""
    try:
        data = request.json
        image_data_list = data.get('images', [data.get('image')])
        if not isinstance(image_data_list, list):
            image_data_list = [image_data_list]
        
        if not image_data_list:
            return jsonify({"status": "error", "message": "No images provided"}), 400
        
        # Decode all images for liveness check
        images = []
        for img_data in image_data_list:
            img = decode_image(img_data)
            if img is not None:
                images.append(img)
        
        if not images:
            return jsonify({"status": "error", "message": "Invalid images"}), 400
        
        # Liveness detection
        if len(images) > 1:
            if not check_liveness(images):
                return jsonify({"status": "error", "message": "Liveness check failed. Please use a live camera feed"}), 400
        
        # Use last image for recognition
        image = images[-1]
        face_gray, face_rect = detect_face(image)
        
        if face_gray is None:
            return jsonify({"status": "error", "message": "No face detected. Please ensure your face is clearly visible"}), 400
        
        # Recognize face using LBPH
        recognizer = get_lbph_recognizer()
        
        if recognizer is None or len(_employee_ids) == 0:
            print(f"⚠️ Recognition failed: recognizer={recognizer is not None}, employee_ids={len(_employee_ids)}")
            # Try retraining
            train_recognizer()
            recognizer = get_lbph_recognizer()
            if recognizer is None or len(_employee_ids) == 0:
                return jsonify({"status": "error", "message": "No registered users found. Please register employees first."}), 400
        
        label_id, confidence = recognizer.predict(face_gray)
        
        # LBPH: Lower confidence = better match (0 = perfect match)
        if confidence > LBPH_THRESHOLD:
            return jsonify({"status": "error", "message": "Face not recognized. Please register first"}), 401
        
        name = _employee_ids.get(label_id)
        if not name:
            return jsonify({"status": "error", "message": "Employee not found"}), 401
        
        # Log attendance - use IST timezone
        _ist_timezone = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.datetime.now(_ist_timezone)
        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        c = conn.cursor()
        last_type = get_last_log_type(name, conn)
        
        if last_type == "LOGIN":
            new_type = "LOGOUT"
            message = f"Goodbye, {name}!"
        elif last_type == "LOGOUT":
            if DB_URL:
                return_db_connection(conn)
            else:
                conn.close()
            return jsonify({
                "status": "error",
                "message": "Today's attendance is already marked for this employee",
            }), 400
        else:
            new_type = "LOGIN"
            message = f"Welcome, {name}!"
        
        # Store timestamp - for PostgreSQL, store as UTC then convert on retrieval
        # For SQLite, store as naive IST datetime
        if DB_URL:
            # Convert IST to UTC for PostgreSQL storage
            now_utc = now_ist.astimezone(pytz.UTC)
            now_to_store = now_utc.replace(tzinfo=None)  # Store as naive UTC
        else:
            # SQLite: Store as naive IST datetime
            now_to_store = now_ist.replace(tzinfo=None)
        
        if DB_URL:
            c.execute(
                "INSERT INTO logs (name, timestamp, type) VALUES (%s, %s, %s)",
                (name, now_to_store, new_type),
            )
            conn.commit()
            return_db_connection(conn)
        else:
            c.execute(
                "INSERT INTO logs (name, timestamp, type) VALUES (?, ?, ?)",
                (name, now_to_store, new_type),
            )
            conn.commit()
            conn.close()
        
        # Convert confidence to similarity score (0-100, higher = better)
        similarity = max(0, 100 - confidence)
        
        return jsonify({
            "status": "success",
            "name": name,
            "type": new_type,
            "message": message,
            "similarity": float(similarity)
        })
    except Exception as e:
        print(f"SCAN ERROR: {e}")
        return jsonify({"status": "error", "message": "Face recognition temporarily unavailable"}), 500

@app.route('/api/report', methods=['GET'])
def report():
    """Return raw log entries for the admin attendance table"""
    conn = get_db_connection()
    if conn is None:
        return jsonify([])
    c = conn.cursor()
    c.execute("SELECT name, timestamp, type FROM logs ORDER BY timestamp DESC")
    rows = c.fetchall()
    if DB_URL:
        return_db_connection(conn)
    else:
        conn.close()

    _ist_timezone = pytz.timezone('Asia/Kolkata')
    _utc_timezone = pytz.UTC
    data = []
    for row in rows:
        n = row[0] if DB_URL else row["name"]
        t = row[1] if DB_URL else row["timestamp"]
        s = row[2] if DB_URL else row["type"]

        # Parse timestamp
        if isinstance(t, str):
            try:
                t_obj = datetime.datetime.strptime(t.split(".")[0], "%Y-%m-%d %H:%M:%S")
            except Exception:
                try:
                    # Try with microseconds
                    t_obj = datetime.datetime.strptime(t.split(".")[0], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
        else:
            t_obj = t

        # Convert to IST based on database type
        if isinstance(t_obj, datetime.datetime) and t_obj.tzinfo is None:
            if DB_URL:
                # PostgreSQL: Stored as UTC (naive), so localize as UTC then convert to IST
                t_obj = _utc_timezone.localize(t_obj)
                t_obj = t_obj.astimezone(_ist_timezone)
            else:
                # SQLite: Stored as IST (naive), so localize as IST
                t_obj = _ist_timezone.localize(t_obj)
        elif isinstance(t_obj, datetime.datetime) and t_obj.tzinfo:
            # Already has timezone, convert to IST
            t_obj = t_obj.astimezone(_ist_timezone)

        data.append({
            "name": n,
            "date": t_obj.strftime("%d-%m-%Y"),
            "time": t_obj.strftime("%H:%M"),
            "type": s,
        })

    return jsonify(data)

# ---------------- Monthly Reports (Email) ----------------
def send_custom_email(to_email, subject, html_content):
    """Send email using Brevo API"""
    url = "https://api.brevo.com/v3/smtp/email"
    api_key = os.environ.get('BREVO_API_KEY')
    admin_email = os.environ.get('ADMIN_EMAIL')

    if not api_key:
        print("No BREVO_API_KEY found")
        return

    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json"
    }

    payload = {
        "sender": {"name": "Office Admin", "email": admin_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_content
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Email Failed: {e}")

def send_monthly_reports():
    """Send monthly attendance reports to all employees"""
    _ist_timezone = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(_ist_timezone)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    conn = get_db_connection()
    if conn is None:
        return "Database unavailable"
    
    c = conn.cursor()
    c.execute("SELECT name, email FROM employees WHERE email IS NOT NULL")
    employees = c.fetchall()

    emails_sent = 0

    for emp in employees:
        emp_name = emp[0] if DB_URL else emp['name']
        emp_email = emp[1] if DB_URL else emp['email']

        if not emp_email:
            continue

        if DB_URL:
            c.execute("SELECT timestamp, type FROM logs WHERE name=%s AND timestamp >= %s ORDER BY timestamp ASC",
                      (emp_name, start_of_month))
        else:
            c.execute("SELECT timestamp, type FROM logs WHERE name=? AND timestamp >= ? ORDER BY timestamp ASC",
                      (emp_name, start_of_month))

        logs = c.fetchall()
        total_seconds = 0
        last_login = None

        for log in logs:
            l_time = log[0] if DB_URL else log['timestamp']
            l_type = log[1] if DB_URL else log['type']

            if isinstance(l_time, str):
                try:
                    l_time = datetime.datetime.strptime(l_time.split('.')[0], "%Y-%m-%d %H:%M:%S")
                    # Convert based on database type
                    if DB_URL:
                        # PostgreSQL: Stored as UTC, convert to IST
                        l_time = pytz.UTC.localize(l_time).astimezone(_ist_timezone)
                    else:
                        # SQLite: Stored as IST
                        l_time = _ist_timezone.localize(l_time)
                except:
                    continue
            elif isinstance(l_time, datetime.datetime):
                # Convert based on database type
                if l_time.tzinfo is None:
                    if DB_URL:
                        # PostgreSQL: Stored as UTC
                        l_time = pytz.UTC.localize(l_time).astimezone(_ist_timezone)
                    else:
                        # SQLite: Stored as IST
                        l_time = _ist_timezone.localize(l_time)
                else:
                    l_time = l_time.astimezone(_ist_timezone)

            if l_type == 'LOGIN':
                last_login = l_time
            elif l_type == 'LOGOUT' and last_login:
                duration = (l_time - last_login).total_seconds()
                total_seconds += duration
                last_login = None

        total_hours = round(total_seconds / 3600, 2)

        subject = f"Monthly Attendance Report: {now.strftime('%B %Y')}"
        html_content = f"""
        <h3>Hello {emp_name},</h3>
        <p>Here is your attendance summary for <b>{now.strftime('%B')}</b>:</p>
        <h2>Total Hours: {total_hours} hrs</h2>
        <p>If you have any queries, please contact Admin before the 30th.</p>
        <p>Regards,<br>Office Admin</p>
        """

        send_custom_email(emp_email, subject, html_content)
        emails_sent += 1

    if DB_URL:
        return_db_connection(conn)
    else:
        conn.close()
    
    print(f"Monthly reports sent to {emails_sent} employees at {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
    return f"Report sent to {emails_sent} employees."

@app.route('/run-monthly-reports', methods=['GET'])
def run_monthly_reports():
    """Cron-job.org compatible endpoint for monthly reports"""
    secret = request.args.get('key')
    expected_secret = os.environ.get('CRON_SECRET', 'default-secret')
    
    if secret != expected_secret:
        return "Unauthorized", 401
    
    result = send_monthly_reports()
    return result

# ---------------- Scheduler ----------------
scheduler = None
_ist_timezone = pytz.timezone('Asia/Kolkata')

def setup_scheduler():
    global scheduler
    scheduler = BackgroundScheduler(timezone=_ist_timezone)
    scheduler.add_job(
        func=send_monthly_reports,
        trigger=CronTrigger(day=25, hour=17, minute=0),
        id='monthly_reports',
        name='Send monthly attendance reports',
        replace_existing=True
    )
    scheduler.start()
    print("✅ Scheduler started: Monthly reports will be sent on 25th of every month at 5:00 PM IST")

# Start scheduler only in main process
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or __name__ == '__main__':
    setup_scheduler()

# ---------------- Run App ----------------
if __name__ == '__main__':
    # Initialize LBPH recognizer
    print("🚀 Starting application...")
    print(f"📁 Data directory: {os.path.abspath(DATA_DIR)}")
    print(f"📁 Data directory exists: {os.path.exists(DATA_DIR)}")
    get_lbph_recognizer()
    print(f"✅ Recognizer initialized. Employee IDs: {_employee_ids}")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
