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
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from functools import wraps

load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
CORS(app)

DB_URL = os.environ.get('DATABASE_URL')

# LBPH Face Recognizer Configuration
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
LBPH_THRESHOLD = 80  # Lower = stricter matching (increased from 60 to be more lenient)

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
            pass
        except Exception as e:
            pass

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
        # Store employee ID, name, email (images stored separately in employee_images table)
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id SERIAL PRIMARY KEY, name TEXT, email TEXT)''')
        # Note: image column removed - images now stored in employee_images table
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id SERIAL PRIMARY KEY, name TEXT, timestamp TIMESTAMP, type TEXT)''')
        try:
            c.execute('''CREATE INDEX IF NOT EXISTS idx_logs_name_date ON logs(name, DATE(timestamp))''')
        except:
            pass
    else:
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT)''')
        # Note: image column removed - images now stored in employee_images table
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, timestamp DATETIME, type TEXT)''')
        try:
            c.execute('''CREATE INDEX IF NOT EXISTS idx_logs_name_date ON logs(name, DATE(timestamp))''')
        except:
            pass

    # Create employee_images table for storing multiple images per employee
    # NOTE: Data in NeonDB persists across Render deploys since it's an external managed database
    if DB_URL:
        c.execute('''CREATE TABLE IF NOT EXISTS employee_images
                     (id SERIAL PRIMARY KEY, employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE, 
                      image BYTEA NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        # Create index for faster queries
        try:
            c.execute('''CREATE INDEX IF NOT EXISTS idx_employee_images_employee_id ON employee_images(employee_id)''')
        except:
            pass
    else:
        c.execute('''CREATE TABLE IF NOT EXISTS employee_images
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE, 
                      image BLOB NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        try:
            c.execute('''CREATE INDEX IF NOT EXISTS idx_employee_images_employee_id ON employee_images(employee_id)''')
        except:
            pass

    if DB_URL:
        c.execute('''CREATE TABLE IF NOT EXISTS clients
                     (id SERIAL PRIMARY KEY, name TEXT, start_date TEXT, end_date TEXT, cost TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS leave_requests
                     (id SERIAL PRIMARY KEY, employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE,
                      employee_name TEXT, start_date DATE, end_date DATE, reason TEXT,
                      status TEXT DEFAULT 'pending', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    else:
        c.execute('''CREATE TABLE IF NOT EXISTS clients
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, start_date TEXT, end_date TEXT, cost TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS leave_requests
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE,
                      employee_name TEXT, start_date DATE, end_date DATE, reason TEXT,
                      status TEXT DEFAULT 'pending', created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
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
        train_recognizer()  # Train first, which will create the recognizer
        # If training didn't create a recognizer (no faces), create an empty one
        if _lbph_recognizer is None:
            _lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=LBPH_RADIUS,
                neighbors=LBPH_NEIGHBORS,
                grid_x=LBPH_GRID_X,
                grid_y=LBPH_GRID_Y
            )
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
    cv2.imwrite(image_path, face_image)
    return image_path

def load_face_images_from_db(employee_id, include_image_ids=False):
    """Load all face images from employee_images table for employee"""
    face_images = []
    image_ids = []
    
    conn = get_db_connection()
    if conn is None:
        return (face_images, image_ids) if include_image_ids else face_images
    
    try:
        c = conn.cursor()
        # Load all images for this employee from employee_images table
        if DB_URL:
            if include_image_ids:
                c.execute("SELECT id, image FROM employee_images WHERE employee_id = %s ORDER BY id", (employee_id,))
            else:
                c.execute("SELECT image FROM employee_images WHERE employee_id = %s ORDER BY id", (employee_id,))
        else:
            if include_image_ids:
                c.execute("SELECT id, image FROM employee_images WHERE employee_id = ? ORDER BY id", (employee_id,))
            else:
                c.execute("SELECT image FROM employee_images WHERE employee_id = ? ORDER BY id", (employee_id,))
        
        rows = c.fetchall()
        
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        
        if not rows:
            return (face_images, image_ids) if include_image_ids else face_images
        
        # Decode each image from bytes
        for idx, row in enumerate(rows):
            if include_image_ids:
                img_id = row[0] if DB_URL else row['id']
                image_data = row[1] if DB_URL else row['image']
            else:
                image_data = row[0] if DB_URL else row['image']
            
            if image_data is None:
                continue
            
            try:
                # Convert to bytes if it's a memoryview (PostgreSQL BYTEA)
                if isinstance(image_data, memoryview):
                    image_data = image_data.tobytes()
                elif not isinstance(image_data, bytes):
                    image_data = bytes(image_data)
                
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                face_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                
                if face_img is not None and face_img.size > 0:
                    # Ensure consistent size (200x200)
                    face_img = cv2.resize(face_img, (200, 200))
                    face_images.append(face_img)
                    if include_image_ids:
                        image_ids.append(img_id)
            except Exception:
                pass
    
    except Exception:
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
    
    return (face_images, image_ids) if include_image_ids else face_images

def train_recognizer():
    """Train LBPH recognizer with all registered faces from employee_images table in NeonDB"""
    global _lbph_recognizer, _employee_ids, _id_to_employee
    
    conn = get_db_connection()
    if conn is None:
        return
    
    c = conn.cursor()
    c.execute("SELECT id, name FROM employees ORDER BY id")
    rows = c.fetchall()
    
    if DB_URL:
        return_db_connection(conn)
    else:
        conn.close()
    
    if not rows:
        _employee_ids = {}
        _id_to_employee = {}
        _lbph_recognizer = None
        return
    
    faces = []
    labels = []
    _employee_ids = {}
    _id_to_employee = {}
    
    for row in rows:
        emp_id = row[0] if DB_URL else row['id']
        name = row[1] if DB_URL else row['name']
        
        # Load all face images from employee_images table for this employee
        face_images = load_face_images_from_db(emp_id, include_image_ids=False)
        
        if not face_images:
            continue
        
        # Add all face images for this employee to training data
        for face_img in face_images:
            # Ensure image is grayscale and 200x200
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = cv2.resize(face_img, (200, 200))
            faces.append(face_img)
            labels.append(emp_id)
        
        _employee_ids[emp_id] = name
        _id_to_employee[name] = emp_id
    
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
    else:
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
    except Exception:
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
            return False
        
        if not eyes_detected:
            return False
        
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

# ---------------- Authentication Helpers ----------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'employee_name' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------- Routes ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            data = request.json or {}
            username = data.get('username', '').strip().lower()
            password = data.get('password', '').strip().lower()
            
            if not username or not password:
                return jsonify({"status": "error", "message": "Username and password required"}), 400
            
            # Expected password format: "password" + username
            expected_password = f"password{username}"
            
            if password != expected_password:
                return jsonify({"status": "error", "message": "Invalid credentials"}), 401
            
            # Check if employee exists
            conn = get_db_connection()
            if conn is None:
                return jsonify({"status": "error", "message": "Database connection failed"}), 500
            
            c = conn.cursor()
            if DB_URL:
                c.execute("SELECT id, name FROM employees WHERE LOWER(name) = %s", (username,))
            else:
                c.execute("SELECT id, name FROM employees WHERE LOWER(name) = ?", (username,))
            
            row = c.fetchone()
            
            if DB_URL:
                return_db_connection(conn)
            else:
                conn.close()
            
            if not row:
                return jsonify({"status": "error", "message": "Invalid credentials"}), 401
            
            # Login successful - store in session
            employee_id = row[0] if DB_URL else row['id']
            employee_name = row[1] if DB_URL else row['name']
            session['employee_id'] = employee_id
            session['employee_name'] = employee_name
            
            return jsonify({
                "status": "success",
                "message": "Login successful",
                "employee_name": employee_name
            })
        except Exception as e:
            print(f"LOGIN ERROR: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    return render_template('login.html')

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    session.clear()
    return jsonify({"status": "success", "message": "Logged out successfully"})

@app.route('/api/current-user', methods=['GET'])
@login_required
def get_current_user():
    return jsonify({
        "status": "success",
        "employee_name": session.get('employee_name'),
        "employee_id": session.get('employee_id')
    })

@app.route('/')
@login_required
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

@app.route('/api/admin/leave-requests', methods=['GET'])
def get_all_leave_requests():
    """Get all leave requests for admin"""
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        c = conn.cursor()
        
        if DB_URL:
            c.execute("""
                SELECT id, employee_id, employee_name, start_date, end_date, reason, status, created_at
                FROM leave_requests
                ORDER BY created_at DESC
            """)
        else:
            c.execute("""
                SELECT id, employee_id, employee_name, start_date, end_date, reason, status, created_at
                FROM leave_requests
                ORDER BY created_at DESC
            """)
        
        rows = c.fetchall()
        
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        
        requests = []
        for row in rows:
            if DB_URL:
                req_id, emp_id, emp_name, start_date, end_date, reason, status, created_at = row
            else:
                req_id, emp_id, emp_name, start_date, end_date, reason, status, created_at = row['id'], row['employee_id'], row['employee_name'], row['start_date'], row['end_date'], row['reason'], row['status'], row['created_at']
            
            requests.append({
                'id': req_id,
                'employee_id': emp_id,
                'employee_name': emp_name,
                'start_date': str(start_date) if start_date else None,
                'end_date': str(end_date) if end_date else None,
                'reason': reason,
                'status': status,
                'created_at': str(created_at) if created_at else None
            })
        
        return jsonify({
            "status": "success",
            "requests": requests
        })
    except Exception as e:
        print(f"GET ALL LEAVE REQUESTS ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/leave-request/<int:request_id>', methods=['PUT'])
def update_leave_request_status(request_id):
    """Update leave request status (approve/reject)"""
    try:
        data = request.json or {}
        status = data.get('status')  # 'approved' or 'rejected'
        
        if status not in ['approved', 'rejected']:
            return jsonify({"status": "error", "message": "Invalid status. Must be 'approved' or 'rejected'"}), 400
        
        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        c = conn.cursor()
        
        if DB_URL:
            c.execute("""
                UPDATE leave_requests
                SET status = %s
                WHERE id = %s
            """, (status, request_id))
        else:
            c.execute("""
                UPDATE leave_requests
                SET status = ?
                WHERE id = ?
            """, (status, request_id))
        
        conn.commit()
        
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        
        return jsonify({
            "status": "success",
            "message": f"Leave request {status} successfully"
        })
    except Exception as e:
        print(f"UPDATE LEAVE REQUEST ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

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
    """Register a new employee and save face images to employee_images table in NeonDB"""
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
            return jsonify({"status": "error", "message": "Face not visible"}), 400
        
        # Check for eyes (liveness)
        if not detect_eyes(face_gray):
            return jsonify({"status": "error", "message": "Face not visible"}), 400
        
        # Convert face image to JPEG bytes for database storage
        # Ensure image is in correct format (grayscale, 200x200)
        face_gray_resized = cv2.resize(face_gray, (200, 200))
        _, img_buffer = cv2.imencode('.jpg', face_gray_resized)
        image_bytes = img_buffer.tobytes()
        
        # Insert employee into database (without image - images stored in employee_images table)
        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        c = conn.cursor()
        if DB_URL:
            c.execute("INSERT INTO employees (name, email) VALUES (%s, %s) RETURNING id",
                      (name, email))
            employee_id = c.fetchone()[0]
            # Insert image into employee_images table
            c.execute("INSERT INTO employee_images (employee_id, image) VALUES (%s, %s)",
                      (employee_id, psycopg2.Binary(image_bytes)))
        else:
            c.execute("INSERT INTO employees (name, email) VALUES (?, ?)",
                      (name, email))
            employee_id = c.lastrowid
            # Insert image into employee_images table
            c.execute("INSERT INTO employee_images (employee_id, image) VALUES (?, ?)",
                      (employee_id, sqlite3.Binary(image_bytes)))
        
        conn.commit()
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        
        # Retrain recognizer in background to speed up registration
        import threading
        threading.Thread(target=train_recognizer, daemon=True).start()
        
        return jsonify({
            "status": "success", 
            "message": f"Successfully registered {name}.",
            "employee_id": employee_id
        })
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

@app.route('/api/fix-image-association', methods=['POST'])
def fix_image_association():
    """Fix incorrect image-to-employee associations"""
    try:
        data = request.json
        image_id = data.get('image_id')
        correct_employee_id = data.get('correct_employee_id')
        
        if not image_id or not correct_employee_id:
            return jsonify({"status": "error", "message": "Missing image_id or correct_employee_id"}), 400
        
        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        c = conn.cursor()
        
        # Get current association
        if DB_URL:
            c.execute("SELECT employee_id FROM employee_images WHERE id = %s", (image_id,))
        else:
            c.execute("SELECT employee_id FROM employee_images WHERE id = ?", (image_id,))
        
        row = c.fetchone()
        if not row:
            if DB_URL:
                return_db_connection(conn)
            else:
                conn.close()
            return jsonify({"status": "error", "message": "Image not found"}), 404
        
        old_employee_id = row[0] if DB_URL else row['employee_id']
        
        # Update association
        if DB_URL:
            c.execute("UPDATE employee_images SET employee_id = %s WHERE id = %s", (correct_employee_id, image_id))
        else:
            c.execute("UPDATE employee_images SET employee_id = ? WHERE id = ?", (correct_employee_id, image_id))
        
        conn.commit()
        
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        
        # Retrain recognizer with corrected associations
        train_recognizer()
        
        return jsonify({
            "status": "success",
            "message": f"Image {image_id} reassociated from employee {old_employee_id} to {correct_employee_id}",
            "image_id": image_id,
            "old_employee_id": old_employee_id,
            "new_employee_id": correct_employee_id
        })
    except Exception as e:
        print(f"FIX ASSOCIATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/debug/employee-images', methods=['GET'])
def debug_employee_images():
    """Debug endpoint to check employee images in database"""
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        c = conn.cursor()
        
        # Get all employees with their image counts and image IDs
        if DB_URL:
            c.execute("""
                SELECT e.id, e.name, 
                       COUNT(ei.id) as image_count,
                       ARRAY_AGG(ei.id ORDER BY ei.id) as image_ids
                FROM employees e 
                LEFT JOIN employee_images ei ON e.id = ei.employee_id 
                GROUP BY e.id, e.name 
                ORDER BY e.id
            """)
        else:
            c.execute("""
                SELECT e.id, e.name, 
                       COUNT(ei.id) as image_count,
                       GROUP_CONCAT(ei.id) as image_ids
                FROM employees e 
                LEFT JOIN employee_images ei ON e.id = ei.employee_id 
                GROUP BY e.id, e.name 
                ORDER BY e.id
            """)
        
        rows = c.fetchall()
        
        employees = []
        for row in rows:
            if DB_URL:
                emp_id, name, img_count = row[0], row[1], row[2]
                image_ids = row[3] if len(row) > 3 else []
            else:
                emp_id, name, img_count = row['id'], row['name'], row['image_count']
                image_ids_str = row.get('image_ids', '')
                image_ids = [int(x) for x in image_ids_str.split(',')] if image_ids_str else []
            
            employees.append({
                "id": emp_id,
                "name": name,
                "image_count": img_count,
                "image_ids": image_ids if isinstance(image_ids, list) else list(image_ids) if image_ids else []
            })
        
        # Also get detailed image-to-employee mapping
        if DB_URL:
            c.execute("""
                SELECT ei.id as image_id, ei.employee_id, e.name as employee_name, ei.created_at
                FROM employee_images ei
                JOIN employees e ON ei.employee_id = e.id
                ORDER BY ei.employee_id, ei.id
            """)
        else:
            c.execute("""
                SELECT ei.id as image_id, ei.employee_id, e.name as employee_name, ei.created_at
                FROM employee_images ei
                JOIN employees e ON ei.employee_id = e.id
                ORDER BY ei.employee_id, ei.id
            """)
        
        image_details = []
        detail_rows = c.fetchall()
        for detail_row in detail_rows:
            if DB_URL:
                img_id, emp_id, emp_name, created_at = detail_row[0], detail_row[1], detail_row[2], detail_row[3]
            else:
                img_id, emp_id, emp_name, created_at = detail_row['image_id'], detail_row['employee_id'], detail_row['employee_name'], detail_row['created_at']
            image_details.append({
                "image_id": img_id,
                "employee_id": emp_id,
                "employee_name": emp_name,
                "created_at": str(created_at) if created_at else None
            })
        
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        
        # Check for potential issues
        issues = []
        for emp in employees:
            if emp['image_count'] == 0:
                issues.append(f"Employee {emp['id']} ({emp['name']}) has no images")
        
        return jsonify({
            "status": "success",
            "employees": employees,
            "image_details": image_details,
            "total_employees": len(employees),
            "total_images": sum(emp['image_count'] for emp in employees),
            "recognizer_trained": _lbph_recognizer is not None,
            "trained_employee_ids": list(_employee_ids.keys()) if _employee_ids else [],
            "trained_employee_names": list(_employee_ids.values()) if _employee_ids else [],
            "issues": issues,
            "instructions": "If recognition is wrong, check image_details to see which images belong to which employee_id. Use /api/fix-image-association to correct wrong associations."
        })
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check recognizer status"""
    # Check database for actual employee and image counts
    conn = get_db_connection()
    db_employee_count = 0
    db_image_count = 0
    if conn:
        try:
            c = conn.cursor()
            if DB_URL:
                c.execute("SELECT COUNT(*) FROM employees")
                db_employee_count = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM employee_images")
                db_image_count = c.fetchone()[0]
            else:
                c.execute("SELECT COUNT(*) FROM employees")
                db_employee_count = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM employee_images")
                db_image_count = c.fetchone()[0]
        except Exception as e:
            print(f"Error checking database status: {e}")
        finally:
            if DB_URL:
                return_db_connection(conn)
            else:
                conn.close()
    
    return jsonify({
        "recognizer_initialized": _lbph_recognizer is not None,
        "employee_count": len(_employee_ids),
        "employee_ids": _employee_ids,
        "db_employee_count": db_employee_count,
        "db_image_count": db_image_count,
        "data_dir": DATA_DIR,
        "data_dir_exists": os.path.exists(DATA_DIR)
    })

@app.route('/scan', methods=['POST'])
@login_required
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
            return jsonify({"status": "error", "message": "Face not visible"}), 400
        
        # Ensure face is exactly 200x200 grayscale (same as training)
        # detect_face already resizes to 200x200, but ensure it's grayscale
        if len(face_gray.shape) == 3:
            face_gray = cv2.cvtColor(face_gray, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (200, 200))
        
        # Recognize face using LBPH
        recognizer = get_lbph_recognizer()
        
        if recognizer is None or len(_employee_ids) == 0:
            train_recognizer()
            recognizer = get_lbph_recognizer()
            if recognizer is None or len(_employee_ids) == 0:
                return jsonify({"status": "error", "message": "No registered users found. Please register employees first."}), 400
        
        label_id, confidence = recognizer.predict(face_gray)
        
        # LBPH: Lower confidence = better match (0 = perfect match)
        if confidence > LBPH_THRESHOLD:
            return jsonify({"status": "error", "message": "Face not visible. Please ensure your face is clearly visible."}), 401
        
        name = _employee_ids.get(label_id)
        if not name:
            return jsonify({"status": "error", "message": "Face not visible. Please ensure your face is clearly visible."}), 401
        
        # Log attendance - use IST timezone
        _ist_timezone = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.datetime.now(_ist_timezone)
        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        c = conn.cursor()
        last_type = get_last_log_type(name, conn)
        
        # Format time nicely
        time_str = now_ist.strftime("%I:%M %p")
        
        if last_type == "LOGIN":
            new_type = "LOGOUT"
            message = f"Goodbye, {time_str}"
        elif last_type == "LOGOUT":
            if DB_URL:
                return_db_connection(conn)
            else:
                conn.close()
            return jsonify({
                "status": "error",
                "message": "You have logged out",
            }), 400
        else:
            new_type = "LOGIN"
            message = f"Welcome, {time_str}"
        
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
        
        return jsonify({
            "status": "success",
            "name": name,
            "type": new_type,
            "message": message
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

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """Return list of all employees"""
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        c = conn.cursor()
        c.execute("SELECT id, name, email FROM employees ORDER BY name")
        rows = c.fetchall()
        
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        
        employees = []
        for row in rows:
            emp_id = row[0] if DB_URL else row['id']
            name = row[1] if DB_URL else row['name']
            email = row[2] if DB_URL else row['email']
            employees.append({
                'id': emp_id,
                'name': name,
                'email': email
            })
        
        return jsonify({
            "status": "success",
            "employees": employees
        })
    except Exception as e:
        print(f"GET EMPLOYEES ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/leave-request', methods=['POST'])
@login_required
def submit_leave_request():
    """Submit a leave request"""
    try:
        data = request.json or {}
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        reason = data.get('reason', '').strip()
        
        if not start_date or not end_date:
            return jsonify({"status": "error", "message": "Start date and end date required"}), 400
        
        employee_id = session.get('employee_id')
        employee_name = session.get('employee_name')
        
        if not employee_id or not employee_name:
            return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        c = conn.cursor()
        
        if DB_URL:
            c.execute("""
                INSERT INTO leave_requests (employee_id, employee_name, start_date, end_date, reason, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
            """, (employee_id, employee_name, start_date, end_date, reason))
        else:
            c.execute("""
                INSERT INTO leave_requests (employee_id, employee_name, start_date, end_date, reason, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
            """, (employee_id, employee_name, start_date, end_date, reason))
        
        conn.commit()
        
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        
        return jsonify({
            "status": "success",
            "message": "Leave request submitted successfully"
        })
    except Exception as e:
        print(f"LEAVE REQUEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/my-leave-requests', methods=['GET'])
@login_required
def get_my_leave_requests():
    """Get leave requests for the logged-in employee"""
    try:
        employee_id = session.get('employee_id')
        if not employee_id:
            return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        c = conn.cursor()
        
        if DB_URL:
            c.execute("""
                SELECT id, start_date, end_date, reason, status, created_at
                FROM leave_requests
                WHERE employee_id = %s
                ORDER BY created_at DESC
            """, (employee_id,))
        else:
            c.execute("""
                SELECT id, start_date, end_date, reason, status, created_at
                FROM leave_requests
                WHERE employee_id = ?
                ORDER BY created_at DESC
            """, (employee_id,))
        
        rows = c.fetchall()
        
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        
        requests = []
        for row in rows:
            if DB_URL:
                req_id, start_date, end_date, reason, status, created_at = row
            else:
                req_id, start_date, end_date, reason, status, created_at = row['id'], row['start_date'], row['end_date'], row['reason'], row['status'], row['created_at']
            
            requests.append({
                'id': req_id,
                'start_date': str(start_date) if start_date else None,
                'end_date': str(end_date) if end_date else None,
                'reason': reason,
                'status': status,
                'created_at': str(created_at) if created_at else None
            })
        
        return jsonify({
            "status": "success",
            "requests": requests
        })
    except Exception as e:
        print(f"GET LEAVE REQUESTS ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/user-attendance', methods=['GET'])
@login_required
def user_attendance():
    """Return daily attendance data for the logged-in user and month"""
    try:
        # Get name from session instead of parameter
        name = session.get('employee_name')
        month = request.args.get('month')  # Format: YYYY-MM
        year = request.args.get('year')
        
        if not name:
            return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
        _ist_timezone = pytz.timezone('Asia/Kolkata')
        _utc_timezone = pytz.UTC
        
        # Determine the month and year
        if month and year:
            try:
                target_date = datetime.datetime(int(year), int(month), 1)
            except:
                return jsonify({"status": "error", "message": "Invalid month/year"}), 400
        else:
            # Default to current month
            target_date = datetime.datetime.now(_ist_timezone).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Get start and end of month in IST
        start_of_month = target_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start_of_month.month == 12:
            end_of_month = start_of_month.replace(year=start_of_month.year + 1, month=1) - datetime.timedelta(days=1)
        else:
            end_of_month = start_of_month.replace(month=start_of_month.month + 1) - datetime.timedelta(days=1)
        end_of_month = end_of_month.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Get logs from a few days before month start to catch cross-month sessions
        # (e.g., if someone logged in on last day of previous month and logged out on first day of current month)
        query_start = start_of_month - datetime.timedelta(days=3)
        
        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        c = conn.cursor()
        
        # Get all logs for this user in this month (plus a few days before to catch cross-month sessions)
        if DB_URL:
            # PostgreSQL: Convert UTC to IST for date comparison
            query_start_str = query_start.strftime('%Y-%m-%d')
            end_str = end_of_month.strftime('%Y-%m-%d')
            c.execute("""
                SELECT timestamp, type 
                FROM logs 
                WHERE name = %s 
                AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata') >= %s
                AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata') <= %s
                ORDER BY timestamp ASC
            """, (name, query_start_str, end_str))
        else:
            # SQLite: Direct date comparison
            query_start_str = query_start.strftime('%Y-%m-%d')
            end_str = end_of_month.strftime('%Y-%m-%d')
            c.execute("""
                SELECT timestamp, type 
                FROM logs 
                WHERE name = ? 
                AND DATE(timestamp) >= ?
                AND DATE(timestamp) <= ?
                ORDER BY timestamp ASC
            """, (name, query_start_str, end_str))
        
        rows = c.fetchall()
        
        # Get approved leave dates for this employee in this month
        employee_id = session.get('employee_id')
        approved_leaves = set()
        paid_leave_dates = set()  # First approved leave day (paid)
        unpaid_leave_dates = set()  # Additional approved leave days (unpaid)
        
        if employee_id:
            if DB_URL:
                c.execute("""
                    SELECT start_date, end_date
                    FROM leave_requests
                    WHERE employee_id = %s
                    AND status = 'approved'
                    AND start_date <= %s
                    AND end_date >= %s
                    ORDER BY start_date ASC
                """, (employee_id, end_of_month.date(), start_of_month.date()))
            else:
                c.execute("""
                    SELECT start_date, end_date
                    FROM leave_requests
                    WHERE employee_id = ?
                    AND status = 'approved'
                    AND start_date <= ?
                    AND end_date >= ?
                    ORDER BY start_date ASC
                """, (employee_id, end_of_month.date(), start_of_month.date()))
            
            leave_rows = c.fetchall()
            all_leave_dates = []
            
            for leave_row in leave_rows:
                if DB_URL:
                    leave_start = leave_row[0]
                    leave_end = leave_row[1]
                else:
                    leave_start = leave_row['start_date']
                    leave_end = leave_row['end_date']
                
                # Convert to date objects if needed
                if isinstance(leave_start, str):
                    leave_start = datetime.datetime.strptime(leave_start, '%Y-%m-%d').date()
                if isinstance(leave_end, str):
                    leave_end = datetime.datetime.strptime(leave_end, '%Y-%m-%d').date()
                
                # Add all dates in the leave range
                current_leave_date = leave_start
                while current_leave_date <= leave_end:
                    if start_of_month.date() <= current_leave_date <= end_of_month.date():
                        date_str = current_leave_date.strftime('%Y-%m-%d')
                        all_leave_dates.append(date_str)
                        approved_leaves.add(date_str)
                    current_leave_date += datetime.timedelta(days=1)
            
            # Sort and separate: first day is paid leave, rest are unpaid
            all_leave_dates.sort()
            if len(all_leave_dates) > 0:
                paid_leave_dates.add(all_leave_dates[0])  # First approved leave day is paid
                for date_str in all_leave_dates[1:]:
                    unpaid_leave_dates.add(date_str)  # Additional days are unpaid
        
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        
        # Process logs to calculate daily hours
        daily_data = {}  # {date: {'hours': float, 'minutes': int, 'has_logs': bool}}
        
        # Initialize all days in the month
        current_date = start_of_month
        while current_date <= end_of_month:
            date_str = current_date.strftime('%Y-%m-%d')
            daily_data[date_str] = {'hours': 0, 'minutes': 0, 'has_logs': False}
            current_date += datetime.timedelta(days=1)
        
        # Process logs
        last_login = None
        for row in rows:
            t = row[0] if DB_URL else row['timestamp']
            log_type = row[1] if DB_URL else row['type']
            
            # Parse and convert timestamp to IST
            if isinstance(t, str):
                try:
                    t_obj = datetime.datetime.strptime(t.split(".")[0], "%Y-%m-%d %H:%M:%S")
                except:
                    continue
            else:
                t_obj = t
            
            # Convert to IST
            if isinstance(t_obj, datetime.datetime) and t_obj.tzinfo is None:
                if DB_URL:
                    t_obj = _utc_timezone.localize(t_obj).astimezone(_ist_timezone)
                else:
                    t_obj = _ist_timezone.localize(t_obj)
            elif isinstance(t_obj, datetime.datetime) and t_obj.tzinfo:
                t_obj = t_obj.astimezone(_ist_timezone)
            
            date_str = t_obj.strftime('%Y-%m-%d')
            
            if date_str not in daily_data:
                continue
            
            daily_data[date_str]['has_logs'] = True
            
            if log_type == 'LOGIN':
                last_login = t_obj
            elif log_type == 'LOGOUT' and last_login:
                # Calculate duration
                duration = t_obj - last_login
                total_seconds = duration.total_seconds()
                
                # Attribute hours to the login date, not logout date
                login_date_str = last_login.strftime('%Y-%m-%d')
                
                # If login date is in the current month, add hours to that date
                if login_date_str in daily_data:
                    existing_seconds = daily_data[login_date_str]['hours'] * 3600 + daily_data[login_date_str]['minutes'] * 60
                    total_seconds += existing_seconds
                    
                    hours = int(total_seconds // 3600)
                    minutes = int((total_seconds % 3600) // 60)
                    
                    daily_data[login_date_str]['hours'] = hours
                    daily_data[login_date_str]['minutes'] = minutes
                    daily_data[login_date_str]['has_logs'] = True
                
                # Also mark logout date as having logs
                daily_data[date_str]['has_logs'] = True
                last_login = None
        
        # Calculate monthly totals and expected hours
        total_calculated_minutes = 0
        total_expected_days = 0
        
        # Convert to list format with status indicators
        result = []
        for date_str in sorted(daily_data.keys()):
            # Parse date to check if it's Sunday
            date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            is_sunday = date_obj.weekday() == 6  # 6 = Sunday
            is_approved_leave = date_str in approved_leaves
            is_paid_leave = date_str in paid_leave_dates
            is_unpaid_leave = date_str in unpaid_leave_dates
            
            data = daily_data[date_str]
            total_minutes = data['hours'] * 60 + data['minutes']
            target_minutes = 7 * 60 + 40  # 7 hours 40 minutes
            
            # Count expected working days (exclude Sundays and paid leave only)
            # Unpaid leave days still count as expected working days
            if not is_sunday and not is_paid_leave:
                total_expected_days += 1
            
            # Add to total calculated minutes
            total_calculated_minutes += total_minutes
            
            # Determine status - both paid and unpaid leave are blue
            if is_paid_leave or is_unpaid_leave:
                status = 'blue'  # Approved leave (both paid and unpaid)
            elif not data['has_logs']:
                status = 'red'  # No logs
            elif total_minutes >= target_minutes:
                status = 'green'  # >= 7hrs 40mins
            else:
                status = 'yellow'  # < 7hrs 40mins but has logs
            
            result.append({
                'date': date_str,
                'hours': data['hours'],
                'minutes': data['minutes'],
                'total_minutes': total_minutes,
                'status': status,
                'is_sunday': is_sunday,
                'is_approved_leave': is_approved_leave,
                'is_paid_leave': is_paid_leave,
                'is_unpaid_leave': is_unpaid_leave
            })
        
        # Calculate total expected hours: total_expected_days * 7.67 hours
        # Note: We already excluded paid leave days from total_expected_days
        # The 1 paid leave per month is already handled by excluding the first approved leave day
        total_expected_hours = total_expected_days * (7 + 40/60)  # 7 hours 40 minutes per day
        total_calculated_hours = total_calculated_minutes / 60
        
        return jsonify({
            "status": "success",
            "month": target_date.strftime('%Y-%m'),
            "data": result,
            "monthly_hours": {
                "calculated_hours": round(total_calculated_hours, 2),
                "total_expected_hours": round(total_expected_hours, 2),
                "calculated_minutes": total_calculated_minutes
            }
        })
    except Exception as e:
        print(f"USER ATTENDANCE ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

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
        <h3>Dear {emp_name},</h3>
        <p>Here is your attendance summary for <b>{now.strftime('%B')}</b>:</p>
        <h2>Total Hours: {total_hours} hrs</h2>
        <p>If you have any queries, please contact before the 30th.</p>
        <p>Regards,<br>Office</p>
        """
        send_custom_email(emp_email, subject, html_content)
        emails_sent += 1

    if DB_URL:
        return_db_connection(conn)
    else:
        conn.close()
    
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
    
# ---------------- Run App ----------------
if __name__ == '__main__':
    # Initialize LBPH recognizer
    get_lbph_recognizer()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
