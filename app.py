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
import json
import PIL.Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

load_dotenv()

app = Flask(__name__)
CORS(app)

DB_URL = os.environ.get('DATABASE_URL')

# Connection pool for Postgres (reduces connection overhead)
_postgres_pool = None

# DeepFace model name and matching threshold
# Using VGG-Face (faster) instead of Facenet512 (more accurate but slower)
DEEPFACE_MODEL = os.environ.get("DEEPFACE_MODEL", "VGG-Face")
COSINE_THRESHOLD = float(os.environ.get("DEEPFACE_THRESHOLD", 0.60))  # VGG-Face uses higher threshold

def init_connection_pool():
    """Initialize connection pool for Postgres"""
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
    """Get database connection - uses pool for Postgres"""
    if DB_URL:
        global _postgres_pool
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
    """Return connection to pool (for Postgres)"""
    if DB_URL and _postgres_pool and conn:
        try:
            _postgres_pool.putconn(conn)
        except Exception:
            pass

def init_db():
    # Use direct connection for init (pool not ready yet)
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
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id SERIAL PRIMARY KEY, name TEXT, email TEXT, encoding BYTEA)''')
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id SERIAL PRIMARY KEY, name TEXT, timestamp TIMESTAMP, type TEXT)''')
        try:
            c.execute('''CREATE INDEX IF NOT EXISTS idx_logs_name_date ON logs(name, DATE(timestamp))''')
        except:
            pass
    else:
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, encoding BLOB)''')
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

# --- HELPER: Check Status (Login vs Logout) ---
def get_last_log_type(name, conn=None):
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    if conn is None:
        return None

    c = conn.cursor()

    if DB_URL:
        today = datetime.date.today()
        c.execute("SELECT type FROM logs WHERE name = %s AND DATE(timestamp) = %s ORDER BY id DESC LIMIT 1",
                  (name, today))
    else:
        c.execute("SELECT type FROM logs WHERE name = ? AND DATE(timestamp) = DATE('now') ORDER BY id DESC LIMIT 1",
                  (name,))

    row = c.fetchone()

    if should_close:
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()

    if row:
        return row[0] if DB_URL else row['type']

    return None

# --- THE FIX: Robust Image Decoding ---
def decode_image(base64_string):
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split(",")[1]

        img_data = base64.b64decode(base64_string)

        pil_image = PIL.Image.open(io.BytesIO(img_data))
        rgb_image = pil_image.convert("RGB")

        # Resize aggressively for faster processing (max 400px width for DeepFace)
        width, height = rgb_image.size
        if width > 400:
            ratio = 400 / width
            new_height = int(height * ratio)
            rgb_image = rgb_image.resize((400, new_height), PIL.Image.Resampling.LANCZOS)

        return np.array(rgb_image)
    except Exception as e:
        print(f"Image Decode Error: {e}")
        return None

def detect_spoofing(images):
    """
    Simple anti-spoofing: Check for movement and texture variance.
    Photos/prints are static and have less texture variation.
    Returns True if spoofing detected, False if likely real face.
    """
    if len(images) < 2:
        return False  # Need at least 2 frames
    
    try:
        # Import cv2 (opencv-python-headless)
        import cv2
        
        # Convert images to grayscale for comparison
        grays = []
        for img_array in images:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            grays.append(gray)
        
        # Check 1: Movement detection - compare consecutive frames
        movements = []
        for i in range(len(grays) - 1):
            # Calculate frame difference
            diff = cv2.absdiff(grays[i], grays[i+1])
            movement = np.mean(diff)
            movements.append(movement)
        
        avg_movement = np.mean(movements)
        
        # Check 2: Texture variance - real faces have more texture
        texture_variances = []
        for gray in grays:
            # Calculate Laplacian variance (measures texture)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_var = laplacian.var()
            texture_variances.append(texture_var)
        
        avg_texture = np.mean(texture_variances)
        
        # Thresholds (tuned for typical webcam quality)
        # Photos/prints: low movement (< 5) and low texture (< 100)
        # Real faces: some movement (> 3) and higher texture (> 80)
        if avg_movement < 3.0:
            print(f"Anti-spoofing: Low movement detected ({avg_movement:.2f})")
            return True  # Likely a photo/print
        
        if avg_texture < 80.0:
            print(f"Anti-spoofing: Low texture variance ({avg_texture:.2f})")
            return True  # Likely a photo/print
        
        print(f"Anti-spoofing: Movement={avg_movement:.2f}, Texture={avg_texture:.2f} - PASSED")
        return False  # Likely real face
        
    except ImportError:
        # If cv2 not available, fall back to basic numpy check
        print("Warning: OpenCV not available, using basic anti-spoofing")
        if len(images) < 2:
            return False
        
        # Simple pixel difference check
        movements = []
        for i in range(len(images) - 1):
            if len(images[i].shape) == 3:
                img1 = np.mean(images[i], axis=2)  # Convert to grayscale
                img2 = np.mean(images[i+1], axis=2)
            else:
                img1 = images[i]
                img2 = images[i+1]
            
            diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
            movement = np.mean(diff)
            movements.append(movement)
        
        avg_movement = np.mean(movements)
        if avg_movement < 2.0:
            return True  # Very static, likely photo
        
        return False
    except Exception as e:
        print(f"Anti-spoofing check failed: {e}")
        return False  # If check fails, allow (fail open)

# Cache for embeddings
_face_cache = None
_cache_timestamp = None
CACHE_DURATION = 300  # 5 minutes cache

def get_known_faces():
    global _face_cache, _cache_timestamp

    now = datetime.datetime.now()
    if _face_cache is not None and _cache_timestamp is not None:
        if (now - _cache_timestamp).total_seconds() < CACHE_DURATION:
            return _face_cache

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM employees")
    rows = c.fetchall()
    if DB_URL:
        return_db_connection(conn)
    else:
        conn.close()

    known_names = []
    known_encodings = []

    for row in rows:
        if DB_URL:
            name = row[0]
            encoding_bytes = row[1]
        else:
            name = row['name']
            encoding_bytes = row['encoding']

        if encoding_bytes:
            try:
                arr = np.frombuffer(encoding_bytes, dtype=np.float32)
                known_names.append(name)
                known_encodings.append(arr)
            except Exception as e:
                print(f"Failed to decode stored embedding for {name}: {e}")

    _face_cache = (known_names, known_encodings)
    _cache_timestamp = now

    return known_names, known_encodings

def invalidate_face_cache():
    global _face_cache, _cache_timestamp
    _face_cache = None
    _cache_timestamp = None

# ========== DeepFace helpers ==========

# Global DeepFace instance to avoid reloading model
_deepface_model = None

def get_deepface():
    """Lazy load DeepFace to avoid import errors at startup"""
    global _deepface_model
    if _deepface_model is None:
        try:
            from deepface import DeepFace
            _deepface_model = DeepFace
        except ImportError as e:
            error_msg = str(e)
            if "tensorflow" in error_msg.lower() or "dll" in error_msg.lower():
                raise RuntimeError(
                    f"TensorFlow installation issue detected.\n"
                    f"Please try:\n"
                    f"1. Uninstall: pip uninstall tensorflow tensorflow-cpu -y\n"
                    f"2. Install: pip install tensorflow-cpu>=2.13.0\n"
                    f"3. If still failing, install Visual C++ Redistributable from:\n"
                    f"   https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                    f"Original error: {e}"
                )
            else:
                raise RuntimeError(
                    f"DeepFace not installed. Please run: pip install deepface tensorflow-cpu\n"
                    f"Original error: {e}"
                )
    return _deepface_model

def compute_embedding(img_array, model_name=DEEPFACE_MODEL, enforce_detection=True):
    """
    Compute embedding (vector) for an image numpy array using DeepFace.
    Returns a 1D numpy float32 array.
    Optimized for speed: uses smaller model and reduced image size.
    """
    try:
        DeepFace = get_deepface()
    except RuntimeError as e:
        # Re-raise with better context
        raise RuntimeError(f"Failed to initialize DeepFace: {e}")
    
    try:
        # DeepFace.represent accepts numpy array directly
        # Using backend='opencv' for faster processing
        rep = DeepFace.represent(
            img_path=img_array,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend='opencv'  # Faster than 'ssd' or 'mtcnn'
        )
        
        # DeepFace.represent returns a list of dictionaries, each with 'embedding' key
        # Handle different return formats
        if isinstance(rep, dict):
            # Single dictionary with 'embedding' key
            if 'embedding' in rep:
                vec = np.array(rep['embedding'], dtype=np.float32)
            else:
                raise RuntimeError(f"DeepFace returned dict without 'embedding' key. Keys: {rep.keys()}")
        elif isinstance(rep, list):
            if len(rep) == 0:
                raise RuntimeError("DeepFace returned empty representation")
            # List of dictionaries or list of arrays
            first_item = rep[0]
            if isinstance(first_item, dict):
                # List of dicts: [{'embedding': [...], ...}, ...]
                if 'embedding' in first_item:
                    vec = np.array(first_item['embedding'], dtype=np.float32)
                else:
                    raise RuntimeError(f"DeepFace dict item missing 'embedding' key. Keys: {first_item.keys()}")
            elif isinstance(first_item, (list, tuple, np.ndarray)):
                # List of arrays: [[...], ...]
                vec = np.array(first_item, dtype=np.float32)
            else:
                # List of numbers: [1, 2, 3, ...]
                vec = np.array(rep, dtype=np.float32)
        elif isinstance(rep, np.ndarray):
            vec = rep.astype(np.float32)
        else:
            raise RuntimeError(f"Unexpected DeepFace return type: {type(rep)}. Value: {rep}")
        
        # Flatten to 1D if needed
        if vec.ndim > 1:
            vec = vec.flatten()
            
        return vec
    except Exception as e:
        error_str = str(e).lower()
        if "dll" in error_str or "tensorflow" in error_str or "_pywrap_tensorflow" in error_str:
            raise RuntimeError(
                f"TensorFlow DLL error detected. This is a common Windows issue.\n"
                f"Solutions:\n"
                f"1. Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                f"2. Reinstall: pip uninstall tensorflow tensorflow-cpu -y && pip install tensorflow-cpu>=2.13.0\n"
                f"3. Restart your computer after installing Visual C++ Redistributable\n"
                f"Original error: {e}"
            )
        else:
            raise RuntimeError(f"DeepFace embedding failed: {str(e)}")

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_user_friendly_error(error_msg):
    """
    Convert technical DeepFace errors to simple, user-friendly messages.
    Focuses on actionable advice like better lighting.
    """
    error_lower = str(error_msg).lower()
    
    # Face detection errors
    if "no face" in error_lower or "face could not be detected" in error_lower or "enforce_detection" in error_lower:
        return "Please ensure your face is clearly visible in good lighting"
    
    # Dimension/shape mismatch errors
    if "shapes" in error_lower and "not aligned" in error_lower:
        return "Face data mismatch. Please re-register your face"
    
    # Embedding errors
    if "embedding" in error_lower and "failed" in error_lower:
        return "Could not process face. Try scanning in better lighting"
    
    # General DeepFace errors
    if "deepface" in error_lower or "tensorflow" in error_lower or "dll" in error_lower:
        return "Face recognition temporarily unavailable. Please try again"
    
    # Unknown person / low similarity
    if "unknown" in error_lower or "not recognized" in error_lower:
        return "Face not recognized. Please ensure good lighting and face the camera directly"
    
    # Default user-friendly message
    return "Face not recognized. Please scan in better lighting and ensure your face is clearly visible"

# ========== ROUTES ==========

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/admin')
def admin_page():
    return render_template('admin.html')

@app.route('/api/clients', methods=['GET'])
def get_clients():
    conn = get_db_connection()
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
            cid = row[0]
            name = row[1]
            start_date = row[2]
            end_date = row[3]
            cost = row[4]
        else:
            cid = row['id']
            name = row['name']
            start_date = row['start_date']
            end_date = row['end_date']
            cost = row['cost']

        try:
            if start_date:
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime("%d-%m-%Y")
            if end_date:
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%Y")
        except ValueError:
            pass

        clients.append({
            "id": cid,
            "name": name,
            "start_date": start_date,
            "end_date": end_date,
            "cost": cost
        })

    return jsonify(clients)

@app.route('/api/clients/add', methods=['POST'])
def add_client():
    try:
        data = request.json
        conn = get_db_connection()
        c = conn.cursor()

        if DB_URL:
            c.execute("INSERT INTO clients (name, start_date, end_date, cost) VALUES (%s, %s, %s, %s)",
                      (data['name'], data['start_date'], data['end_date'], data['cost']))
        else:
            c.execute("INSERT INTO clients (name, start_date, end_date, cost) VALUES (?, ?, ?, ?)",
                      (data['name'], data['start_date'], data['end_date'], data['cost']))

        conn.commit()
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        return jsonify({"status": "success", "message": "Client Added"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/report')
def report():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT name, timestamp, type FROM logs ORDER BY timestamp DESC")
    rows = c.fetchall()
    if DB_URL:
        return_db_connection(conn)
    else:
        conn.close()

    data = []
    for row in rows:
        n = row[0] if DB_URL else row['name']
        t = row[1] if DB_URL else row['timestamp']
        s = row[2] if DB_URL else row['type']

        if isinstance(t, str):
            t_obj = datetime.datetime.strptime(t.split('.')[0], "%Y-%m-%d %H:%M:%S")
        else:
            t_obj = t

        data.append({
            "name": n,
            "date": t_obj.strftime("%d-%m-%Y"),
            "time": t_obj.strftime("%H:%M"),
            "type": s
        })
    return jsonify(data)

# ---------------- Registration (DeepFace) ----------------
@app.route('/register', methods=['POST'])
@app.route('/api/deepface/register', methods=['POST'])  # alias
def register():
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

        # Compute embedding using DeepFace
        try:
            embedding = compute_embedding(image, model_name=DEEPFACE_MODEL, enforce_detection=True)
        except Exception as e:
            friendly_msg = get_user_friendly_error(str(e))
            print(f"REGISTER ERROR (converted): {e}")
            return jsonify({"status": "error", "message": friendly_msg}), 400
            
        if embedding is None or len(embedding) == 0:
            return jsonify({"status": "error", "message": "Could not detect face. Please ensure good lighting and face the camera"}), 400

        # store as float32 bytes
        embedding_bytes = embedding.astype(np.float32).tobytes()

        conn = get_db_connection()
        c = conn.cursor()

        if DB_URL:
            c.execute("INSERT INTO employees (name, email, encoding) VALUES (%s, %s, %s)",
                      (name, email, psycopg2.Binary(embedding_bytes)))
        else:
            c.execute("INSERT INTO employees (name, email, encoding) VALUES (?, ?, ?)",
                      (name, email, sqlite3.Binary(embedding_bytes)))

        conn.commit()
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()

        invalidate_face_cache()

        return jsonify({"status": "success", "message": f"Registered {name}!"})

    except Exception as e:
        print(f"REGISTER ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------- Scan (DeepFace compare embeddings) ----------------
@app.route('/scan', methods=['POST'])
@app.route('/api/deepface/scan', methods=['POST'])  # alias
def scan():
    try:
        data = request.json
        # Support both single image (backward compat) and multiple images (anti-spoofing)
        image_data_list = data.get('images', [data.get('image')])
        if not isinstance(image_data_list, list):
            image_data_list = [image_data_list]

        known_names, known_encodings = get_known_faces()
        if not known_names or not known_encodings:
            return jsonify({"status": "error", "message": "No registered users"}), 400

        # Decode all images
        images = []
        for img_data in image_data_list:
            img = decode_image(img_data)
            if img is not None:
                images.append(img)
        
        if not images:
            return jsonify({"status": "error", "message": "Invalid image. Please try again"}), 400

        # Anti-spoofing check: Detect if it's a photo/print
        if len(images) > 1:
            if detect_spoofing(images):
                return jsonify({"status": "error", "message": "Please use a live camera feed, not a photo"}), 400

        # Use the last (most recent) frame for face recognition
        image = images[-1]

        # Compute embedding for captured image
        try:
            unknown_embedding = compute_embedding(image, model_name=DEEPFACE_MODEL, enforce_detection=True)
        except Exception as e:
            # Convert technical errors to user-friendly messages
            friendly_msg = get_user_friendly_error(str(e))
            print(f"DeepFace error (converted to user-friendly): {e}")
            return jsonify({"status": "error", "message": friendly_msg}), 400

        if unknown_embedding is None or len(unknown_embedding) == 0:
            return jsonify({"status": "error", "message": "Could not process face. Try scanning in better lighting"}), 400

        # Compare against known embeddings using cosine similarity (optimized with numpy)
        unknown_embedding_norm = unknown_embedding / np.linalg.norm(unknown_embedding)
        sims = []
        for ke in known_encodings:
            try:
                # Check if embedding dimensions match
                if len(ke) != len(unknown_embedding):
                    print(f"Warning: Embedding dimension mismatch. Expected {len(unknown_embedding)}, got {len(ke)}")
                    continue  # Skip this embedding if dimensions don't match
                
                ke_norm = ke / np.linalg.norm(ke)
                sim = np.dot(unknown_embedding_norm, ke_norm)
                sims.append(sim)
            except Exception as e:
                print(f"Error comparing embeddings: {e}")
                continue  # Skip this embedding if comparison fails
        
        if not sims:
            return jsonify({"status": "error", "message": "Face data mismatch. Please re-register your face"}), 400
        
        sims = np.array(sims)
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim < COSINE_THRESHOLD:
            return jsonify({"status": "error", "message": "Face not recognized. Please scan in better lighting"}), 401

        name = known_names[best_idx]
        now = datetime.datetime.now()

        conn = get_db_connection()
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500

        try:
            last_type = get_last_log_type(name, conn)

            if last_type == 'LOGIN':
                new_type = 'LOGOUT'
                message = f"Goodbye, {name}!"
            else:
                new_type = 'LOGIN'
                message = f"Welcome, {name}!"

            c = conn.cursor()
            if DB_URL:
                c.execute("INSERT INTO logs (name, timestamp, type) VALUES (%s, %s, %s)", (name, now, new_type))
            else:
                c.execute("INSERT INTO logs (name, timestamp, type) VALUES (?, ?, ?)", (name, now, new_type))

            conn.commit()
        finally:
            if DB_URL:
                return_db_connection(conn)
            else:
                conn.close()

        return jsonify({"status": "success", "name": name, "type": new_type, "message": message, "similarity": float(best_sim)})

    except Exception as e:
        # Convert any technical errors to user-friendly messages
        friendly_msg = get_user_friendly_error(str(e))
        print(f"SCAN ERROR (technical): {e}")
        print(f"SCAN ERROR (user-friendly): {friendly_msg}")
        return jsonify({"status": "error", "message": friendly_msg}), 500

@app.route('/logs', methods=['GET'])
def logs():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT name, timestamp FROM logs ORDER BY id DESC LIMIT 10")
    rows = c.fetchall()
    if DB_URL:
        return_db_connection(conn)
    else:
        conn.close()

    results = []
    for row in rows:
        n = row[0] if DB_URL else row['name']
        t = row[1] if DB_URL else row['timestamp']
        results.append([n, t])

    return jsonify(results)

# Email sending (Brevo)
def send_custom_email(to_email, subject, html_content):
    url = "https://api.brevo.com/v3/smtp/email"
    api_key = os.environ.get('BREVO_API_KEY')
    admin_email = os.environ.get('ADMIN_EMAIL')

    if not api_key:
        print("No API Key found")
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
        requests.post(url, headers=headers, json=payload)
    except Exception as e:
        print(f"Email Failed: {e}")

# Monthly reports logic (unchanged except timezone usage)
def send_monthly_reports():
    now = datetime.datetime.now(_ist_timezone)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    conn = get_db_connection()
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
            c.execute("SELECT timestamp, type FROM logs WHERE name=%s AND timestamp >= %s ORDER BY timestamp ASC", (emp_name, start_of_month))
        else:
            c.execute("SELECT timestamp, type FROM logs WHERE name=? AND timestamp >= ? ORDER BY timestamp ASC", (emp_name, start_of_month))

        logs = c.fetchall()

        total_seconds = 0
        last_login = None

        for log in logs:
            l_time = log[0] if DB_URL else log['timestamp']
            l_type = log[1] if DB_URL else log['type']

            if isinstance(l_time, str):
                try:
                    l_time = datetime.datetime.strptime(l_time.split('.')[0], "%Y-%m-%d %H:%M:%S")
                except:
                    continue

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

@app.route('/run-monthly-reports')
def run_monthly_reports():
    secret = request.args.get('key')
    if secret != os.environ.get('CRON_SECRET', 'default-secret'):
        return "Unauthorized", 401

    result = send_monthly_reports()
    return result

# Scheduler setup
scheduler = None
_ist_timezone = pytz.timezone('Asia/Kolkata')

def setup_scheduler():
    global scheduler
    try:
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
    except Exception as e:
        print(f"⚠️ Scheduler setup failed: {e}")
        print("   You can still trigger reports manually via /run-monthly-reports?key=YOUR_SECRET")

    return scheduler

# Start scheduler only in main process to avoid duplicates
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or __name__ == '__main__':
    setup_scheduler()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
