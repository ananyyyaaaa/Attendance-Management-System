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
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

load_dotenv()
app = Flask(__name__)
CORS(app)

DB_URL = os.environ.get('DATABASE_URL')
DEEPFACE_MODEL = os.environ.get("DEEPFACE_MODEL", "VGG-Face")
COSINE_THRESHOLD = float(os.environ.get("DEEPFACE_THRESHOLD", 0.60))

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
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id SERIAL PRIMARY KEY, name TEXT, email TEXT, encoding BYTEA)''')
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id SERIAL PRIMARY KEY, name TEXT, timestamp TIMESTAMP, type TEXT)''')
        try:
            c.execute('''CREATE INDEX IF NOT EXISTS idx_logs_name_date ON logs(name, DATE(timestamp))''')
        except: pass
    else:
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, encoding BLOB)''')
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, timestamp DATETIME, type TEXT)''')
        try:
            c.execute('''CREATE INDEX IF NOT EXISTS idx_logs_name_date ON logs(name, DATE(timestamp))''')
        except: pass

    if DB_URL:
        c.execute('''CREATE TABLE IF NOT EXISTS clients
                     (id SERIAL PRIMARY KEY, name TEXT, start_date TEXT, end_date TEXT, cost TEXT)''')
    else:
        c.execute('''CREATE TABLE IF NOT EXISTS clients
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, start_date TEXT, end_date TEXT, cost TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ---------------- Helpers ----------------
def decode_image(base64_string):
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split(",")[1]
        img_data = base64.b64decode(base64_string)
        pil_image = PIL.Image.open(io.BytesIO(img_data))
        rgb_image = pil_image.convert("RGB")
        width, height = rgb_image.size
        if width > 250:  # smaller for CPU speed
            ratio = 250 / width
            new_height = int(height * ratio)
            rgb_image = rgb_image.resize((250, new_height), PIL.Image.Resampling.LANCZOS)
        return np.array(rgb_image)
    except Exception as e:
        print(f"Image Decode Error: {e}")
        return None

# ---------------- Preload DeepFace Model ----------------
_deepface_model_instance = None
def preload_deepface_model(model_name=DEEPFACE_MODEL):
    global _deepface_model_instance
    if _deepface_model_instance is None:
        from deepface import DeepFace
        print(f"Loading DeepFace model '{model_name}'...")
        _deepface_model_instance = DeepFace.build_model(model_name)
        print("✅ DeepFace model loaded")
    return _deepface_model_instance

def compute_embedding(img_array, model_name=DEEPFACE_MODEL, enforce_detection=True):
    from deepface import DeepFace
    model = preload_deepface_model(model_name)
    rep = DeepFace.represent(
        img_path=img_array,
        model_name=model_name,
        model=model,
        enforce_detection=enforce_detection,
        detector_backend='opencv'
    )
    # flatten embedding
    if isinstance(rep, dict) and 'embedding' in rep:
        vec = np.array(rep['embedding'], dtype=np.float32)
    elif isinstance(rep, list) and len(rep) > 0:
        if isinstance(rep[0], dict) and 'embedding' in rep[0]:
            vec = np.array(rep[0]['embedding'], dtype=np.float32)
        else:
            vec = np.array(rep[0], dtype=np.float32)
    else:
        vec = np.array(rep, dtype=np.float32)
    if vec.ndim > 1:
        vec = vec.flatten()
    return vec

_face_cache = None
_cache_timestamp = None
CACHE_DURATION = 300  # 5 min cache

def get_known_faces():
    global _face_cache, _cache_timestamp
    now = datetime.datetime.now()
    if _face_cache and _cache_timestamp and (now - _cache_timestamp).total_seconds() < CACHE_DURATION:
        return _face_cache
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM employees")
    rows = c.fetchall()
    if DB_URL:
        return_db_connection(conn)
    else:
        conn.close()
    known_names, known_encodings = [], []
    for row in rows:
        name = row[0] if DB_URL else row['name']
        encoding_bytes = row[1] if DB_URL else row['encoding']
        if encoding_bytes:
            try:
                arr = np.frombuffer(encoding_bytes, dtype=np.float32)
                known_names.append(name)
                known_encodings.append(arr)
            except Exception as e:
                print(f"Failed to decode {name}: {e}")
    _face_cache = (known_names, known_encodings)
    _cache_timestamp = now
    return known_names, known_encodings

def invalidate_face_cache():
    global _face_cache, _cache_timestamp
    _face_cache = None
    _cache_timestamp = None

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- Routes ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/register', methods=['POST'])
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
        embedding = compute_embedding(image)
        embedding_bytes = embedding.astype(np.float32).tobytes()
        conn = get_db_connection()
        c = conn.cursor()
        if DB_URL:
            c.execute("INSERT INTO employees (name, email, encoding) VALUES (%s, %s, %s)",
                      (name, email, psycopg2.Binary(embedding_bytes)))
            return_db_connection(conn)
        else:
            c.execute("INSERT INTO employees (name, email, encoding) VALUES (?, ?, ?)",
                      (name, email, sqlite3.Binary(embedding_bytes)))
            conn.close()
        invalidate_face_cache()
        return jsonify({"status": "success", "message": f"Registered {name}!"})
    except Exception as e:
        print(f"REGISTER ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/scan', methods=['POST'])
def scan():
    try:
        data = request.json
        image_data_list = data.get('images', [data.get('image')])
        if not isinstance(image_data_list, list):
            image_data_list = [image_data_list]
        known_names, known_encodings = get_known_faces()
        if not known_names or not known_encodings:
            return jsonify({"status": "error", "message": "No registered users"}), 400
        # decode last image
        image = decode_image(image_data_list[-1])
        if image is None:
            return jsonify({"status": "error", "message": "Invalid image"}), 400
        unknown_embedding = compute_embedding(image)
        unknown_embedding /= np.linalg.norm(unknown_embedding)
        sims = []
        for ke in known_encodings:
            if len(ke) != len(unknown_embedding):
                continue
            ke_norm = ke / np.linalg.norm(ke)
            sims.append(np.dot(unknown_embedding, ke_norm))
        if not sims:
            return jsonify({"status": "error", "message": "Face data mismatch"}), 400
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        if best_sim < COSINE_THRESHOLD:
            return jsonify({"status": "error", "message": "Face not recognized"}), 401
        name = known_names[best_idx]
        now = datetime.datetime.now()
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT type FROM logs WHERE name=? ORDER BY id DESC LIMIT 1" if not DB_URL else
                  "SELECT type FROM logs WHERE name=%s ORDER BY id DESC LIMIT 1", (name,))
        row = c.fetchone()
        last_type = row[0] if row else None
        new_type = 'LOGOUT' if last_type == 'LOGIN' else 'LOGIN'
        message = f"Goodbye, {name}!" if new_type == 'LOGOUT' else f"Welcome, {name}!"
        if DB_URL:
            c.execute("INSERT INTO logs (name, timestamp, type) VALUES (%s, %s, %s)", (name, now, new_type))
            return_db_connection(conn)
        else:
            c.execute("INSERT INTO logs (name, timestamp, type) VALUES (?, ?, ?)", (name, now, new_type))
            conn.commit()
            conn.close()
        return jsonify({"status": "success", "name": name, "type": new_type, "message": message, "similarity": float(best_sim)})
    except Exception as e:
        print(f"SCAN ERROR: {e}")
        return jsonify({"status": "error", "message": "Face recognition temporarily unavailable"}), 500

# ---------------- Scheduler ----------------
scheduler = None
_ist_timezone = pytz.timezone('Asia/Kolkata')
def setup_scheduler():
    global scheduler
    scheduler = BackgroundScheduler(timezone=_ist_timezone)
    scheduler.add_job(
        func=lambda: print("Monthly report placeholder"),
        trigger=CronTrigger(day=25, hour=17, minute=0),
        id='monthly_reports',
        replace_existing=True
    )
    scheduler.start()
    print("✅ Scheduler started")
setup_scheduler()

# ---------------- Run App ----------------
if __name__ == '__main__':
    preload_deepface_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
