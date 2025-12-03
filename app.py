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
        # For current DeepFace version we don't pass the built model
        # into represent(), but we keep this call to warm up weights.
        _deepface_model_instance = DeepFace.build_model(model_name)
        print("✅ DeepFace model preloaded")
    return _deepface_model_instance

def compute_embedding(img_array, model_name=DEEPFACE_MODEL, enforce_detection=True):
    """
    Compute a face embedding using DeepFace.represent.
    Uses numpy array input and avoids passing unsupported args like `model`.
    """
    from deepface import DeepFace
    # Optionally warm up model once at startup
    preload_deepface_model(model_name)
    rep = DeepFace.represent(
        img_path=img_array,
        model_name=model_name,
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


# ---------------- Attendance helpers ----------------
def get_last_log_type(name, conn=None):
    """
    Return the last log type (LOGIN / LOGOUT) for *today* for a given employee.
    This keeps the original behaviour: first scan of the day = LOGIN,
    second scan of the day = LOGOUT, and from the next day it resets.
    """
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    if conn is None:
        return None

    c = conn.cursor()
    try:
        if DB_URL:
            today = datetime.date.today()
            c.execute(
                "SELECT type FROM logs WHERE name = %s AND DATE(timestamp) = %s "
                "ORDER BY id DESC LIMIT 1",
                (name, today),
            )
        else:
            c.execute(
                "SELECT type FROM logs WHERE name = ? AND DATE(timestamp) = DATE('now') "
                "ORDER BY id DESC LIMIT 1",
                (name,),
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

    # For sqlite we use row_factory=Row so we can index by column name
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
    """Return all client packages for the admin client table."""
    conn = get_db_connection()
    if conn is None:
        return jsonify([])  # fail-soft
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
            cid = row["id"]
            name = row["name"]
            start_date = row["start_date"]
            end_date = row["end_date"]
            cost = row["cost"]

        # Keep dates formatted as DD-MM-YYYY to match existing frontend logic
        try:
            if start_date:
                start_date = datetime.datetime.strptime(
                    start_date, "%Y-%m-%d"
                ).strftime("%d-%m-%Y")
            if end_date:
                end_date = datetime.datetime.strptime(
                    end_date, "%Y-%m-%d"
                ).strftime("%d-%m-%Y")
        except Exception:
            # If already in desired format, just keep as-is
            pass

        clients.append(
            {
                "id": cid,
                "name": name,
                "start_date": start_date,
                "end_date": end_date,
                "cost": cost,
            }
        )

    return jsonify(clients)


@app.route('/api/clients/add', methods=['POST'])
def add_client():
    """Add a new client package from the admin panel."""
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
        else:
            c.execute("INSERT INTO employees (name, email, encoding) VALUES (?, ?, ?)",
                      (name, email, sqlite3.Binary(embedding_bytes)))
        # Ensure data is actually written to the DB
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
        if conn is None:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        c = conn.cursor()

        # Determine today's last log to keep the original behaviour:
        # first scan of the day = LOGIN, second = LOGOUT, later scans show "already marked".
        last_type = get_last_log_type(name, conn)
        if last_type == "LOGIN":
            new_type = "LOGOUT"
            message = f"Goodbye, {name}!"
        elif last_type == "LOGOUT":
            # Already logged in and out today – don't create more rows
            if DB_URL:
                return_db_connection(conn)
            else:
                conn.close()
            return jsonify(
                {
                    "status": "error",
                    "message": "Today's attendance is already marked for this employee",
                }
            ), 400
        else:
            new_type = "LOGIN"
            message = f"Welcome, {name}!"

        if DB_URL:
            c.execute(
                "INSERT INTO logs (name, timestamp, type) VALUES (%s, %s, %s)",
                (name, now, new_type),
            )
            conn.commit()
            return_db_connection(conn)
        else:
            c.execute(
                "INSERT INTO logs (name, timestamp, type) VALUES (?, ?, ?)",
                (name, now, new_type),
            )
            conn.commit()
            conn.close()

        return jsonify({"status": "success", "name": name, "type": new_type, "message": message, "similarity": float(best_sim)})
    except Exception as e:
        print(f"SCAN ERROR: {e}")
        return jsonify({"status": "error", "message": "Face recognition temporarily unavailable"}), 500


@app.route('/api/report', methods=['GET'])
def report():
    """
    Return raw log entries for the admin attendance table.
    Each row: { name, date (DD-MM-YYYY), time (HH:MM), type }.
    """
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

    data = []
    for row in rows:
        n = row[0] if DB_URL else row["name"]
        t = row[1] if DB_URL else row["timestamp"]
        s = row[2] if DB_URL else row["type"]

        if isinstance(t, str):
            try:
                # strip microseconds if present
                t_obj = datetime.datetime.strptime(t.split(".")[0], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
        else:
            t_obj = t

        data.append(
            {
                "name": n,
                "date": t_obj.strftime("%d-%m-%Y"),
                "time": t_obj.strftime("%H:%M"),
                "type": s,
            }
        )

    return jsonify(data)

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
