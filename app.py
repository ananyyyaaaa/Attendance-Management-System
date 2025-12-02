import os
import datetime
import numpy as np
import face_recognition
import base64
import io
import sqlite3
import psycopg2
import PIL.Image  # <--- Essential fix for image errors
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

DB_URL = os.environ.get('DATABASE_URL')

def get_db_connection():
    if DB_URL:
        # Cloud / Postgres
        try:
            if 'localhost' in DB_URL:
                conn = psycopg2.connect(DB_URL)
            else:
                conn = psycopg2.connect(DB_URL, sslmode='require')
            return conn
        except Exception as e:
            print(f"Postgres Connection Failed: {e}")
            return None
    else:
        # Local / SQLite
        conn = sqlite3.connect('attendance.db')
        conn.row_factory = sqlite3.Row
        return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    if DB_URL:
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id SERIAL PRIMARY KEY, name TEXT, encoding BYTEA)''')
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id SERIAL PRIMARY KEY, name TEXT, timestamp TIMESTAMP, type TEXT)''')
    else:
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, encoding BLOB)''')
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, timestamp DATETIME, type TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- HELPER: Check Status (Login vs Logout) ---
def get_last_log_type(name):
    """
    Fetches the last log for the user.
    Returns 'LOGIN' if they are currently clocked in today.
    Returns None if they are clocked out or haven't scanned today.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Get the very last log for this person (regardless of date)
    if DB_URL:
        c.execute("SELECT type, timestamp FROM logs WHERE name = %s ORDER BY id DESC LIMIT 1", (name,))
    else:
        c.execute("SELECT type, timestamp FROM logs WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
        
    row = c.fetchone()
    conn.close()
    
    if row:
        last_type = row[0] if DB_URL else row['type']
        last_ts = row[1] if DB_URL else row['timestamp']
        
        # 2. Parse the timestamp to check if it matches TODAY
        # SQLite stores dates as Strings, Postgres as Objects
        if isinstance(last_ts, str):
            # Handle format "2023-10-27 10:00:00.123456" or "2023-10-27 10:00:00"
            try:
                # Truncate milliseconds for safety if present
                ts_str = last_ts.split('.')[0]
                last_date = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").date()
            except ValueError:
                # Fallback for unexpected formats
                return None 
        else:
            # Postgres datetime object
            last_date = last_ts.date()
            
        today = datetime.date.today()
        
        # 3. If the last log was TODAY, return its type
        if last_date == today:
            return last_type
            
    return None

# --- THE FIX: Robust Image Decoding ---
def decode_image(base64_string):
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        img_data = base64.b64decode(base64_string)
        
        # Open image with PIL
        pil_image = PIL.Image.open(io.BytesIO(img_data))
        
        # Convert to RGB (Removes Transparency/Alpha channel that causes crashes)
        rgb_image = pil_image.convert("RGB")
        
        # Convert to Numpy Array
        return np.array(rgb_image)
    except Exception as e:
        print(f"Image Decode Error: {e}")
        return None

def get_known_faces():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM employees")
    rows = c.fetchall()
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
            
        known_names.append(name)
        known_encodings.append(np.frombuffer(encoding_bytes, dtype=np.float64))
        
    return known_names, known_encodings

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/admin')
def admin_page():
    return render_template('admin.html')

@app.route('/api/report')
def report():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT name, timestamp, type FROM logs ORDER BY timestamp DESC")
    rows = c.fetchall()
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

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        name = data.get('name')
        image_data = data.get('image')

        if not name or not image_data:
            return jsonify({"status": "error", "message": "Missing Data"}), 400

        image = decode_image(image_data)
        if image is None:
             return jsonify({"status": "error", "message": "Invalid Image Format"}), 400

        encodings = face_recognition.face_encodings(image)

        if not encodings:
            return jsonify({"status": "error", "message": "No face detected"}), 400

        encoding_bytes = encodings[0].tobytes()
        
        conn = get_db_connection()
        c = conn.cursor()
        
        if DB_URL:
            c.execute("INSERT INTO employees (name, encoding) VALUES (%s, %s)", (name, encoding_bytes))
        else:
            # SQLite needs explicit Binary wrapper
            c.execute("INSERT INTO employees (name, encoding) VALUES (?, ?)", (name, sqlite3.Binary(encoding_bytes)))
            
        conn.commit()
        conn.close()

        return jsonify({"status": "success", "message": f"Registered {name}!"})

    except Exception as e:
        print(f"REGISTER ERROR: {e}") # Print error to terminal
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/scan', methods=['POST'])
def scan():
    try:
        data = request.json
        image_data = data.get('image')

        known_names, known_encodings = get_known_faces()
        if not known_names:
            return jsonify({"status": "error", "message": "No registered users"}), 400

        image = decode_image(image_data)
        if image is None:
             return jsonify({"status": "error", "message": "Invalid Image Format"}), 400

        unknown_encodings = face_recognition.face_encodings(image)

        if not unknown_encodings:
            return jsonify({"status": "error", "message": "No face visible"}), 400

        matches = face_recognition.compare_faces(known_encodings, unknown_encodings[0], tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, unknown_encodings[0])

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_names[best_match_index]
            now = datetime.datetime.now()
            
            # --- FIXED TOGGLE LOGIC ---
            last_type = get_last_log_type(name)
            
            if last_type == 'LOGIN':
                new_type = 'LOGOUT'
                message = f"Goodbye, {name}!"
            else:
                new_type = 'LOGIN'
                message = f"Welcome, {name}!"
            
            conn = get_db_connection()
            c = conn.cursor()
            
            if DB_URL:
                c.execute("INSERT INTO logs (name, timestamp, type) VALUES (%s, %s, %s)", (name, now, new_type))
            else:
                c.execute("INSERT INTO logs (name, timestamp, type) VALUES (?, ?, ?)", (name, now, new_type))
                
            conn.commit()
            conn.close()

            return jsonify({"status": "success", "name": name, "type": new_type, "message": message})
        else:
            return jsonify({"status": "error", "message": "Unknown Person"}), 401

    except Exception as e:
        print(f"SCAN ERROR: {e}") # Print error to terminal
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/logs', methods=['GET'])
def logs():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT name, timestamp FROM logs ORDER BY id DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        n = row[0] if DB_URL else row['name']
        t = row[1] if DB_URL else row['timestamp']
        results.append([n, t])
        
    return jsonify(results)

if __name__ == '__main__':
    # Debug=True helps show errors in the browser
    app.run(host='0.0.0.0', port=5000, debug=True)