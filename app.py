import os
import datetime
import numpy as np
import face_recognition
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

# Connection pool for Postgres (reduces connection overhead)
_postgres_pool = None

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
        # Cloud / Postgres - use connection pool
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
            # Fallback to direct connection
            try:
                if 'localhost' in DB_URL:
                    return psycopg2.connect(DB_URL)
                else:
                    return psycopg2.connect(DB_URL, sslmode='require')
            except Exception as e:
                print(f"Postgres Connection Failed: {e}")
                return None
    else:
        # Local / SQLite
        conn = sqlite3.connect('attendance.db', check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

def return_db_connection(conn):
    """Return connection to pool (for Postgres)"""
    if DB_URL and _postgres_pool and conn:
        try:
            _postgres_pool.putconn(conn)
        except:
            pass  # Ignore errors when returning to pool

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
        # Create index for faster queries on name + date
        try:
            c.execute('''CREATE INDEX IF NOT EXISTS idx_logs_name_date ON logs(name, DATE(timestamp))''')
        except:
            pass  # Index might already exist
    else:
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, encoding BLOB)''')
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, timestamp DATETIME, type TEXT)''')
        # Create index for faster queries
        try:
            c.execute('''CREATE INDEX IF NOT EXISTS idx_logs_name_date ON logs(name, DATE(timestamp))''')
        except:
            pass  # Index might already exist
    
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
    """
    Fetches the last log for the user.
    Returns 'LOGIN' if they are currently clocked in today.
    Returns None if they are clocked out or haven't scanned today.
    Optimized: Can reuse existing connection to avoid extra DB calls.
    """
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True
    
    if conn is None:
        return None
    
    c = conn.cursor()
    
    # Optimized: Check for today's date directly in SQL to avoid Python date parsing
    # Use index-friendly query (name + date filter)
    if DB_URL:
        # Postgres: Use DATE() function for efficient date comparison
        today = datetime.date.today()
        c.execute("SELECT type FROM logs WHERE name = %s AND DATE(timestamp) = %s ORDER BY id DESC LIMIT 1", 
                  (name, today))
    else:
        # SQLite: Use DATE() function - faster than Python parsing
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
        
        # Open image with PIL
        pil_image = PIL.Image.open(io.BytesIO(img_data))
        
        # Convert to RGB (Removes Transparency/Alpha channel that causes crashes)
        rgb_image = pil_image.convert("RGB")
        
        # Resize if too large for faster processing (max 800px width)
        width, height = rgb_image.size
        if width > 800:
            ratio = 800 / width
            new_height = int(height * ratio)
            rgb_image = rgb_image.resize((800, new_height), PIL.Image.Resampling.LANCZOS)
        
        # Convert to Numpy Array
        return np.array(rgb_image)
    except Exception as e:
        print(f"Image Decode Error: {e}")
        return None

# Cache for face encodings to improve performance
_face_cache = None
_cache_timestamp = None
CACHE_DURATION = 300  # Cache for 5 minutes

def get_known_faces():
    global _face_cache, _cache_timestamp
    
    # Check if cache is valid
    now = datetime.datetime.now()
    if _face_cache is not None and _cache_timestamp is not None:
        if (now - _cache_timestamp).total_seconds() < CACHE_DURATION:
            return _face_cache
    
    # Load from database
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
            
        if encoding_bytes:  # Skip None encodings
            known_names.append(name)
            known_encodings.append(np.frombuffer(encoding_bytes, dtype=np.float64))
    
    # Update cache
    _face_cache = (known_names, known_encodings)
    _cache_timestamp = now
        
    return known_names, known_encodings

def invalidate_face_cache():
    """Call this when a new employee is registered to refresh the cache"""
    global _face_cache, _cache_timestamp
    _face_cache = None
    _cache_timestamp = None

# --- ROUTES ---

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
        # 1. Extract raw data first
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

        # 2. CHANGE THE FORMAT HERE
        # Convert YYYY-MM-DD (Database) -> DD-MM-YYYY (Display)
        try:
            if start_date:
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime("%d-%m-%Y")
            if end_date:
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%Y")
        except ValueError:
            pass # If data is empty or corrupted, keep original value

        # 3. Add to list
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

        encodings = face_recognition.face_encodings(image)

        if not encodings:
            return jsonify({"status": "error", "message": "No face detected"}), 400

        encoding_bytes = encodings[0].tobytes()
        
        conn = get_db_connection()
        c = conn.cursor()
        
        if DB_URL:
            c.execute("INSERT INTO employees (name, email, encoding) VALUES (%s, %s, %s)", (name, email, encoding_bytes))
        else:
            # SQLite needs explicit Binary wrapper
            c.execute("INSERT INTO employees (name, email, encoding) VALUES (?, ?, ?)", (name, email, sqlite3.Binary(encoding_bytes)))
            
        conn.commit()
        if DB_URL:
            return_db_connection(conn)
        else:
            conn.close()
        
        # Invalidate cache so new employee is included in next scan
        invalidate_face_cache()

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

        # Optimize face detection: use HOG model (faster) with minimal upsampling
        face_locations = face_recognition.face_locations(image, model="hog", number_of_times_to_upsample=0)

        # Only encode if face is found
        if not face_locations:
            return jsonify({"status": "error", "message": "No face visible"}), 400

        # Fast encoding: minimal jitters for speed
        unknown_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations, num_jitters=1)

        if not unknown_encodings:
            return jsonify({"status": "error", "message": "Face encoding failed"}), 400

        # Fast comparison with optimized tolerance
        matches = face_recognition.compare_faces(known_encodings, unknown_encodings[0], tolerance=0.6)
        
        # Only calculate distances if we have matches (optimization)
        if True not in matches:
            return jsonify({"status": "error", "message": "Unknown Person"}), 401
            
        face_distances = face_recognition.face_distance(known_encodings, unknown_encodings[0])

        best_match_index = np.argmin(face_distances)

        # Double-check match (already verified above, but keep for safety)
        if matches[best_match_index]:
            name = known_names[best_match_index]
            # Use UTC now() - faster than timezone-aware datetime
            now = datetime.datetime.now()
            
            # --- OPTIMIZED: Single DB connection for both operations ---
            conn = get_db_connection()
            if conn is None:
                return jsonify({"status": "error", "message": "Database connection failed"}), 500
            
            try:
                # Get last log type using same connection
                last_type = get_last_log_type(name, conn)
                
                if last_type == 'LOGIN':
                    new_type = 'LOGOUT'
                    message = f"Goodbye, {name}!"
                else:
                    new_type = 'LOGIN'
                    message = f"Welcome, {name}!"
                
                # Insert log using same connection
                c = conn.cursor()
                if DB_URL:
                    c.execute("INSERT INTO logs (name, timestamp, type) VALUES (%s, %s, %s)", (name, now, new_type))
                else:
                    c.execute("INSERT INTO logs (name, timestamp, type) VALUES (?, ?, ?)", (name, now, new_type))
                    
                conn.commit()
            finally:
                # Always return connection to pool
                if DB_URL:
                    return_db_connection(conn)
                else:
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

def send_monthly_reports():
    """Internal function to send monthly reports - can be called by cron or scheduler"""
    # 2. Get Date Range (use cached timezone)
    now = datetime.datetime.now(_ist_timezone)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    conn = get_db_connection()
    c = conn.cursor()

    # 3. Get employees with emails
    c.execute("SELECT name, email FROM employees WHERE email IS NOT NULL")
    employees = c.fetchall()

    emails_sent = 0

    for emp in employees:
        # Handle SQLite vs Postgres
        emp_name = emp[0] if DB_URL else emp['name']
        emp_email = emp[1] if DB_URL else emp['email']
        
        if not emp_email: continue

        # 4. Get Logs
        if DB_URL:
            c.execute("SELECT timestamp, type FROM logs WHERE name=%s AND timestamp >= %s ORDER BY timestamp ASC", (emp_name, start_of_month))
        else:
            c.execute("SELECT timestamp, type FROM logs WHERE name=? AND timestamp >= ? ORDER BY timestamp ASC", (emp_name, start_of_month))
        
        logs = c.fetchall()

        # 5. Calculate Hours
        total_seconds = 0
        last_login = None

        for log in logs:
            l_time = log[0] if DB_URL else log['timestamp']
            l_type = log[1] if DB_URL else log['type']

            if isinstance(l_time, str):
                try:
                    l_time = datetime.datetime.strptime(l_time.split('.')[0], "%Y-%m-%d %H:%M:%S")
                except: continue

            if l_type == 'LOGIN':
                last_login = l_time
            elif l_type == 'LOGOUT' and last_login:
                duration = (l_time - last_login).total_seconds()
                total_seconds += duration
                last_login = None 
        
        total_hours = round(total_seconds / 3600, 2)

        # 6. Send Email
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
    # 1. Check Secret Key (for manual trigger or external cron)
    secret = request.args.get('key')
    if secret != os.environ.get('CRON_SECRET', 'default-secret'):
        return "Unauthorized", 401
    
    # Call the internal function
    result = send_monthly_reports()
    return result

# --- HELPER FUNCTION (Must be outside the route) ---

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
        # This requires 'import requests' at the top of the file
        requests.post(url, headers=headers, json=payload)
    except Exception as e:
        print(f"Email Failed: {e}")



# Setup scheduler for monthly emails on 25th at 5pm IST
scheduler = None
# Cache timezone object to avoid repeated creation
_ist_timezone = pytz.timezone('Asia/Kolkata')

def setup_scheduler():
    global scheduler
    try:
        scheduler = BackgroundScheduler(timezone=_ist_timezone)
        
        # Schedule monthly reports on 25th of every month at 5:00 PM IST
        scheduler.add_job(
            func=send_monthly_reports,
            trigger=CronTrigger(day=25, hour=17, minute=0),  # 25th day, 5pm (17:00)
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

# Initialize scheduler when module loads (works for both dev and production)
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or __name__ == '__main__':
    # Only start scheduler in main process (avoid duplicate schedulers in reloader)
    setup_scheduler()

if __name__ == '__main__':
    # Debug=True helps show errors in the browser
    app.run(host='0.0.0.0', port=5000, debug=True)