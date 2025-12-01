import face_recognition
import numpy as np
import sqlite3
import os

print("--- DIAGNOSIS STARTED ---")

# 1. Check Database File
if os.path.exists("attendance.db"):
    print("✅ Database file found.")
else:
    print("❌ Database file MISSING.")
    # Attempt to create it manually
    try:
        conn = sqlite3.connect('attendance.db')
        conn.close()
        print("   -> Created database file manually.")
    except Exception as e:
        print(f"   -> Failed to create DB: {e}")

# 2. Check AI Library
try:
    print("⏳ Testing Face Recognition library (this might take a moment)...")
    # Create a dummy image (100x100 pixels, black)
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Try to find faces in the black image (Should find 0, but not crash)
    encodings = face_recognition.face_encodings(dummy_image)
    print("✅ Face Recognition is WORKING! (Library is safe)")
except Exception as e:
    print(f"❌ Face Recognition CRASHED.")
    print(f"   Error details: {e}")
    print("   SOLUTION: This usually means 'face_recognition_models' is missing.")

print("--- DIAGNOSIS FINISHED ---")
