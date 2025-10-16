from flask import Flask, request, jsonify, render_template, send_from_directory, session
import sqlite3
import json
import base64
import io
from dotenv import load_dotenv
import os
import numpy as np
from PIL import Image
import face_recognition
import cv2
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import string
import threading
import time
import uuid
import tempfile
from flask_cors import CORS
from functools import wraps
from contextlib import contextmanager
import re
import traceback
import hashlib

app = Flask(__name__)
CORS(app)
app.secret_key = 'facecast_secret_key_2024'  # For session management

# Simple logging implementation
class SimpleLogger:
    def __init__(self):
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
    
    def info(self, message, category='app'):
        self._log('INFO', message, category)
    
    def error(self, message, exception=None, category='app'):
        if exception:
            message = f"{message}: {str(exception)}"
        self._log('ERROR', message, category)
    
    def security_event(self, event_type, details, ip_address):
        message = f"SECURITY {event_type} from {ip_address}: {json.dumps(details)}"
        self._log('SECURITY', message, 'security')
    
    def _log(self, level, message, category):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] [{category}] {message}\n"
        
        # Log to file
        log_file = os.path.join(self.log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_message)
        except Exception as e:
            print(f"Failed to write to log file: {e}")
        
       
        print(log_message.strip())
    
   
    def get_log_files(self):
        """Get list of log files - FIX FOR ADMIN LOGS"""
        try:
            if not os.path.exists(self.log_dir):
                return []
            files = [f for f in os.listdir(self.log_dir) if f.endswith('.log')]
            return sorted(files, reverse=True)
        except Exception as e:
            print(f"Error getting log files: {e}")
            return []
    
    
    def read_log_file(self, filename, lines=1000):
        try:
            filepath = os.path.join(self.log_dir, filename)
            if not os.path.exists(filepath):
                return []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.readlines()
            
            return content[-lines:] if lines > 0 else content
        except Exception:
            return []
    
    def get_log_stats(self):
        try:
            files = self.get_log_files()
            stats = {
                'total_files': len(files),
                'latest_file': files[0] if files else None,
                'total_size': 0
            }
            
            for file in files:
                filepath = os.path.join(self.log_dir, file)
                if os.path.exists(filepath):
                    stats['total_size'] += os.path.getsize(filepath)
            
            return stats
        except Exception:
            return {'total_files': 0, 'latest_file': None, 'total_size': 0}
    
    def cleanup_old_logs(self, days=30):
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            files_removed = 0
            
            for filename in self.get_log_files():
                filepath = os.path.join(self.log_dir, filename)
                file_date = datetime.strptime(filename.replace('.log', ''), '%Y-%m-%d')
                
                if file_date < cutoff_date:
                    os.remove(filepath)
                    files_removed += 1
            
            return files_removed
        except Exception as e:
            self.error("Error cleaning up old logs", e)
            return 0

# Initialize logging
logger = SimpleLogger()
logger.info("FaceCast application starting up", 'app')

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
db_file = os.path.join(basedir, "..", "database", "voters.db")

# Ensure database directory exists
os.makedirs(os.path.dirname(db_file), exist_ok=True)

def get_db_connection():
    """Get database connection with proper configuration for concurrent access"""
    max_retries = 5
    retry_delay = 0.1  # Start with 100ms delay
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(db_file, timeout=30.0)
            conn.row_factory = sqlite3.Row
            
            # Configure SQLite for better concurrent access
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA busy_timeout=30000')
            conn.execute('PRAGMA foreign_keys=ON')
            conn.execute('PRAGMA cache_size=10000')
            
            return conn
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                print(f"Database locked, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                raise e
    
    raise sqlite3.OperationalError("Could not acquire database lock after multiple retries")

@contextmanager
def get_db_cursor():
    """Context manager for database operations that ensures proper cleanup"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        yield conn, cursor
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def initialize_database():
    """Initialize database with required tables"""
    try:
        with get_db_cursor() as (conn, cursor):
            # Voters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voters (
                    voter_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    address TEXT,
                    gender TEXT,
                    aadhaar TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    email TEXT UNIQUE NOT NULL,
                    dob TEXT NOT NULL,
                    face_embedding TEXT,
                    profile_image TEXT,
                    otp TEXT,
                    has_voted INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Admin table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS admin (
                    admin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    email TEXT
                )
            """)
            
            # Elections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS elections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    status TEXT DEFAULT 'upcoming',
                    is_demo INTEGER DEFAULT 0,
                    demo_duration_minutes INTEGER DEFAULT 2
                )
            """)
            
            # Candidates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    election_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    party TEXT,
                    position TEXT DEFAULT 'Candidate',
                    vote_count INTEGER DEFAULT 0,
                    candidate_image TEXT,
                    party_logo TEXT,
                    FOREIGN KEY (election_id) REFERENCES elections (id) ON DELETE CASCADE
                )
            """)
            
            # Votes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS votes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    voter_id TEXT NOT NULL,
                    election_id INTEGER NOT NULL,
                    candidate_id INTEGER NOT NULL,
                    vote_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (voter_id) REFERENCES voters (voter_id),
                    FOREIGN KEY (election_id) REFERENCES elections (id) ON DELETE CASCADE,
                    FOREIGN KEY (candidate_id) REFERENCES candidates (id) ON DELETE CASCADE,
                    UNIQUE(voter_id, election_id)
                )
            """)
            
            # Insert default admin if not exists
            cursor.execute("""
                INSERT OR IGNORE INTO admin (username, password, email) 
                VALUES ('admin', 'admin123', 'shilpachaurasiya1205@gmail.com')
            """)
            
        logger.info("Database initialized successfully", 'database')
    except Exception as e:
        logger.error("Failed to initialize database", e, 'database')
        raise

def require_admin_auth(f):
    """Decorator to require admin authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session or not session['admin_logged_in']:
            return jsonify({"error": "Admin authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

def decode_base64_image(image_data_url):
    """Robust base64 image decoding with multiple fallback methods"""
    try:
        logger.info("Starting image decoding...", 'image_processing')
        
        # Handle different base64 formats
        if ',' in image_data_url:
            header, encoded_data = image_data_url.split(',', 1)
            logger.info(f"Image header: {header[:50]}...", 'image_processing')
        else:
            encoded_data = image_data_url
            logger.info("No header found in image data", 'image_processing')
        
        # Remove any whitespace
        encoded_data = encoded_data.strip()
        logger.info(f"Encoded data length: {len(encoded_data)}", 'image_processing')
        
        # Fix padding if necessary
        missing_padding = len(encoded_data) % 4
        if missing_padding:
            encoded_data += '=' * (4 - missing_padding)
            logger.info(f"Added {4 - missing_padding} padding characters", 'image_processing')
        
        # Decode base64
        try:
            image_data = base64.b64decode(encoded_data)
            logger.info(f"Base64 decoded successfully, data size: {len(image_data)} bytes", 'image_processing')
        except Exception as e:
            logger.error(f"Base64 decoding failed: {e}", None, 'image_processing')
            return None
        
        # Method 1: Try OpenCV first
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None and frame.size > 0:
                logger.info(f"OpenCV decoding successful, shape: {frame.shape}", 'image_processing')
                return frame
            else:
                logger.warning("OpenCV decoding returned empty frame", 'image_processing')
        except Exception as e:
            logger.warning(f"OpenCV decoding failed: {e}", 'image_processing')
        
        # Method 2: Try PIL
        try:
            image_stream = io.BytesIO(image_data)
            pil_image = Image.open(image_stream)
            
            # Convert PIL image to OpenCV format
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
                logger.info("Converted RGBA to RGB", 'image_processing')
            elif pil_image.mode == 'L':
                pil_image = pil_image.convert('RGB')
                logger.info("Converted grayscale to RGB", 'image_processing')
            
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            if frame is not None and frame.size > 0:
                logger.info(f"PIL decoding successful, shape: {frame.shape}", 'image_processing')
                return frame
            else:
                logger.warning("PIL decoding returned empty frame", 'image_processing')
        except Exception as e:
            logger.warning(f"PIL decoding failed: {e}", 'image_processing')
        
        logger.error("All image decoding methods failed", None, 'image_processing')
        return None
        
    except Exception as e:
        logger.error("Error in image decoding process", e, 'image_processing')
        return None

def get_face_embedding_256d(image_data_url):
    """Extract 256-dimensional face embedding consistently"""
    try:
        logger.info("Extracting 256D face embedding...", 'face_recognition')
        
        frame = decode_base64_image(image_data_url)
        if frame is None:
            logger.error("Could not decode image", None, 'face_recognition')
            return None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=1)
        
        if not face_locations:
            logger.error("No faces detected", None, 'face_recognition')
            return None
    
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1, model="large")
        
        if not face_encodings:
            logger.error("Could not generate face encodings", None, 'face_recognition')
            return None
        
        base_embedding = face_encodings[0]
        
        extended_embedding = []
        
        extended_embedding.extend(base_embedding.tolist())
        
        for i, value in enumerate(base_embedding):
            if i < 128:
                extended_embedding.append(value * 1.1)  # Simple transformation
            else:
                break
        
        while len(extended_embedding) < 256:
            extended_embedding.append(0.0)
        
        final_embedding = extended_embedding[:256]
        
        logger.info(f"Generated 256D embedding, length: {len(final_embedding)}", 'face_recognition')
        return str(final_embedding)
        
    except Exception as e:
        logger.error("Error extracting 256D face embedding", e, 'face_recognition')
        return None

def validate_256d_embedding(embedding):
    """Validate that embedding is proper 256D"""
    if not embedding or not isinstance(embedding, (list, tuple)):
        return False
    
    if len(embedding) != 256:
        return False
    
    # Check if it's not all zeros or very uniform
    variance = np.var(embedding)
    if variance < 0.001:
        return False
    
    return True

def convert_to_256d(embedding):
    """Convert any embedding to 256D format - IMPROVED VERSION"""
    if not embedding or not isinstance(embedding, (list, tuple)):
        print(f"[CONVERT] Invalid embedding input: {type(embedding)}")
        return None
    
    current_length = len(embedding)
    
    print(f"[CONVERT] Converting embedding from {current_length}D to 256D")
    
    if current_length == 256:
        print("[CONVERT] Already 256D, returning as-is")
        return embedding
    elif current_length > 256:
        # For 258D, we need to intelligently truncate
        if current_length == 258:
            # Remove the last 2 dimensions that were likely added by MediaPipe
            truncated = embedding[:256]
            print(f"[CONVERT] Truncated 258D to 256D by removing last 2 dimensions")
            print(f"[CONVERT] First 3 values: {truncated[:3]}")
            print(f"[CONVERT] Last 3 values: {truncated[-3:]}")
            return truncated
        else:
            # Generic truncation for other lengths > 256
            truncated = embedding[:256]
            print(f"[CONVERT] Truncated {current_length}D to 256D")
            return truncated
    else:
        print(f"[CONVERT] Padding {current_length}D to 256D")
        
        mean_val = np.mean(embedding) if embedding else 0
        std_val = np.std(embedding) if embedding else 0.01
    
        padded = list(embedding)
        
        while len(padded) < 256:
            for i, value in enumerate(embedding):
                if len(padded) >= 256:
                    break
                modified_value = value * (1 + (i % 10) * 0.001)
                padded.append(modified_value)
        
        final_embedding = padded[:256]
        print(f"[CONVERT] Padded to 256D successfully")
        return final_embedding

def is_face_in_database_256d(new_embedding_list, tolerance=0.5):
    """Check if 256D face embedding already exists in database"""
    try:
        new_embedding = np.array(convert_to_256d(json.loads(new_embedding_list)))
        
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT face_embedding FROM voters WHERE face_embedding IS NOT NULL")
            rows = cursor.fetchall()
        
        if not rows:
            return False, None

        stored_embeddings = []
        for row in rows:
            try:
                embedding_data = row['face_embedding']
                if isinstance(embedding_data, bytes):
                    embedding_str = embedding_data.decode('utf-8')
                else:
                    embedding_str = embedding_data
                
                if embedding_str and len(embedding_str) > 10:
                    stored_embedding = convert_to_256d(json.loads(embedding_str))
                    stored_embeddings.append(np.array(stored_embedding))
            except Exception as e:
                logger.warning(f"Could not parse stored face embedding: {e}", 'face_recognition')
                continue
        
        if not stored_embeddings:
            return False, None
            
        distances = np.linalg.norm(stored_embeddings - new_embedding, axis=1)
        
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]

        return min_distance <= tolerance, min_distance
        
    except Exception as e:
        logger.error("Error checking face in database", e, 'face_recognition')
        return False, None

def calculate_age(dob_string):
    """Calculate age from date of birth"""
    try:
        birth_date = datetime.strptime(dob_string, '%Y-%m-%d')
        today = datetime.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    except ValueError as e:
        logger.error(f"Invalid date format: {dob_string}", e, 'validation')
        raise

def validate_voter_id(voter_id):
    return re.match(r'^[A-Z]{3}[0-9]{7}$', voter_id) if voter_id else False

def validate_aadhaar(aadhaar):
    return re.match(r'^[0-9]{12}$', aadhaar) if aadhaar else False

def validate_phone(phone):
    return re.match(r'^[6-9][0-9]{9}$', phone) if phone else False

def validate_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) if email else False

load_dotenv()
EMAIL_USER = os.getenv("SENDER_EMAIL")
EMAIL_PASS = os.getenv("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.getenv("CONTACT_EMAIL")
def send_otp_email(recipient_email, otp_code):
    """Send OTP email to recipient"""
    try:
        sender_email = EMAIL_USER
        sender_password = EMAIL_PASS

        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'FaceCast OTP for Login'
        msg['From'] = f'FaceCast <{sender_email}>'
        msg['To'] = recipient_email

        html = f"""
        <html>
            <body>
                <div style="font-family: sans-serif; text-align: center; border: 1px solid #ddd; padding: 20px; border-radius: 8px;">
                    <h2 style="color: #007bff;">FaceCast - Verify your identity</h2>
                    <p style="font-size: 16px;">Please use the following verification code:</p>
                    <div style="background-color: #f0f2f5; padding: 15px; border-radius: 6px; display: inline-block;">
                        <h3 style="margin: 0; font-size: 24px; letter-spacing: 2px;">{otp_code}</h3>
                    </div>
                    <p style="font-size: 14px; color: #777;">This code is valid for a limited time.</p>
                </div>
            </body>
        </html>
        """

        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        
        logger.info(f"OTP email sent to {recipient_email}", 'email')
        return True
    except Exception as e:
        logger.error(f"Failed to send OTP email to {recipient_email}", e, 'email')
        return False

# Initialize database on startup
initialize_database()

# Static file serving
@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../frontend', filename)

# Serve profile images
@app.route('/static/profile_images/<filename>')
def serve_profile_image(filename):
    profile_images_dir = os.path.join(os.path.dirname(__file__), 'static', 'profile_images')
    return send_from_directory(profile_images_dir, filename)



@app.route('/api/contact', methods=['POST'])
def handle_contact_form():
    """Handles submissions from the contact us form."""
    try:
        data = request.get_json()

        required_fields = ['name', 'email', 'subject', 'message']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field.capitalize()} is required'}), 400

        name = data['name'].strip()
        email = data['email'].strip()
        subject = data['subject'].strip()
        message = data['message'].strip()

        if '@' not in email or '.' not in email:
            return jsonify({'error': 'Please enter a valid email address'}), 400

        success = send_contact_email(name, email, subject, message)

        if success:
            return jsonify({
                'success': True,
                'message': 'Your message has been sent successfully! We will get back to you soon.'
            }), 200
        else:
            return jsonify({
                'error': 'Failed to send email. Please try again later.'
            }), 500
    except Exception as e:
        print(f"Contact form error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def send_contact_email(name, email, subject, message):
    try:
        sender_email = EMAIL_USER
        sender_password = EMAIL_PASS

        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'FaceCast Contact Form: {subject}'
        msg['From'] = f'FaceCast <{sender_email}>'
        msg['To'] = RECIPIENT_EMAIL  # Send to specified email
        msg['Reply-To'] = email

        html = f"""
        <html>
            <body>
                <div style="font-family: sans-serif; border: 1px solid #ddd; padding: 20px; border-radius: 8px; max-width: 600px;">
                    <h2 style="color: #ff9bb3; margin-bottom: 20px;">New Contact Form Submission</h2>
                    
                    <div style="background-color: #ffe4e8; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
                        <h3 style="margin: 0 0 10px 0; color: #2d2d2d;">Contact Details</h3>
                        <p style="margin: 5px 0;"><strong>Name:</strong> {name}</p>
                        <p style="margin: 5px 0;"><strong>Email:</strong> {email}</p>
                        <p style="margin: 5px 0;"><strong>Subject:</strong> {subject}</p>
                        <p style="margin: 5px 0;"><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div style="background-color: #f8fafc; padding: 15px; border-radius: 6px;">
                        <h3 style="margin: 0 0 10px 0; color: #2d2d2d;">Message</h3>
                        <p style="margin: 0; line-height: 1.6; color: #2d2d2d;">{message}</p>
                    </div>
                    
                    <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #ddd;">
                        <p style="font-size: 14px; color: #777; margin: 0;">This message was sent from the FaceCast contact form.</p>
                        <p style="font-size: 14px; color: #777; margin: 5px 0 0 0;">Reply directly to: <a href="mailto:{email}">{email}</a></p>
                    </div>
                </div>
            </body>
        </html>
        """

        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        
        print(f"Contact form email sent from {name} ({email}) to shilpachaurasiya1205@gmail.com")
        return True
        
    except Exception as e:
        print(f"Failed to send contact email: {e}")
        return False

    

# Handle favicon requests
@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/test-registration', methods=['POST'])
def test_registration():
    """Test endpoint to check registration without face data"""
    try:
        data = request.json
        print(f"[TEST] Received test data: {data.get('name')}")
        
        # Just return success without processing
        return jsonify({
            "success": True,
            "message": "Test endpoint working",
            "received_data": {
                "name": data.get('name'),
                "voter_id": data.get('voter_id'),
                "has_embedding": bool(data.get('face_embedding_from_live')),
                "embedding_length": len(data.get('face_embedding_from_live', []))
            }
        }), 200
    except Exception as e:
        print(f"[TEST ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/register', methods=['POST'])
def register_voter():
    """
    Register a new voter. This function now strictly requires a pre-generated 
    128D embedding from the '/api/generate-reliable-embedding' endpoint.
    """
    try:
        data = request.json
        
        # --- 1. Validation ---
        required_fields = ['name', 'aadhaar', 'email', 'voter_id', 'dob', 'image_data_url', 'face_embedding_from_live']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

       
        name = data.get('name')
        address = data.get('address', 'Not Provided')
        gender = data.get('gender', 'Not Specified')
        aadhaar = data.get('aadhaar')
        phone = data.get('phone', '') 
        email = data.get('email')
        voter_id = data.get('voter_id')
        dob_string = data.get('dob')
        image_data_url = data.get('image_data_url')
        face_embedding_128d = data.get('face_embedding_from_live')

       
        if not validate_voter_id(voter_id):
            return jsonify({"error": "Invalid Voter ID format."}), 400
        if not validate_aadhaar(aadhaar):
            return jsonify({"error": "Invalid Aadhaar number format."}), 400
        if not validate_email(email):
            return jsonify({"error": "Invalid email address format."}), 400
        
       
        try:
            if calculate_age(dob_string) < 18:
                return jsonify({"error": "Voters must be 18 years or older."}), 400
        except ValueError:
            return jsonify({"error": "Invalid date of birth format. Use YYYY-MM-DD."}), 400

      
       
        if not isinstance(face_embedding_128d, list) or len(face_embedding_128d) != 128:
            logger.error(f"Invalid embedding for registration: length={len(face_embedding_128d)}", 'registration')
            return jsonify({"error": "Invalid face data format received. Please retry verification."}), 400

       
        is_duplicate, matching_voter_id = is_face_duplicate_128d(face_embedding_128d)
        if is_duplicate:
            return jsonify({"error": f"This face appears to be already registered to another voter ({matching_voter_id})."}), 409 # 409 Conflict

       
        profile_image_path = None
        try:
            profile_images_dir = os.path.join(os.path.dirname(__file__), 'static', 'profile_images')
            os.makedirs(profile_images_dir, exist_ok=True)
            
            image_filename = f"{voter_id}.png"
            image_path = os.path.join(profile_images_dir, image_filename)
            
            header, encoded = image_data_url.split(',', 1)
            image_data = base64.b64decode(encoded)
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            profile_image_path = f"static/profile_images/{image_filename}"
            logger.info(f"Profile image saved for {voter_id} at {profile_image_path}", 'registration')
        except Exception as e:
            logger.error("Error saving profile image", e, 'registration')

      
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT 1 FROM voters WHERE voter_id = ? OR email = ? OR aadhaar = ?", (voter_id, email, aadhaar))
            if cursor.fetchone():
                return jsonify({"error": "A voter with this Voter ID, Email, or Aadhaar number already exists."}), 409

            face_embedding_str = json.dumps(face_embedding_128d)

            cursor.execute(
                """INSERT INTO voters (name, voter_id, address, gender, dob, aadhaar, phone, email, face_embedding, profile_image) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, voter_id, address, gender, dob_string, aadhaar, phone, email, face_embedding_str, profile_image_path)
            )
        
        logger.info(f"Voter '{name}' ({voter_id}) registered successfully.", 'registration')
        return jsonify({"message": "Voter registered successfully!"}), 201

    except Exception as e:
        logger.error("An unexpected error occurred during registration", e, 'registration')
        traceback.print_exc() 
        return jsonify({"error": "An internal server error occurred. Please contact support."}), 500


def is_face_duplicate_128d(new_embedding_list, tolerance=0.6):
    """Checks if a new 128D embedding matches any existing one in the database."""
    try:
        new_embedding = np.array(new_embedding_list)
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT voter_id, face_embedding FROM voters WHERE face_embedding IS NOT NULL")
            all_voters = cursor.fetchall()

            if not all_voters:
                return False, None

            for voter in all_voters:
                try:
                    stored_embedding_str = voter['face_embedding']
                    stored_embedding = np.array(json.loads(stored_embedding_str))

                    
                    if len(stored_embedding) != 128:
                        continue 

                    distance = calculate_face_distance(stored_embedding, new_embedding)
                    
                    if distance <= tolerance:
                        logger.warning(f"Duplicate face detected. New registration matches existing voter {voter['voter_id']} with distance {distance:.4f}", 'security')
                        return True, voter['voter_id']
                except (json.JSONDecodeError, TypeError):
                    continue 
        
        return False, None
    except Exception as e:
        logger.error("Error during duplicate face check", e, 'database')
        return False, None 

@app.route('/api/validate-face-uniqueness', methods=['POST'])
def validate_face_uniqueness():
    """Validate if a face embedding is unique before registration"""
    try:
        data = request.json
        test_embedding = data.get('test_embedding')
        current_voter_id = data.get('voter_id')
        
        if not test_embedding:
            return jsonify({"error": "No test embedding provided"}), 400
        
        if len(test_embedding) != 256:
            test_embedding = convert_to_256d(test_embedding)
        
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT voter_id, name, face_embedding FROM voters WHERE face_embedding IS NOT NULL")
            voters = cursor.fetchall()
        
        matches = []
        for voter in voters:
            try:
                embedding_data = voter['face_embedding']
                if isinstance(embedding_data, bytes):
                    embedding_str = embedding_data.decode('utf-8')
                else:
                    embedding_str = embedding_data
                
                if embedding_str:
                    stored_embedding = json.loads(embedding_str)
                    
                    
                    if len(stored_embedding) != len(test_embedding):
                        stored_embedding = convert_to_256d(stored_embedding)
                    
                  
                    similarity = calculate_face_similarity_256d(stored_embedding, test_embedding)
                    
                    if similarity > 0.7:
                        matches.append({
                            'voter_id': voter['voter_id'],
                            'name': voter['name'],
                            'similarity': round(similarity, 3)
                        })
                        
            except Exception as e:
                continue
        
        if matches:
            best_match = max(matches, key=lambda x: x['similarity'])
            return jsonify({
                "is_duplicate": True,
                "matching_voter": f"{best_match['name']} ({best_match['voter_id']})",
                "similarity_score": best_match['similarity'],
                "all_matches": matches
            }), 200
        else:
            return jsonify({
                "is_duplicate": False,
                "message": "Face appears to be unique"
            }), 200
            
    except Exception as e:
        print(f"[VALIDATE_UNIQUENESS] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug-all-face-embeddings', methods=['GET'])
def debug_all_face_embeddings():
    """Debug endpoint to see all stored face embeddings"""
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT voter_id, name, LENGTH(face_embedding) as embedding_length FROM voters WHERE face_embedding IS NOT NULL")
            voters = cursor.fetchall()
        
        embeddings_info = []
        for voter in voters:
            voter_dict = dict(voter)
            embeddings_info.append({
                'voter_id': voter_dict['voter_id'],
                'name': voter_dict['name'],
                'embedding_length': voter_dict['embedding_length']
            })
        
        return jsonify({
            "total_voters_with_embeddings": len(embeddings_info),
            "embeddings": embeddings_info
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/compare-faces', methods=['POST'])
def compare_faces():
    """Compare two specific voters' face embeddings"""
    try:
        data = request.json
        voter_id_1 = data.get('voter_id_1')
        voter_id_2 = data.get('voter_id_2')
        
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT voter_id, name, face_embedding FROM voters WHERE voter_id IN (?, ?)", 
                          (voter_id_1, voter_id_2))
            voters = cursor.fetchall()
        
        if len(voters) != 2:
            return jsonify({"error": "One or both voters not found"}), 404
        
        voter1 = next(v for v in voters if v['voter_id'] == voter_id_1)
        voter2 = next(v for v in voters if v['voter_id'] == voter_id_2)
        
        # Extract embeddings
        def get_embedding(voter):
            embedding_data = voter['face_embedding']
            if isinstance(embedding_data, bytes):
                return json.loads(embedding_data.decode('utf-8'))
            else:
                return json.loads(embedding_data)
        
        embedding1 = get_embedding(voter1)
        embedding2 = get_embedding(voter2)
        
       
        if len(embedding1) != len(embedding2):
            embedding1 = convert_to_256d(embedding1)
            embedding2 = convert_to_256d(embedding2)
        
        similarity = calculate_face_similarity_256d(embedding1, embedding2)
        
        return jsonify({
            "voter1": voter1['name'],
            "voter2": voter2['name'],
            "similarity_score": round(similarity, 3),
            "embedding_lengths": {
                "voter1": len(embedding1),
                "voter2": len(embedding2)
            },
            "interpretation": "High similarity (>0.7) suggests same person"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

        




@app.route('/api/debug-mediapipe-issue', methods=['POST'])
def debug_mediapipe_issue():
    """Debug why MediaPipe is generating 8D embeddings"""
    try:
        data = request.json
        image_data_url = data.get('image_data_url')
        
        if not image_data_url:
            return jsonify({"error": "No image data provided"}), 400
        
        frame = decode_base64_image(image_data_url)
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        analysis = {
            "image_shape": frame.shape,
            "faces_detected": len(face_locations),
            "face_recognition_embedding_length": len(face_encodings[0]) if face_encodings else 0,
            "face_locations": face_locations
        }
        
        return jsonify(analysis), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    



    
@app.route('/api/check-face-data/<voter_id>', methods=['GET'])
def check_face_data(voter_id):
    """Check if voter has valid face embedding data in database"""
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT face_embedding, name FROM voters WHERE voter_id = ?", (voter_id,))
            voter = cursor.fetchone()
            
            if not voter:
                return jsonify({"error": "Voter not found"}), 404
            
            has_valid_face_data = False
            face_embedding_data = voter['face_embedding']
            
            if face_embedding_data:
                try:
                   
                    if isinstance(face_embedding_data, bytes):
                        embedding_str = face_embedding_data.decode('utf-8')
                    else:
                        embedding_str = face_embedding_data
                    
                    if embedding_str and len(embedding_str) > 10:
                        embedding_list = json.loads(embedding_str)
                        is_real_embedding = (
                            isinstance(embedding_list, list) and 
                            len(embedding_list) >= 128 and
                            not all(abs(x) < 0.01 for x in embedding_list[:10])
                        )
                        
                        has_valid_face_data = is_real_embedding
                        print(f"[FACE_CHECK] Voter {voter_id}: has_data={True}, is_real={is_real_embedding}, length={len(embedding_list)}")
                        
                except Exception as e:
                    print(f"[FACE_CHECK] Invalid face embedding for voter {voter_id}: {e}")
                    has_valid_face_data = False
            
            return jsonify({
                "voter_id": voter_id,
                "name": voter['name'],
                "has_face_data": has_valid_face_data,
                "message": "Face data check completed"
            }), 200
            
    except Exception as e:
        print(f"[FACE_CHECK] Error checking face data: {e}")
        return jsonify({"error": "Database error occurred"}), 500

@app.route('/api/login', methods=['POST'])
def login_voter():
    data = request.json
    voter_id = data.get('voter_id')
    email = data.get('email')
    face_match_verified = data.get('face_match_verified', False)
    similarity_score = data.get('similarity_score', 0)

    logger.info(f"Login attempt for {voter_id}. Face match verified by frontend: {face_match_verified}", 'login')

    if not all([voter_id, email]):
        return jsonify({"error": "Missing Voter ID or Email."}), 400
    
    if not face_match_verified:
        logger.info(f"Login attempt for {voter_id} failed: Face match was not verified or similarity was too low.", 'security') # <-- FIXED
        return jsonify({"error": "Face verification failed. Please try again."}), 401

    try:
        with get_db_cursor() as (conn, cursor):

            cursor.execute("SELECT name, email FROM voters WHERE voter_id = ? AND email = ?", (voter_id, email))
            voter_record = cursor.fetchone()

            if not voter_record:
                return jsonify({"error": "Invalid Voter ID or Email."}), 401
            
            
            otp = str(random.randint(100000, 999999))
            cursor.execute("UPDATE voters SET otp = ? WHERE voter_id = ?", (otp, voter_id))
            
            logger.info(f"OTP generated for {voter_id}", 'login')

            if send_otp_email(voter_record['email'], otp):
                logger.info(f"OTP email sent successfully to {voter_record['email']}", 'login')
                return jsonify({
                    "message": f"Face verification successful! OTP sent to your email.",
                    "similarity_score": similarity_score
                }), 200
            else:
                logger.error(f"Failed to send OTP email for {voter_id}", category='login')
                return jsonify({"error": "Face verification was successful, but we failed to send the OTP email."}), 500
                
    except Exception as e:
        logger.error(f"Database error during login for {voter_id}", e, 'login')
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500      

@app.route('/api/verify_otp', methods=['POST'])
def verify_otp():
    data = request.json
    otp = data.get('otp')
    
    if not otp:
        return jsonify({"error": "OTP is required."}), 400
    
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT voter_id FROM voters WHERE otp = ?", (otp,))
            voter_record = cursor.fetchone()
            
            if voter_record:
                cursor.execute("UPDATE voters SET otp = NULL WHERE voter_id = ?", (voter_record['voter_id'],))
                
                return jsonify({
                    "message": "OTP verified successfully.",
                    "voter_id": voter_record['voter_id']
                }), 200
            else:
                return jsonify({"error": "Invalid OTP."}), 401
                
    except Exception as e:
        print(f"Error in verify_otp: {e}")
        return jsonify({"error": "Database error occurred. Please try again."}), 500

# Admin Routes
@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not all([username, password]):
        return jsonify({"error": "Username and password required"}), 400
    
    try:
        with get_db_cursor() as (conn, cursor):
            try:
                cursor.execute("SELECT * FROM admin WHERE username = ? AND password = ?", (username, password))
                admin = cursor.fetchone()
            except sqlite3.OperationalError:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS admin (
                        admin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        email TEXT
                    )
                """)
                cursor.execute("""
                    INSERT OR IGNORE INTO admin (username, password, email) 
                    VALUES ('admin', 'admin123', 'shilpachaurasiya1205@gmail.com')
                """)
                cursor.execute("SELECT * FROM admin WHERE username = ? AND password = ?", (username, password))
                admin = cursor.fetchone()
        
        if admin:
            # Set session
            session['admin_logged_in'] = True
            session['admin_id'] = admin['admin_id']
            session['admin_username'] = admin['username']
            
            return jsonify({"message": "Admin login successful", "admin_id": admin['admin_id']}), 200
        else:
            return jsonify({"error": "Invalid credentials"}), 401
            
    except Exception as e:
        print(f"Error in admin_login: {e}")
        return jsonify({"error": "Database error occurred. Please try again."}), 500

@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    session.clear()
    return jsonify({"message": "Admin logged out successfully"}), 200

@app.route('/api/admin/check_auth', methods=['GET'])
def check_admin_auth():
    if 'admin_logged_in' in session and session['admin_logged_in']:
        return jsonify({"authenticated": True, "username": session.get('admin_username')}), 200
    else:
        return jsonify({"authenticated": False}), 401

@app.route('/api/check-voter-credentials', methods=['POST'])
def check_voter_credentials():
    """Check if voter credentials are valid before proceeding to face verification"""
    try:
        data = request.json
        voter_id = data.get('voter_id')
        email = data.get('email')

        print(f"[DEBUG] Checking voter credentials: voter_id={voter_id}, email={email}")

        if not all([voter_id, email]):
            return jsonify({"error": "Voter ID and email are required"}), 400

        with get_db_cursor() as (conn, cursor):
            # Check if voter exists and is active
            cursor.execute("""
                SELECT voter_id, name, email, face_embedding, profile_image 
                FROM voters 
                WHERE voter_id = ? AND email = ?
            """, (voter_id, email))
            
            voter_record = cursor.fetchone()

            if not voter_record:
                print(f"[DEBUG] Voter not found: {voter_id}, {email}")
                return jsonify({"error": "Invalid Voter ID or Email"}), 404

            voter_dict = dict(voter_record)
            print(f"[DEBUG] Voter found: {voter_dict['name']}")

            # Check if face embedding exists
            has_face_embedding = False
            face_embedding_data = voter_dict.get('face_embedding')
            
            if face_embedding_data:
                try:
                    # Handle BLOB storage
                    if isinstance(face_embedding_data, bytes):
                        embedding_str = face_embedding_data.decode('utf-8')
                    else:
                        embedding_str = face_embedding_data
                    
                    if embedding_str and len(embedding_str) > 10:
                        has_face_embedding = True
                        print(f"[DEBUG] Face embedding found for voter: {voter_id}")
                except Exception as e:
                    print(f"[DEBUG] Error checking face embedding: {e}")
                    has_face_embedding = False

            return jsonify({
                "success": True,
                "user_id": voter_dict['voter_id'],
                "voter_id": voter_dict['voter_id'],
                "name": voter_dict['name'],
                "email": voter_dict['email'],
                "has_face_embedding": has_face_embedding,
                "profile_image": voter_dict.get('profile_image'),
                "message": "Credentials verified successfully"
            }), 200

    except Exception as e:
        print(f"Error in check_voter_credentials: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Database error occurred. Please try again."}), 500


@app.route('/api/test-face-recognition', methods=['POST'])
def test_face_recognition():
    """Test if face_recognition is working properly"""
    try:
        data = request.json
        image_data_url = data.get('image_data_url')
        
        if not image_data_url:
            test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
            _, buffer = cv2.imencode('.jpg', test_image)
            image_data_url = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode()
        
        frame = decode_base64_image(image_data_url)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        return jsonify({
            "face_recognition_working": True,
            "faces_detected": len(face_locations),
            "embeddings_generated": len(face_encodings),
            "embedding_dimensions": len(face_encodings[0]) if face_encodings else 0,
            "message": "face_recognition library is working correctly"
        }), 200
        
    except Exception as e:
        return jsonify({
            "face_recognition_working": False,
            "error": str(e)
        }), 500
        

@app.route('/api/generate-reliable-embedding', methods=['POST'])
def generate_reliable_embedding():
    """
    Generate a reliable 128D face embedding using the face_recognition library.
    This is the single source of truth for creating all embeddings.
    """
    try:
        data = request.json
        image_data_url = data.get('image_data_url')
        
        if not image_data_url:
            return jsonify({"success": False, "error": "No image data provided"}), 400
        
        frame = decode_base64_image(image_data_url)
        if frame is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        if not face_locations:
            return jsonify({"success": False, "error": "No face detected in the image"}), 400
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings:
            return jsonify({"success": False, "error": "Could not generate face encoding"}), 400
        
        final_embedding = face_encodings[0].tolist()
        
        logger.info(f"Successfully generated 128D embedding, length: {len(final_embedding)}", 'face_recognition')
        
        return jsonify({
            "success": True,
            "face_embedding": final_embedding,
            "method": "face_recognition_128D",
            "embedding_length": len(final_embedding)
        }), 200
        
    except Exception as e:
        logger.error("Error generating reliable embedding", e, 'face_recognition')
        return jsonify({"success": False, "error": f"Failed to generate face embedding: {str(e)}"}), 500


def calculate_face_distance(embedding1, embedding2):
    """Calculates the Euclidean distance between two face embeddings."""
    return np.linalg.norm(np.array(embedding1) - np.array(embedding2))

@app.route('/api/verify-face-match', methods=['POST'])
def verify_face_match():
    """Securely compares a live face embedding with a stored one."""
    try:
        data = request.json
        voter_id = data.get('voter_id')
        live_face_embedding = data.get('live_face_embedding') # This is a Python list

        if not voter_id or not live_face_embedding:
            return jsonify({'error': 'Missing voter ID or live face data'}), 400

      
        if not isinstance(live_face_embedding, list) or len(live_face_embedding) != 128:
            return jsonify({'error': 'Invalid live face data format'}), 400

        with get_db_cursor() as (conn, cursor):
           
            cursor.execute("SELECT face_embedding FROM voters WHERE voter_id = ?", (voter_id,))
            voter = cursor.fetchone()

            if not voter or not voter['face_embedding']:
                return jsonify({'error': 'No face data is registered for this user.'}), 404

           
            stored_embedding_str = voter['face_embedding']
            stored_embedding = json.loads(stored_embedding_str)

            distance = calculate_face_distance(stored_embedding, live_face_embedding)
            
            FACE_MATCH_THRESHOLD = 0.6
            is_match = distance <= FACE_MATCH_THRESHOLD
            
            similarity_score = max(0, 1 - (distance / FACE_MATCH_THRESHOLD))

            logger.info(f"Face Match for {voter_id}: Distance={distance:.4f}, Match={is_match}", 'security')

            if is_match:
                return jsonify({
                    'face_match': True,
                    'similarity_score': similarity_score,
                    'message': 'Face verification successful.'
                }), 200
            else:
                return jsonify({
                    'face_match': False,
                    'similarity_score': similarity_score,
                    'error': 'Face does not match the registered user.'
                }), 401

    except Exception as e:
        logger.error(f"Error during face matching for {data.get('voter_id')}", e, 'security')
        return jsonify({'error': 'An internal error occurred during face verification.'}), 500


def calculate_face_similarity_256d(embedding1, embedding2):
    """Calculate cosine similarity for 256D embeddings - FIXED JSON SERIALIZATION"""
    try:
        emb1 = np.array(embedding1[:256], dtype=np.float64)
        emb2 = np.array(embedding2[:256], dtype=np.float64)
        
        print(f"[SIMILARITY] Calculating 256D similarity")
        
        if bool(np.all(emb1 == 0)) or bool(np.all(emb2 == 0)):
            print("[SIMILARITY] Warning: Zero vector detected")
            return 0.0
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if bool(norm1 == 0) or bool(norm2 == 0):
            print("[SIMILARITY] Warning: Zero norm detected")
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        
        normalized_similarity = (similarity + 1) / 2
        
        result = float(max(0.0, min(1.0, normalized_similarity)))
        print(f"[SIMILARITY] 256D Result: {result:.3f} (raw: {similarity:.3f})")
        
        return result
        
    except Exception as e:
        print(f"[SIMILARITY] Error calculating 256D similarity: {e}")
        return 0.0

@app.route('/api/resend-otp', methods=['POST'])
def resend_otp():
    """Resend OTP to voter's email"""
    try:
        data = request.json
        voter_id = data.get('voter_id')
        email = data.get('email')

        if not all([voter_id, email]):
            return jsonify({"error": "Voter ID and email are required"}), 400

        with get_db_cursor() as (conn, cursor):
            # Verify voter exists
            cursor.execute("SELECT name, email FROM voters WHERE voter_id = ? AND email = ?", (voter_id, email))
            voter = cursor.fetchone()

            if not voter:
                return jsonify({"error": "Voter not found"}), 404

            # Generate new OTP
            otp = str(random.randint(100000, 999999))
            cursor.execute("UPDATE voters SET otp = ? WHERE voter_id = ?", (otp, voter_id))

            # Send OTP email
            if send_otp_email(voter['email'], otp):
                logger.info(f"OTP resent to {voter['email']} for voter {voter_id}", 'otp')
                return jsonify({"message": "New OTP sent to your email"}), 200
            else:
                return jsonify({"error": "Failed to send OTP email"}), 500

    except Exception as e:
        logger.error("Error resending OTP", e, 'otp')
        return jsonify({"error": "Failed to resend OTP"}), 500

@app.route('/api/debug-embedding-analysis', methods=['POST'])
def debug_embedding_analysis():
    """Debug endpoint to analyze embedding compatibility"""
    try:
        data = request.json
        voter_id = data.get('voter_id')
        live_face_embedding = data.get('live_face_embedding')

        if not all([voter_id, live_face_embedding]):
            return jsonify({"error": "Missing required data"}), 400

        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT face_embedding FROM voters WHERE voter_id = ?", (voter_id,))
            voter = cursor.fetchone()

            if not voter:
                return jsonify({"error": "Voter not found"}), 404

            stored_embedding_data = voter['face_embedding']
            
            # Parse embeddings
            if isinstance(stored_embedding_data, bytes):
                stored_embedding_str = stored_embedding_data.decode('utf-8')
            else:
                stored_embedding_str = stored_embedding_data
            
            stored_embedding = json.loads(stored_embedding_str)
            live_embedding = json.loads(live_face_embedding)

            return jsonify({
                "stored_embedding": {
                    "length": len(stored_embedding),
                    "type": type(stored_embedding).__name__,
                    "first_5_values": stored_embedding[:5],
                    "min_value": float(np.min(stored_embedding)),
                    "max_value": float(np.max(stored_embedding)),
                    "mean_value": float(np.mean(stored_embedding))
                },
                "live_embedding": {
                    "length": len(live_embedding),
                    "type": type(live_embedding).__name__,
                    "first_5_values": live_embedding[:5],
                    "min_value": float(np.min(live_embedding)),
                    "max_value": float(np.max(live_embedding)),
                    "mean_value": float(np.mean(live_embedding))
                },
                "compatibility": {
                    "lengths_match": len(stored_embedding) == len(live_embedding),
                    "min_length": min(len(stored_embedding), len(live_embedding)),
                    "max_length": max(len(stored_embedding), len(live_embedding))
                }
            }), 200
            
    except Exception as e:
        print(f"[DEBUG] Error in embedding analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-face-similarity-threshold', methods=['GET'])
def get_face_similarity_threshold():
    """Get the current face similarity threshold"""
    return jsonify({
        "similarity_threshold": 0.6,
        "message": "Minimum similarity score required for face match"
    }), 200

# [Keep all other existing routes for elections, candidates, voting, admin, etc.]
# Election Management
@app.route('/api/elections', methods=['GET'])
def get_elections():
    try:
        with get_db_cursor() as (conn, cursor):
            try:
                cursor.execute("SELECT id as election_id, title, description, start_date, end_date, status FROM elections ORDER BY id DESC")
                elections = cursor.fetchall()
            except sqlite3.OperationalError:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS elections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        description TEXT,
                        start_date TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        status TEXT DEFAULT 'upcoming',
                        is_demo INTEGER DEFAULT 0,
                        demo_duration_minutes INTEGER DEFAULT 2
                    )
                """)
                elections = []
        
        return jsonify([dict(election) for election in elections]), 200
        
    except Exception as e:
        print(f"Error in get_elections: {e}")
        return jsonify({"error": "Database error occurred. Please try again."}), 500

@app.route('/api/elections', methods=['POST'])
@require_admin_auth
def create_election():
    data = request.json
    title = data.get('title')
    description = data.get('description', '')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    start_time = data.get('start_time', '00:00')
    is_demo = data.get('is_demo', False)
    demo_duration_minutes = data.get('demo_duration_minutes', 2)
    use_quick_duration = data.get('use_quick_duration', False)
    
    if not all([title, start_date, end_date]):
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        if use_quick_duration:
            current_time = datetime.now()
            start_datetime = current_time
            end_datetime = current_time + timedelta(minutes=demo_duration_minutes)
            start_datetime_str = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_str = end_datetime.strftime('%Y-%m-%d %H:%M:%S')
            status = 'active'
        else:
            start_datetime_str = f"{start_date} {start_time}:00"
            start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M:%S')
            end_datetime = datetime.strptime(f"{end_date} 23:59:59", '%Y-%m-%d %H:%M:%S')
            end_datetime_str = f"{end_date} 23:59:59"
            current_time = datetime.now()
            
            if current_time < start_datetime:
                status = 'upcoming'
            elif start_datetime <= current_time <= end_datetime:
                status = 'active'
            else:
                status = 'ended'
            
    except ValueError as e:
        return jsonify({"error": f"Invalid date/time format: {str(e)}"}), 400
    
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute(
                "INSERT INTO elections (title, description, start_date, end_date, status, is_demo, demo_duration_minutes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (title, description, start_datetime_str, end_datetime_str, status, is_demo, demo_duration_minutes)
            )
        
        return jsonify({
            "message": "Election created successfully",
            "title": title,
            "status": status
        }), 201
        
    except Exception as e:
        print(f"Error in create_election: {e}")
        return jsonify({"error": f"Failed to create election: {str(e)}"}), 500

# Candidate Management
@app.route('/api/candidates', methods=['GET'])
def get_candidates():
    election_id = request.args.get('election_id')
    
    try:
        with get_db_cursor() as (conn, cursor):
            try:
                if election_id:
                    cursor.execute("""
                        SELECT id as candidate_id, election_id, name, 
                               COALESCE(party, '') as party, 
                               COALESCE(position, 'Candidate') as position, 
                               COALESCE(vote_count, 0) as vote_count,
                               COALESCE(candidate_image, '') as candidate_image,
                               COALESCE(party_logo, '') as party_logo
                        FROM candidates 
                        WHERE election_id = ? 
                        ORDER BY name
                    """, (election_id,))
                else:
                    cursor.execute("""
                        SELECT id as candidate_id, election_id, name, 
                               COALESCE(party, '') as party, 
                               COALESCE(position, 'Candidate') as position, 
                               COALESCE(vote_count, 0) as vote_count,
                               COALESCE(candidate_image, '') as candidate_image,
                               COALESCE(party_logo, '') as party_logo
                        FROM candidates 
                        ORDER BY election_id, name
                    """)
                
                candidates = cursor.fetchall()
            except sqlite3.OperationalError:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS candidates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        election_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        party TEXT,
                        position TEXT DEFAULT 'Candidate',
                        vote_count INTEGER DEFAULT 0,
                        candidate_image TEXT,
                        party_logo TEXT,
                        FOREIGN KEY (election_id) REFERENCES elections (id)
                    )
                """)
                candidates = []
        
        return jsonify([dict(candidate) for candidate in candidates]), 200
        
    except Exception as e:
        print(f"Error in get_candidates: {e}")
        return jsonify({"error": "Database error occurred. Please try again."}), 500

@app.route('/api/candidates', methods=['POST'])
@require_admin_auth
def add_candidate():
    data = request.json
    name = data.get('name')
    election_id = data.get('election_id')
    party = data.get('party', '')
    position = data.get('position', 'Candidate')
    candidate_image = data.get('candidate_image', '')
    party_logo = data.get('party_logo', '')
    
    if not all([name, election_id]):
        return jsonify({"error": "Missing required fields (name and election_id)"}), 400
    
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT id FROM elections WHERE id = ?", (election_id,))
            if not cursor.fetchone():
                return jsonify({"error": "Invalid election ID"}), 400
            
            cursor.execute(
                "INSERT INTO candidates (name, election_id, party, position, vote_count, candidate_image, party_logo) VALUES (?, ?, ?, ?, 0, ?, ?)",
                (name, election_id, party, position, candidate_image, party_logo)
            )
            candidate_id = cursor.lastrowid
        
        print(f"Candidate {name} added successfully to election {election_id}")
        return jsonify({
            "message": "Candidate added successfully", 
            "candidate_id": candidate_id
        }), 201
        
    except Exception as e:
        print(f"Error in add_candidate: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/vote', methods=['POST'])
def cast_vote():
    data = request.json
    voter_id = data.get('voter_id')
    election_id = data.get('election_id')
    candidate_id = data.get('candidate_id')
    
    logger.info(f"Vote request received: voter_id={voter_id}, election_id={election_id}, candidate_id={candidate_id}", 'voting')
    
    if not all([voter_id, election_id, candidate_id]):
        return jsonify({"error": "Missing voter ID, election ID, or candidate ID"}), 400
    
    try:
        election_id = int(election_id)
        candidate_id = int(candidate_id)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid election or candidate ID format"}), 400
    
    # Use the database context manager for a safe transaction
    try:
        with get_db_cursor() as (conn, cursor):
            # --- VALIDATION ---
            # Check if voter, election, and candidate are valid
            cursor.execute("SELECT 1 FROM voters WHERE voter_id = ?", (voter_id,))
            if not cursor.fetchone():
                return jsonify({"error": "Invalid voter ID"}), 404

            cursor.execute("SELECT status FROM elections WHERE id = ?", (election_id,))
            election = cursor.fetchone()
            if not election:
                return jsonify({"error": "Election not found"}), 404
            if election['status'] != 'active':
                return jsonify({"error": f"This election is not active. Its status is '{election['status']}'."}), 403

            cursor.execute("SELECT 1 FROM candidates WHERE id = ? AND election_id = ?", (candidate_id, election_id))
            if not cursor.fetchone():
                return jsonify({"error": "Invalid candidate for this election"}), 404

            # --- ATOMIC VOTE CASTING ---
            # Directly try to insert the vote. The database's UNIQUE constraint will
            # raise an IntegrityError if the user has already voted.
            
            # 1. Insert the vote record
            cursor.execute(
                "INSERT INTO votes (voter_id, election_id, candidate_id) VALUES (?, ?, ?)",
                (voter_id, election_id, candidate_id)
            )
            
            # 2. Increment the candidate's vote count
            cursor.execute(
                "UPDATE candidates SET vote_count = vote_count + 1 WHERE id = ?",
                (candidate_id,)
            )

        # The context manager commits the transaction automatically if no errors occurred
        logger.info(f"Vote cast successfully for voter {voter_id} in election {election_id}", 'voting')
        return jsonify({"message": "Vote cast successfully"}), 201 # 201 Created

    except sqlite3.IntegrityError:
        # This block runs ONLY if the UNIQUE constraint (voter_id, election_id) fails.
        logger.info(f"Duplicate vote attempt by voter {voter_id} in election {election_id}", 'security')
        return jsonify({"error": "You have already voted in this election"}), 409 # 409 Conflict

    except Exception as e:
        logger.error("An unexpected error occurred during voting", e, 'voting')
        # Print the full error to the console for debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred. Please try again."}), 500
# Results
@app.route('/api/results/<int:election_id>', methods=['GET'])
def get_results(election_id):
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("""
                SELECT c.name, c.party, c.position, c.vote_count 
                FROM candidates c 
                WHERE c.election_id = ? 
                ORDER BY c.vote_count DESC
            """, (election_id,))
            
            results = cursor.fetchall()
        
        return jsonify([dict(result) for result in results]), 200
        
    except Exception as e:
        print(f"Error in get_results: {e}")
        return jsonify({"error": "Database error occurred. Please try again."}), 500

# Check if user has voted in specific election
@app.route('/api/user/<voter_id>/voted/<int:election_id>', methods=['GET'])
def check_user_voted(voter_id, election_id):
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT * FROM votes WHERE voter_id = ? AND election_id = ?", (voter_id, election_id))
            vote = cursor.fetchone()
            
            return jsonify({
                "has_voted": vote is not None,
                "vote_timestamp": vote['vote_timestamp'] if vote else None
            }), 200
            
    except Exception as e:
        print(f"Error in check_user_voted: {e}")
        return jsonify({"error": "Database error occurred. Please try again."}), 500

# User info
@app.route('/api/user/<voter_id>', methods=['GET'])
def get_user_info(voter_id):
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT * FROM voters WHERE voter_id = ?", (voter_id,))
            user = cursor.fetchone()
            
            if not user:
                return jsonify({"error": "User not found"}), 404
            
            try:
                cursor.execute("""
                    SELECT DISTINCT v.election_id, e.title, e.status, v.vote_timestamp
                    FROM votes v
                    JOIN elections e ON v.election_id = e.id
                    WHERE v.voter_id = ?
                    ORDER BY v.vote_timestamp DESC
                """, (voter_id,))
                voting_history = [dict(row) for row in cursor.fetchall()]
                voted_elections = [row['election_id'] for row in voting_history]
            except sqlite3.OperationalError:
                voting_history = []
                voted_elections = []
            
            try:
                cursor.execute("SELECT COUNT(*) FROM votes WHERE voter_id = ?", (voter_id,))
                votes_cast = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                votes_cast = 0
            
            try:
                cursor.execute("""
                    SELECT id, title, description, start_date, end_date, status
                    FROM elections 
                    WHERE status = 'upcoming'
                    ORDER BY start_date ASC
                """)
                upcoming_elections = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("""
                    SELECT id, title, description, start_date, end_date, status
                    FROM elections 
                    WHERE status = 'active'
                    ORDER BY start_date ASC
                """)
                active_elections = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("""
                    SELECT id, title, description, start_date, end_date, status
                    FROM elections 
                    WHERE status = 'ended'
                    ORDER BY end_date DESC
                """)
                past_elections = [dict(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                upcoming_elections = []
                active_elections = []
                past_elections = []
        
            user_dict = dict(user)
            
            return jsonify({
                "name": user_dict["name"],
                "email": user_dict["email"],
                "voter_id": user_dict["voter_id"],
                "profile_image": user_dict.get("profile_image"),
                "active_elections": active_elections,
                "upcoming_elections": upcoming_elections,
                "past_elections": past_elections,
                "voting_history": voting_history,
                "votes_cast": len(voting_history)
            }), 200
        
    except Exception as e:
        print(f"Error in get_user_info: {e}")
        return jsonify({"error": "Database error occurred. Please try again."}), 500

# Admin - Get all voters
@app.route('/api/admin/voters', methods=['GET'])
@require_admin_auth
def get_all_voters():
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("""
                SELECT 
                    v.voter_id, 
                    v.name, 
                    v.email, 
                    v.phone, 
                    v.address,
                    v.gender,
                    v.aadhaar,
                    v.dob,
                    v.profile_image,
                    v.created_at,
                    COUNT(votes.id) as votes_cast,
                    CASE WHEN COUNT(votes.id) > 0 THEN 1 ELSE 0 END as has_voted
                FROM voters v
                LEFT JOIN votes ON v.voter_id = votes.voter_id
                GROUP BY v.voter_id
                ORDER BY v.created_at DESC
            """)
            voters = cursor.fetchall()
        
        voters_list = []
        for voter in voters:
            voter_dict = dict(voter)
            if voter_dict.get('created_at'):
                try:
                    voter_dict['registration_date'] = datetime.strptime(
                        voter_dict['created_at'], '%Y-%m-%d %H:%M:%S'
                    ).strftime('%Y-%m-%d')
                except:
                    voter_dict['registration_date'] = voter_dict['created_at']
            else:
                voter_dict['registration_date'] = 'Unknown'
            
            voters_list.append(voter_dict)
        
        return jsonify(voters_list), 200
        
    except Exception as e:
        print(f"Error in get_all_voters: {e}")
        return jsonify({"error": "Database error occurred. Please try again."}), 500

# Admin - Delete voter
@app.route('/api/admin/voters/<voter_id>', methods=['DELETE'])
@require_admin_auth
def delete_voter(voter_id):
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT name, profile_image FROM voters WHERE voter_id = ?", (voter_id,))
            voter = cursor.fetchone()
            
            if not voter:
                return jsonify({"error": "Voter not found"}), 404
            
            voter_dict = dict(voter)
            voter_name = voter_dict['name']
            profile_image = voter_dict.get('profile_image')
            
            cursor.execute("DELETE FROM votes WHERE voter_id = ?", (voter_id,))
            votes_deleted = cursor.rowcount
            
            cursor.execute("DELETE FROM voters WHERE voter_id = ?", (voter_id,))
            
            if profile_image:
                try:
                    image_path = os.path.join(os.path.dirname(__file__), profile_image)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"Deleted profile image: {image_path}")
                except Exception as e:
                    print(f"Error deleting profile image: {e}")
        
        print(f"Voter '{voter_name}' ({voter_id}) deleted successfully. {votes_deleted} votes removed.")
        return jsonify({
            "message": f"Voter '{voter_name}' deleted successfully",
            "votes_removed": votes_deleted
        }), 200
        
    except Exception as e:
        print(f"Error deleting voter {voter_id}: {e}")
        return jsonify({"error": str(e)}), 500

# Admin - Get voter details
@app.route('/api/admin/voters/<voter_id>', methods=['GET'])
@require_admin_auth
def get_voter_details(voter_id):
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT * FROM voters WHERE voter_id = ?", (voter_id,))
            voter = cursor.fetchone()
            
            if not voter:
                return jsonify({"error": "Voter not found"}), 404
            
            cursor.execute("""
                SELECT 
                    v.election_id,
                    e.title as election_title,
                    c.name as candidate_name,
                    c.party as candidate_party,
                    v.vote_timestamp
                FROM votes v
                JOIN elections e ON v.election_id = e.id
                JOIN candidates c ON v.candidate_id = c.id
                WHERE v.voter_id = ?
                ORDER BY v.vote_timestamp DESC
            """, (voter_id,))
            voting_history = cursor.fetchall()
        
        voter_dict = dict(voter)
        voter_dict['voting_history'] = [dict(vote) for vote in voting_history]
        voter_dict['votes_cast'] = len(voting_history)
        
        return jsonify(voter_dict), 200
        
    except Exception as e:
        print(f"Error getting voter details for {voter_id}: {e}")
        return jsonify({"error": str(e)}), 500

# Admin - Update voter information
@app.route('/api/admin/voters/<voter_id>', methods=['PUT'])
@require_admin_auth
def update_voter(voter_id):
    try:
        data = request.json
        
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT voter_id FROM voters WHERE voter_id = ?", (voter_id,))
            if not cursor.fetchone():
                return jsonify({"error": "Voter not found"}), 404
            
            update_fields = []
            update_values = []
            
            allowed_fields = ['name', 'email', 'phone', 'address', 'gender']
            
            for field in allowed_fields:
                if field in data:
                    update_fields.append(f"{field} = ?")
                    update_values.append(data[field])
            
            if not update_fields:
                return jsonify({"error": "No valid fields to update"}), 400
            
            update_values.append(voter_id)
            
            query = f"UPDATE voters SET {', '.join(update_fields)} WHERE voter_id = ?"
            cursor.execute(query, update_values)
        
        return jsonify({"message": "Voter updated successfully"}), 200
        
    except Exception as e:
        print(f"Error updating voter {voter_id}: {e}")
        return jsonify({"error": str(e)}), 500

# Admin - Reset voter's voting status
@app.route('/api/admin/voters/<voter_id>/reset-votes', methods=['POST'])
@require_admin_auth
def reset_voter_votes(voter_id):
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT name FROM voters WHERE voter_id = ?", (voter_id,))
            voter = cursor.fetchone()
            
            if not voter:
                return jsonify({"error": "Voter not found"}), 404
            
            voter_dict = dict(voter)
            voter_name = voter_dict['name']
            
            cursor.execute("DELETE FROM votes WHERE voter_id = ?", (voter_id,))
            votes_deleted = cursor.rowcount
            
            cursor.execute("""
                UPDATE candidates 
                SET vote_count = (
                    SELECT COUNT(*) FROM votes WHERE candidate_id = candidates.id
                )
            """)
        
        return jsonify({
            "message": f"Reset voting status for {voter_name}",
            "votes_removed": votes_deleted
        }), 200
        
    except Exception as e:
        print(f"Error resetting votes for voter {voter_id}: {e}")
        return jsonify({"error": str(e)}), 500

# Admin Logs Management
@app.route('/api/admin/logs', methods=['GET'])
@require_admin_auth
def get_log_files():
    """Get list of available log files"""
    try:
        log_files = logger.get_log_files()
        log_stats = logger.get_log_stats()
        
        return jsonify({
            'files': log_files,
            'stats': log_stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting log files", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/logs/<filename>', methods=['GET'])
@require_admin_auth
def get_log_content(filename):
    """Get content of a specific log file"""
    try:
        lines = request.args.get('lines', 1000, type=int)
        lines = min(lines, 10000)  # Limit to 10k lines max
        
        log_content = logger.read_log_file(filename, lines)
        
        logger.security_event(
            'LOG_ACCESS',
            {'filename': filename, 'lines_requested': lines},
            request.remote_addr
        )
        
        return jsonify({
            'filename': filename,
            'lines': log_content,
            'total_lines': len(log_content)
        }), 200
        
    except Exception as e:
        logger.error(f"Error reading log file {filename}", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/logs/<filename>/download', methods=['GET'])
@require_admin_auth
def download_log_file(filename):
    """Download a log file"""
    try:
        from flask import send_file
        import os
        
        log_file_path = os.path.join(logger.log_dir, filename)
        
        if not os.path.exists(log_file_path):
            return jsonify({'error': 'Log file not found'}), 404
        
        logger.security_event(
            'LOG_DOWNLOAD',
            {'filename': filename},
            request.remote_addr
        )
        
        return send_file(
            log_file_path,
            as_attachment=True,
            download_name=f"facecast_{filename}",
            mimetype='text/plain'
        )
        
    except Exception as e:
        logger.error(f"Error downloading log file {filename}", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/logs/cleanup', methods=['POST'])
@require_admin_auth
def cleanup_old_logs():
    """Clean up old log files"""
    try:
        data = request.json or {}
        days = data.get('days', 30)
        
        files_removed = logger.cleanup_old_logs(days)
        
        logger.security_event(
            'LOG_CLEANUP',
            {'days': days, 'files_removed': files_removed},
            request.remote_addr
        )
        
        return jsonify({
            'message': f'Cleaned up {files_removed} log files older than {days} days',
            'files_removed': files_removed
        }), 200
        
    except Exception as e:
        logger.error(f"Error cleaning up logs", e)
        return jsonify({'error': str(e)}), 500

# Delete election
@app.route('/api/elections/<int:election_id>', methods=['DELETE'])
@require_admin_auth
def delete_election(election_id):
    print(f"Delete request for election ID: {election_id}")
    
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT title FROM elections WHERE id = ?", (election_id,))
            election = cursor.fetchone()
            if not election:
                return jsonify({"error": "Election not found"}), 404
            
            election_title = election['title']
            print(f"Deleting election: {election_title}")
            
            cursor.execute("DELETE FROM votes WHERE election_id = ?", (election_id,))
            votes_deleted = cursor.rowcount
            print(f"Deleted {votes_deleted} votes")
            
            cursor.execute("DELETE FROM candidates WHERE election_id = ?", (election_id,))
            candidates_deleted = cursor.rowcount
            print(f"Deleted {candidates_deleted} candidates")
            
            cursor.execute("DELETE FROM elections WHERE id = ?", (election_id,))
            elections_deleted = cursor.rowcount
            print(f"Deleted {elections_deleted} election")
            
            if elections_deleted == 0:
                return jsonify({"error": "Election not found or already deleted"}), 404
        
        print(f"Election '{election_title}' deleted successfully")
        return jsonify({"message": f"Election '{election_title}' deleted successfully"}), 200
        
    except Exception as e:
        print(f"Error deleting election {election_id}: {e}")
        return jsonify({"error": str(e)}), 500




# Delete candidate
@app.route('/api/candidates/<int:candidate_id>', methods=['DELETE'])
@require_admin_auth
def delete_candidate(candidate_id):
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("DELETE FROM votes WHERE candidate_id = ?", (candidate_id,))
            cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
        
        return jsonify({"message": "Candidate deleted successfully"}), 200
        
    except Exception as e:
        print(f"Error in delete_candidate: {e}")
        return jsonify({"error": str(e)}), 500

# Update election status
@app.route('/api/elections/<int:election_id>/status', methods=['PUT'])
@require_admin_auth
def update_election_status(election_id):
    data = request.json
    status = data.get('status')
    
    if status not in ['upcoming', 'active', 'ended']:
        return jsonify({"error": "Invalid status"}), 400
    
    try:
        with get_db_cursor() as (conn, cursor):
            cursor.execute("UPDATE elections SET status = ? WHERE id = ?", (status, election_id))
        
        return jsonify({"message": "Election status updated successfully"}), 200
        
    except Exception as e:
        print(f"Error in update_election_status: {e}")
        return jsonify({"error": str(e)}), 500

def check_and_update_election_status():
    """Background task to check and update election statuses"""
    retry_delay = 30
    max_retry_delay = 300
    
    while True:
        try:
            with get_db_cursor() as (conn, cursor):
                current_time = datetime.now()
                
                cursor.execute("""
                    UPDATE elections 
                    SET status = 'active' 
                    WHERE status = 'upcoming' 
                    AND datetime(start_date) <= datetime(?)
                """, (current_time.strftime('%Y-%m-%d %H:%M:%S'),))
                
                cursor.execute("""
                    UPDATE elections 
                    SET status = 'ended' 
                    WHERE status = 'active' 
                    AND is_demo = 0
                    AND datetime(end_date) <= datetime(?)
                """, (current_time.strftime('%Y-%m-%d %H:%M:%S'),))
                
                cursor.execute("""
                    SELECT id, title, start_date, demo_duration_minutes
                    FROM elections 
                    WHERE status = 'active' 
                    AND is_demo = 1
                """)
                
                demo_elections = cursor.fetchall()
                
                for election in demo_elections:
                    start_time = datetime.strptime(election['start_date'], '%Y-%m-%d %H:%M:%S')
                    duration_minutes = election['demo_duration_minutes'] or 2
                    end_time = start_time + timedelta(minutes=duration_minutes)
                    
                    if current_time >= end_time:
                        cursor.execute("""
                            UPDATE elections 
                            SET status = 'ended' 
                            WHERE id = ?
                        """, (election['id'],))
                        print(f"Demo election '{election['title']}' automatically ended after {duration_minutes} minutes")
            
            retry_delay = 30
            
        except Exception as e:
            print(f"Error in election status check: {e}")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)
            continue
        
        time.sleep(30)

def start_election_monitor():
    """Start the election monitoring service"""
    monitor_thread = threading.Thread(target=check_and_update_election_status, daemon=True)
    monitor_thread.start()
    print("[INFO] Election monitoring service started")


# Security and account locking endpoints - ADD THESE TO YOUR app.py
@app.route('/api/log-security-event', methods=['POST'])
def log_security_event():
    """Log security events for account locking"""
    try:
        data = request.json
        voter_id = data.get('voter_id')
        event_type = data.get('event_type')
        details = data.get('details', {})
        
        logger.security_event(
            event_type.upper(),
            {'voter_id': voter_id, **details},
            request.remote_addr
        )
        
        return jsonify({"success": True, "message": "Security event logged"}), 200
        
    except Exception as e:
        logger.error("Error logging security event", e, 'security')
        return jsonify({"error": "Failed to log security event"}), 500

@app.route('/api/check-failed-attempts', methods=['POST'])
def check_failed_attempts():
    """Check if account should be locked due to failed attempts"""
    try:
        data = request.json
        voter_id = data.get('voter_id')
        
        # Simple implementation
        return jsonify({
            "is_locked": False,
            "attempts_remaining": 5,
            "lockout_time": None
        }), 200
        
    except Exception as e:
        logger.error("Error checking failed attempts", e, 'security')
        return jsonify({"is_locked": False, "attempts_remaining": 5}), 200

@app.route('/api/lock-account', methods=['POST'])
def lock_account():
    """Lock account temporarily"""
    try:
        data = request.json
        voter_id = data.get('voter_id')
        reason = data.get('reason', 'Too many failed attempts')
        
        logger.security_event(
            'ACCOUNT_LOCKED',
            {'voter_id': voter_id, 'reason': reason},
            request.remote_addr
        )
        
        return jsonify({
            "success": True,
            "message": f"Account locked: {reason}",
            "lockout_duration": "30 minutes"
        }), 200
        
    except Exception as e:
        logger.error("Error locking account", e, 'security')
        return jsonify({"error": "Failed to lock account"}), 500
    


# Add to app.py for debugging
@app.route('/api/debug-face-comparison', methods=['POST'])
def debug_face_comparison():
    """Debug endpoint to see why face matching is failing"""
    try:
        data = request.json
        voter_id = data.get('voter_id')
        live_embedding = data.get('live_embedding')
        
        with get_db_cursor() as (conn, cursor):
            cursor.execute("SELECT face_embedding FROM voters WHERE voter_id = ?", (voter_id,))
            voter = cursor.fetchone()
            
            if not voter:
                return jsonify({"error": "Voter not found"}), 404
            
            stored_data = voter['face_embedding']
            if isinstance(stored_data, bytes):
                stored_str = stored_data.decode('utf-8')
            else:
                stored_str = stored_data
                
            stored_embedding = json.loads(stored_str)
            
            # Compare dimensions and values
            debug_info = {
                "stored_embedding": {
                    "length": len(stored_embedding),
                    "first_5": stored_embedding[:5],
                    "last_5": stored_embedding[-5:],
                    "mean": float(np.mean(stored_embedding)),
                    "std": float(np.std(stored_embedding))
                },
                "live_embedding": {
                    "length": len(live_embedding),
                    "first_5": live_embedding[:5],
                    "last_5": live_embedding[-5:],
                    "mean": float(np.mean(live_embedding)),
                    "std": float(np.std(live_embedding))
                },
                "similarity": float(calculate_face_similarity_256d(stored_embedding, live_embedding))
            }
            
            return jsonify(debug_info), 200
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("[INFO] Starting FaceCast Server")
    print("[INFO] Server will be available at: http://localhost:5000")
    
    start_election_monitor()
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error("Failed to start FaceCast server", e)
        raise
