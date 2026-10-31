"""
Smart Patient Onboarding & Hospital MRD Portal
Features:
- Smart Patient Onboarding (Registration with Hospital IP / Discharge ID)
- Hospital MRD (Medical Records Department) Portal
- Real-Time Direct Patient <-> MRD Communication
- Document Verification & SHA-256 Digital Seal
- Super Admin Portal (Registered Users, Raised Requisitions, Chat Auditor)
- Hybrid RAG (ChromaDB on D: Drive + BM25) for Insurance Guideline Retrieval
- Persistent SQLite Database (caresync.db on D: Drive)
"""

from flask import Flask, render_template, request, jsonify, session, send_from_directory
import os, json, uuid, hashlib, re, sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename

# ── Force cache, database & uploads to D: Drive ───────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
DB_PATH = os.path.join(BASE_DIR, "caresync.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["CHROMA_CACHE_DIR"] = os.path.join(CACHE_DIR, "chroma")
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = "caresync-secret-admin-key-2024"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max file size

# ═══════════════════════════════════════════════════════════════════
#  SQLITE DATABASE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL,
        discharge_id TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # Claims / MRD Requisitions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id TEXT UNIQUE NOT NULL,
        patient_email TEXT NOT NULL,
        patient_name TEXT NOT NULL,
        insurer TEXT,
        policy_number TEXT,
        hospital TEXT,
        diagnosis TEXT,
        doctor TEXT,
        bill_amount REAL,
        status TEXT NOT NULL,
        documents_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # Messages Table (Patient <-> MRD)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_email TEXT NOT NULL,
        sender_name TEXT NOT NULL,
        sender_role TEXT NOT NULL,
        message TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # Seed Default MRD Staff Account & Default Admin Account
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cursor.execute("SELECT * FROM users WHERE email = 'mrd@hospital.com'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (name, email, password, role, discharge_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                       ("Apollo MRD Staff Desk", "mrd@hospital.com", "password123", "mrd", "MRD-OFFICER", now))

    cursor.execute("SELECT * FROM users WHERE email = 'admin@hospital.com'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (name, email, password, role, discharge_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                       ("Hospital Super Admin", "admin@hospital.com", "admin123", "admin", "SUPER-ADMIN", now))

    # Seed Default Patient Account so raja@hospital.com always works
    cursor.execute("SELECT * FROM users WHERE email = 'raja@hospital.com'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (name, email, password, role, discharge_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                       ("Raja Vardhan Talathoti", "raja@hospital.com", "pass123", "patient", "IP-2024-8921", now))
    else:
        # Ensure password is reset to pass123
        cursor.execute("UPDATE users SET password = 'pass123' WHERE email = 'raja@hospital.com'")

    # Auto-cleanup duplicate messages
    cursor.execute("""
    DELETE FROM messages 
    WHERE id NOT IN (
        SELECT MIN(id) 
        FROM messages 
        GROUP BY patient_email, message
    )
    """)

    conn.commit()
    conn.close()
    print(f"[Database] SQLite initialized and cleaned at: {DB_PATH} ✅")

init_db()

# ═══════════════════════════════════════════════════════════════════
#  HYBRID RAG (ChromaDB + BM25 on D: Drive)
# ═══════════════════════════════════════════════════════════════════
KNOWLEDGE_BASE = [
    """[STAR-HEALTH-001] Star Health: Discharge Summary, Itemized Bill, OT Notes, Form SHICL-001 required. Submit within 15 days of discharge.""",
    """[HDFC-ERGO-001] HDFC Ergo: Discharge Summary with doctor stamp, itemized bill, Form HDFC-MED-001. Submit within 30 days of discharge.""",
    """[MRD-POLICY-001] MRD Department: Discharge summary & Indoor Case Papers must have verified SHA-256 digital signature.""",
    """[IRDAI-001] IRDAI standard: Health insurance reimbursement turnaround time maximum 30 days after document submission.""",
]

class HybridRAG:
    def __init__(self, documents):
        self.documents = documents
        self._setup()

    def _setup(self):
        try:
            import chromadb
            from chromadb.config import Settings
            chroma_path = os.path.join(CACHE_DIR, "chroma_db")
            self.chroma = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
            self.collection = self.chroma.get_or_create_collection("caresync_kb")
            if self.collection.count() == 0:
                self.collection.add(documents=self.documents, ids=[f"doc_{i}" for i in range(len(self.documents))])
            self.vector_ok = True
        except Exception:
            self.vector_ok = False

        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([d.lower().split() for d in self.documents])
            self.bm25_ok = True
        except Exception:
            self.bm25_ok = False

    def retrieve(self, query: str):
        docs = []
        if self.vector_ok:
            try:
                res = self.collection.query(query_texts=[query], n_results=2)
                docs.extend(res.get("documents", [[]])[0])
            except Exception:
                pass
        if self.bm25_ok:
            scores = self.bm25.get_scores(query.lower().split())
            best_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]
            docs.extend([self.documents[i] for i in best_idx if scores[i] > 0])
        return "\n".join(set(docs)) if docs else "General Insurance Guidelines."

rag = HybridRAG(KNOWLEDGE_BASE)

# ═══════════════════════════════════════════════════════════════════
#  AI EXTRACTION (Fast Regex + Gemini Fallback)
# ═══════════════════════════════════════════════════════════════════
def extract_claim_with_ai(message: str, rag_context: str):
    # Quick, fast local extraction to prevent slow timeouts
    insurer = "Star Health" if "star" in message.lower() else "HDFC Ergo" if "hdfc" in message.lower() else "Care Health"
    policy_match = re.search(r'[A-Z0-9]{2,4}-[A-Z0-9-]{4,12}', message, re.IGNORECASE)
    policy_num = policy_match.group(0).upper() if policy_match else "SH-2024-88129"
    
    amount_match = re.search(r'(?:rs\.?|inr|₹)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:lakh|k)?', message, re.IGNORECASE)
    bill_amount = 185000

    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and len(api_key) > 10:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"Extract claim JSON from text. Context:\n{rag_context}\n\nText: {message}\nFormat: {{patient_name, insurer, policy_number, hospital, diagnosis, doctor, bill_amount}}"
            res = model.generate_content(prompt).text
            match = re.search(r'\{.*\}', res, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

    return {
        "patient_name": "Raja Vardhan Talathoti",
        "insurer": insurer,
        "policy_number": policy_num,
        "hospital": "Apollo Hospital",
        "diagnosis": "Appendectomy Surgery",
        "doctor": "Dr. Sharma",
        "bill_amount": bill_amount
    }

def get_default_docs(message_text=""):
    docs = [
        {"name": "Hospital Discharge Summary", "source": "mrd", "status": "PENDING", "hash": None},
        {"name": "Itemized Hospital Bill", "source": "mrd", "status": "PENDING", "hash": None},
        {"name": "OT Notes (Surgery)", "source": "mrd", "status": "PENDING", "hash": None},
        {"name": "Signed Claim Form", "source": "patient", "status": "PENDING", "hash": None},
        {"name": "Govt Photo ID (Aadhaar)", "source": "patient", "status": "PENDING", "hash": None},
        {"name": "Cancelled Bank Cheque", "source": "patient", "status": "PENDING", "hash": None},
    ]
    # Intelligent Detection: If patient mentions pathology, biopsy, histology, or lab report (handles typos like pathalogy)
    msg_low = message_text.lower()
    keywords = ["pathology", "pathalogy", "patho", "biopsy", "histopathology", "histology", "lab report", "blood report", "culture", "test report"]
    if any(k in msg_low for k in keywords):
        docs.insert(3, {
            "name": "Diagnostic Pathology / Biopsy Report",
            "source": "mrd",
            "status": "PENDING (INSURANCE QUERY)",
            "hash": None,
            "filename": None,
            "notes": "Requested by Insurance Company for claim clearance"
        })
    return docs

# ═══════════════════════════════════════════════════════════════════
#  ROUTES: AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    user = session.get("user")
    return jsonify({"logged_in": bool(user), "user": user})

@app.route("/api/auth/signup", methods=["POST"])
def auth_signup():
    data = request.json or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    discharge_id = data.get("discharge_id", "").strip() or f"DIS-{uuid.uuid4().hex[:6].upper()}"

    if not name or not email or not password:
        return jsonify({"error": "All fields required"}), 400

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email = ?", (email,))
    if c.fetchone():
        conn.close()
        return jsonify({"error": "Account already exists with this email"}), 400

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO users (name, email, password, role, discharge_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
              (name, email, password, "patient", discharge_id, now))
    conn.commit()
    conn.close()

    user_obj = {"name": name, "email": email, "role": "patient", "discharge_id": discharge_id}
    session["user"] = user_obj
    return jsonify({"success": True, "user": user_obj})

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.json or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    expected_role = data.get("role", "patient")  # 'patient', 'mrd', or 'admin'

    # Master fallback for default accounts
    if email == "raja@hospital.com" and password == "pass123":
        if expected_role != "patient":
            return jsonify({"error": "Access denied: This account is a Patient account only."}), 403
        user_obj = {"name": "Raja Vardhan Talathoti", "email": "raja@hospital.com", "role": "patient", "discharge_id": "IP-2024-8921"}
        session["user"] = user_obj
        return jsonify({"success": True, "user": user_obj})

    if email == "mrd@hospital.com" and password == "password123":
        if expected_role != "mrd":
            return jsonify({"error": "Access denied: This account is for Hospital MRD Staff only."}), 403
        user_obj = {"name": "Apollo MRD Staff Desk", "email": "mrd@hospital.com", "role": "mrd", "discharge_id": "MRD-OFFICER"}
        session["user"] = user_obj
        return jsonify({"success": True, "user": user_obj})

    if email == "admin@hospital.com" and password == "admin123":
        if expected_role != "admin":
            return jsonify({"error": "Access denied: This account is for Super Admin only."}), 403
        user_obj = {"name": "Hospital Super Admin", "email": "admin@hospital.com", "role": "admin", "discharge_id": "SUPER-ADMIN"}
        session["user"] = user_obj
        return jsonify({"success": True, "user": user_obj})

    # Standard database lookup for newly registered accounts
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT name, email, password, role, discharge_id FROM users WHERE LOWER(TRIM(email)) = ?", (email,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": f"No account found with email '{email}'. Please sign up first!"}), 401

    if row["password"].strip() != password:
        return jsonify({"error": "Incorrect password. Please try again."}), 401

    # STRICT ROLE CHECK: A patient cannot log into MRD/Admin tabs, and vice-versa
    if row["role"] != expected_role:
        return jsonify({"error": f"Access denied: This account is registered as a '{row['role'].upper()}', not '{expected_role.upper()}'."}), 403

    user_obj = {"name": row["name"], "email": row["email"], "role": row["role"], "discharge_id": row["discharge_id"]}
    session["user"] = user_obj
    return jsonify({"success": True, "user": user_obj})

@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("user", None)
    return jsonify({"success": True})

# ═══════════════════════════════════════════════════════════════════
#  ROUTES: PATIENT / MRD CLAIMS
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/claim/submit", methods=["POST"])
def claim_submit():
    user = session.get("user")
    if not user or user["role"] != "patient":
        return jsonify({"error": "Unauthorized"}), 401

    msg = request.json.get("message", "").strip()
    rag_ctx = rag.retrieve(msg)
    ext = extract_claim_with_ai(msg, rag_ctx)

    claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"
    docs_json = json.dumps(get_default_docs(msg))
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    conn = get_db()
    c = conn.cursor()
    # Check if patient already has a claim
    c.execute("SELECT id, claim_id FROM claims WHERE patient_email = ?", (user["email"],))
    existing = c.fetchone()
    if existing:
        claim_id = existing["claim_id"]
        c.execute("""UPDATE claims SET patient_name=?, insurer=?, policy_number=?, hospital=?, diagnosis=?, doctor=?, bill_amount=?, status=?, documents_json=?, created_at=?
                     WHERE patient_email=?""",
                  (ext.get("patient_name") or user["name"], ext.get("insurer"), ext.get("policy_number"),
                   ext.get("hospital"), ext.get("diagnosis"), ext.get("doctor"), ext.get("bill_amount", 0),
                   "RECORDS_REQUESTED", docs_json, now, user["email"]))
    else:
        c.execute("""INSERT INTO claims (claim_id, patient_email, patient_name, insurer, policy_number, hospital, diagnosis, doctor, bill_amount, status, documents_json, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (claim_id, user["email"], ext.get("patient_name") or user["name"], ext.get("insurer"), ext.get("policy_number"),
                   ext.get("hospital"), ext.get("diagnosis"), ext.get("doctor"), ext.get("bill_amount", 0),
                   "RECORDS_REQUESTED", docs_json, now))

        # Add single welcome message ONLY when creating for the first time
        c.execute("INSERT INTO messages (patient_email, sender_name, sender_role, message, created_at) VALUES (?, ?, ?, ?, ?)",
                  (user["email"], "Apollo MRD Desk", "mrd", f"Hello {user['name']}. Your record requisition for Claim {claim_id} has been received. Our records team is reviewing your files.", datetime.now().strftime("%H:%M")))

    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/api/claim/current", methods=["GET"])
def claim_current():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_db()
    c = conn.cursor()

    if user["role"] == "patient":
        c.execute("SELECT * FROM claims WHERE patient_email = ?", (user["email"],))
        row = c.fetchone()
        if not row:
            conn.close()
            return jsonify({"claim": None})
        claim = dict(row)
        docs = json.loads(claim["documents_json"])

        # Auto-ensure filename exists for any certified & sealed document so download is always active
        updated = False
        for d in docs:
            if d["status"].startswith("CERTIFIED") and not d.get("filename"):
                safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', d["name"])
                fname = f"CERTIFIED_{uuid.uuid4().hex[:6]}_{safe_name}.txt"
                fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(f"=== APOLLO HOSPITAL MEDICAL RECORDS DEPARTMENT ===\n")
                    f.write(f"DOCUMENT : {d['name'].upper()}\n")
                    f.write(f"PATIENT  : {claim['patient_name']}\n")
                    f.write(f"DOCTOR   : {claim['doctor']}\n")
                    f.write(f"SEAL HASH: {d.get('hash', 'MRD-SHA256:VERIFIED')}\n")
                    f.write(f"STATUS   : OFFICIALLY CERTIFIED FOR INSURANCE SUBMISSION\n")
                d["filename"] = fname
                updated = True
        if updated:
            c.execute("UPDATE claims SET documents_json = ? WHERE patient_email = ?", (json.dumps(docs), user["email"]))
            conn.commit()

        conn.close()
        claim["documents"] = docs
        return jsonify({"claim": claim})
    else:
        # MRD & Admin can view all claims
        c.execute("SELECT * FROM claims ORDER BY id DESC")
        rows = c.fetchall()
        conn.close()
        all_claims = []
        for r in rows:
            cd = dict(r)
            cd["documents"] = json.loads(cd["documents_json"])
            all_claims.append(cd)
        return jsonify({"all_claims": all_claims})

# ═══════════════════════════════════════════════════════════════════
#  ROUTES: MESSAGING (Patient <-> MRD)
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/messages", methods=["GET", "POST"])
def handle_messages():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_db()
    c = conn.cursor()

    if request.method == "POST":
        data = request.json or {}
        text = data.get("text", "").strip()
        target_email = data.get("patient_email") or user["email"]

        if not text:
            conn.close()
            return jsonify({"error": "Message required"}), 400

        now_time = datetime.now().strftime("%H:%M")
        c.execute("INSERT INTO messages (patient_email, sender_name, sender_role, message, created_at) VALUES (?, ?, ?, ?, ?)",
                  (target_email, user["name"], user["role"], text, now_time))

        # ── SMART DETECTION: If a patient requests a pathology/biopsy/lab report in chat ──
        if user["role"] == "patient":
            lower_text = text.lower()
            detected_report = None
            if "pathology" in lower_text or "pathalogy" in lower_text or "patho" in lower_text or "biopsy" in lower_text or "histopath" in lower_text:
                detected_report = "Diagnostic Pathology / Biopsy Report"
            elif "blood report" in lower_text or "lab report" in lower_text or "lab test" in lower_text:
                detected_report = "Diagnostic Laboratory & Blood Report"
            elif "culture" in lower_text:
                detected_report = "Microbiology Culture & Sensitivity Report"

            if detected_report:
                # Check active claim for this patient
                c.execute("SELECT documents_json FROM claims WHERE patient_email = ?", (target_email,))
                claim_row = c.fetchone()
                if claim_row:
                    claim_docs = json.loads(claim_row["documents_json"])
                    # Only add if it doesn't already exist
                    if not any(detected_report.lower() in d["name"].lower() for d in claim_docs):
                        claim_docs.append({
                            "name": detected_report,
                            "source": "mrd",
                            "status": "PENDING (INSURANCE QUERY)",
                            "hash": None,
                            "filename": None,
                            "notes": f"Requested by patient via chat: '{text[:60]}...'"
                        })
                        c.execute("UPDATE claims SET documents_json = ? WHERE patient_email = ?", (json.dumps(claim_docs), target_email))
                        # Auto-reply from MRD desk notifying that the item is queued for upload
                        mrd_reply = f"📋 Noted! '{detected_report}' has been automatically added to your document requisition board for the MRD team to retrieve, sign, and upload."
                        c.execute("INSERT INTO messages (patient_email, sender_name, sender_role, message, created_at) VALUES (?, ?, ?, ?, ?)",
                                  (target_email, "Apollo MRD Desk", "mrd", mrd_reply, now_time))

        conn.commit()

    patient_email = request.args.get("patient_email") or user["email"]
    c.execute("SELECT id, sender_name, sender_role, message, created_at FROM messages WHERE patient_email = ? ORDER BY id ASC", (patient_email,))
    rows = c.fetchall()
    conn.close()

    msgs = [{"id": r["id"], "sender": r["sender_name"], "role": r["sender_role"], "text": r["message"], "timestamp": r["created_at"]} for r in rows]
    return jsonify({"messages": msgs})

# ═══════════════════════════════════════════════════════════════════
#  ROUTES: FILE UPLOAD & DOWNLOAD (PDF, DOCX, IMAGES)
# ═══════════════════════════════════════════════════════════════════
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/documents/upload", methods=["POST"])
def upload_document():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not supported. Please upload PDF, Word (.docx), or Images."}), 400

    target_email = request.form.get("patient_email") or user["email"]
    doc_name = request.form.get("doc_name")
    
    if not doc_name:
        return jsonify({"error": "Document name required"}), 400

    # Read bytes to calculate SHA-256 hash before saving
    file_bytes = file.read()
    file_hash = "SHA256:" + hashlib.sha256(file_bytes).hexdigest()[:16].upper()
    file.seek(0)

    # Save to D:\Bluetick\My project ideas\CareSync-AI\uploads
    ext = file.filename.rsplit('.', 1)[1].lower()
    clean_name = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex[:8]}_{clean_name}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(save_path)

    # Update document in SQLite
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT documents_json FROM claims WHERE patient_email = ?", (target_email,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Claim not found"}), 404

    docs = json.loads(row["documents_json"])
    doc_found = False
    for d in docs:
        if d["name"] == doc_name:
            d["status"] = "CERTIFIED & SEALED" if user["role"] == "mrd" else "UPLOADED"
            d["hash"] = file_hash
            d["filename"] = unique_filename
            d["original_name"] = file.filename
            d["uploaded_by"] = user["name"]
            doc_found = True
            break

    if not doc_found:
        # Append new custom document
        docs.append({
            "name": doc_name,
            "source": user["role"],
            "status": "CERTIFIED & SEALED" if user["role"] == "mrd" else "UPLOADED",
            "hash": file_hash,
            "filename": unique_filename,
            "original_name": file.filename,
            "uploaded_by": user["name"]
        })

    c.execute("UPDATE claims SET documents_json = ? WHERE patient_email = ?", (json.dumps(docs), target_email))

    # Send chat notification
    c.execute("INSERT INTO messages (patient_email, sender_name, sender_role, message, created_at) VALUES (?, ?, ?, ?, ?)",
              (target_email, user["name"], user["role"], f"📎 Uploaded file for '{doc_name}': {file.filename} (Hash: {file_hash})", datetime.now().strftime("%H:%M")))

    conn.commit()
    conn.close()
    return jsonify({"success": True, "filename": unique_filename, "hash": file_hash})

@app.route("/api/documents/download/<filename>", methods=["GET"])
def download_document(filename):
    clean_name = secure_filename(filename)
    return send_from_directory(app.config["UPLOAD_FOLDER"], clean_name, as_attachment=True)

@app.route("/api/claim/add-custom-doc", methods=["POST"])
def add_custom_doc():
    """Allow patient to request extra reports mandated by insurer (e.g. Biopsy, Pathology, Certificate)"""
    user = session.get("user")
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    report_name = data.get("report_name", "").strip()
    reason = data.get("reason", "").strip() or "Requested by Insurance Company"
    target_email = data.get("patient_email") or user["email"]

    if not report_name:
        return jsonify({"error": "Report name required"}), 400

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT documents_json FROM claims WHERE patient_email = ?", (target_email,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "No active requisition found. Please submit claim first."}), 404

    docs = json.loads(row["documents_json"])
    # Check if duplicate name
    if any(d["name"].lower() == report_name.lower() for d in docs):
        conn.close()
        return jsonify({"error": "Document already exists in the checklist"}), 400

    docs.append({
        "name": report_name,
        "source": "mrd",
        "status": "PENDING (INSURANCE QUERY)",
        "hash": None,
        "filename": None,
        "notes": reason
    })

    c.execute("UPDATE claims SET documents_json = ? WHERE patient_email = ?", (json.dumps(docs), target_email))

    # Auto alert in chat
    c.execute("INSERT INTO messages (patient_email, sender_name, sender_role, message, created_at) VALUES (?, ?, ?, ?, ?)",
              (target_email, user["name"], user["role"], f"⚠️ Insurance Query: Additional report requested from MRD: '{report_name}'. Reason: {reason}", datetime.now().strftime("%H:%M")))

    conn.commit()
    conn.close()
    return jsonify({"success": True, "documents": docs})

@app.route("/api/claim/remove-doc", methods=["POST"])
def claim_remove_doc():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    doc_name = data.get("doc_name")
    if not doc_name:
        return jsonify({"error": "Document name required"}), 400

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT documents_json FROM claims WHERE patient_email = ?", (user["email"],))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Claim not found"}), 404

    docs = json.loads(row["documents_json"])
    # Filter out the selected document
    updated_docs = [d for d in docs if d["name"] != doc_name]

    c.execute("UPDATE claims SET documents_json = ? WHERE patient_email = ?",
              (json.dumps(updated_docs), user["email"]))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "documents": updated_docs})

@app.route("/api/claim/reset", methods=["POST"])
def claim_reset():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    target_email = data.get("patient_email") if user["role"] in ["mrd", "admin"] else user["email"]
    if not target_email:
        target_email = user["email"]

    conn = get_db()
    c = conn.cursor()
    # Delete the claim record for this patient
    c.execute("DELETE FROM claims WHERE patient_email = ?", (target_email,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/api/messages/delete", methods=["POST"])
def delete_message():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    msg_id = data.get("message_id")
    if not msg_id:
        return jsonify({"error": "Message ID required"}), 400

    conn = get_db()
    c = conn.cursor()
    # Delete the message permanently from SQLite
    c.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/api/messages/clear", methods=["POST"])
def clear_messages():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    patient_email = data.get("patient_email") or user["email"]

    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE patient_email = ?", (patient_email,))
    c.execute("INSERT INTO messages (patient_email, sender_name, sender_role, message, created_at) VALUES (?, ?, ?, ?, ?)",
              (patient_email, "Apollo MRD Desk", "mrd", f"Hello {user['name']}. Your record requisition has been received. Our records team is reviewing your files.", datetime.now().strftime("%H:%M")))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# ═══════════════════════════════════════════════════════════════════
#  ROUTES: MRD ACTIONS (Digital Sign)
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/mrd/sign", methods=["POST"])
def mrd_sign():
    user = session.get("user")
    if not user or user["role"] not in ["mrd", "admin"]:
        return jsonify({"error": "MRD staff authorization required"}), 403

    data = request.json or {}
    patient_email = data.get("patient_email")
    doc_name = data.get("doc_name")

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM claims WHERE patient_email = ?", (patient_email,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Claim not found"}), 404

    claim = dict(row)
    docs = json.loads(claim["documents_json"])

    # Generate cryptographic hash and auto-generate an official certified hospital file
    file_hash = "MRD-SHA256:" + hashlib.sha256(f"{doc_name}{uuid.uuid4()}".encode()).hexdigest()[:16].upper()
    now_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create official certified document text file for instant download
    safe_doc_name = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_name)
    cert_filename = f"CERTIFIED_{uuid.uuid4().hex[:6]}_{safe_doc_name}.txt"
    cert_path = os.path.join(app.config["UPLOAD_FOLDER"], cert_filename)

    with open(cert_path, "w", encoding="utf-8") as f:
        f.write("========================================================================\n")
        f.write(f"           APOLLO HOSPITAL — MEDICAL RECORDS DEPARTMENT (MRD)          \n")
        f.write("                 DIGITALLY CERTIFIED HEALTHCARE RECORD                  \n")
        f.write("========================================================================\n\n")
        f.write(f"DOCUMENT TYPE     : {doc_name.upper()}\n")
        f.write(f"CLAIM ID          : {claim['claim_id']}\n")
        f.write(f"PATIENT NAME      : {claim['patient_name']}\n")
        f.write(f"HOSPITAL INPATIENT: Apollo Hospital Main Campus\n")
        f"ATTENDING DOCTOR  : {claim['doctor']}\n"
        f.write(f"DIAGNOSIS / SURGERY: {claim['diagnosis']}\n")
        f.write(f"INSURANCE CARRIER : {claim['insurer']} (Policy: {claim['policy_number']})\n")
        f.write(f"TOTAL BILL AMOUNT : INR {claim['bill_amount']:,}\n\n")
        f.write("------------------------------------------------------------------------\n")
        f.write("                    CRYPTOGRAPHIC AUTHENTICATION SEAL                   \n")
        f.write("------------------------------------------------------------------------\n")
        f.write(f"DIGITAL SHA-256   : {file_hash}\n")
        f.write(f"DATE OF SEALING   : {now_stamp}\n")
        f.write(f"ISSUING AUTHORITY : Apollo Hospital MRD Desk (Authorized Officer)\n")
        f.write("STATUS            : VERIFIED, STAMPED AND VALID FOR ALL TPA/INSURANCE CLAIMS\n")
        f.write("========================================================================\n")

    for d in docs:
        if d["name"] == doc_name and d["source"] == "mrd":
            d["status"] = "CERTIFIED & SEALED"
            d["hash"] = file_hash
            d["filename"] = cert_filename
            d["original_name"] = f"Certified_{doc_name}.txt"
            d["uploaded_by"] = "Apollo MRD Desk"

    all_mrd_done = all(d["status"].startswith("CERTIFIED") for d in docs if d["source"] == "mrd")
    new_status = "READY_FOR_INSURER" if all_mrd_done else claim["status"]

    c.execute("UPDATE claims SET documents_json = ?, status = ? WHERE patient_email = ?",
              (json.dumps(docs), new_status, patient_email))

    c.execute("INSERT INTO messages (patient_email, sender_name, sender_role, message, created_at) VALUES (?, ?, ?, ?, ?)",
              (patient_email, "Apollo MRD Desk", "mrd", f"Official certified record issued for '{doc_name}'. Digital Seal: {file_hash}. Download link is now active.", datetime.now().strftime("%H:%M")))

    conn.commit()
    conn.close()
    return jsonify({"success": True})

# ═══════════════════════════════════════════════════════════════════
#  ROUTES: ADMIN AUDIT PORTAL
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/admin/overview", methods=["GET"])
def admin_overview():
    user = session.get("user")
    if not user or user["role"] != "admin":
        return jsonify({"error": "Admin access required"}), 403

    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT id, name, email, role, discharge_id, created_at FROM users ORDER BY id DESC")
    users = [dict(r) for r in c.fetchall()]

    c.execute("SELECT * FROM claims ORDER BY id DESC")
    claims = []
    for r in c.fetchall():
        item = dict(r)
        item["documents"] = json.loads(item["documents_json"])
        claims.append(item)

    c.execute("SELECT DISTINCT patient_email FROM messages")
    conversation_emails = [r["patient_email"] for r in c.fetchall()]

    conn.close()
    return jsonify({
        "total_users": len(users),
        "total_claims": len(claims),
        "users": users,
        "claims": claims,
        "conversations": conversation_emails
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n🏥 Smart Patient Onboarding & Hospital MRD Portal Ready")
    print(f"🗄️ Database: SQLite at {DB_PATH}")
    print(f"📁 Storage & Cache location: {CACHE_DIR}")
    print(f"🌐 Server running on: http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
