# 🏥 CareSync-AI: Smart Patient Onboarding & Hospital MRD Portal

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask_3.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![RAG Architecture](https://img.shields.io/badge/Retrieval-Hybrid_RAG_(ChromaDB_%2B_BM25)-purple.svg)]()
[![Digital Signatures](https://img.shields.io/badge/Security-SHA--256_Cryptographic_Seals-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> A streamlined, full-stack healthcare documentation and health insurance claim management platform connecting **Discharged Patients**, the **Hospital Medical Records Department (MRD)**, and **Hospital Super Administrators** in real-time.

---

## 📌 Problem Statement & Overview

In conventional hospital workflows:
- Discharged patients must physically return to hospital desks to collect missing discharge summaries, itemized bills, and surgeon OT notes required for insurance reimbursement.
- Communication with records officers is slow, opaque, and unverified, leading to claim delays or rejections by TPAs (Third-Party Administrators).
- Documents are prone to tampering or lack tamper-evident digital verification keys.

**CareSync-AI** eliminates physical hospital visits by providing a multi-stakeholder web platform featuring **Hybrid RAG retrieval**, **AI claim entity extraction**, **SHA-256 cryptographic document certification**, and **real-time direct patient-to-hospital chat**.

---

## 🌟 Key Features

### 👤 1. Discharged Patient Portal
- **Self-Service Onboarding**: Register with Patient Name, Inpatient (IP) / Discharge ID, Email, and Password.
- **Natural Language Claim Requests**: Describe admission details in plain English (e.g. procedure, insurer, doctor, bills).
- **Auto-Generated Document Checklists**: Dynamic checklists showing requirements for Star Health, HDFC Ergo, Care Health, etc.
- **Tamper-Proof Verification Tracking**: Real-time status badges showing when documents are certified and sealed with cryptographic verification hashes (`MRD-SHA256`).
- **Direct Live MRD Chat**: Instant messaging directly with the hospital's medical records desk.

### 🏥 2. Hospital MRD (Medical Records Department) Portal
- **Live Requisition Board**: Real-time incoming board of all patient claim requests, policy numbers, and hospital details.
- **One-Click Digital Seal (`✍ Seal`)**: Issues tamper-evident SHA-256 cryptographic hashes for discharge summaries, itemized pharmacy bills, and OT notes.
- **Document Upload & Certification**: Upload official signed PDFs/scans directly to patient requisition files.
- **Patient Communication Desk**: Real-time threaded replies to patient questions regarding specific reports.

### 🛡️ 3. Super Admin Audit Portal
- **Live Metrics Dashboard**: Total registered patients, active claims/requisitions raised, and database status.
- **Registered User Directory**: Search and audit all user accounts across patients, MRD staff, and administrators.
- **Requisition Tracking Directory**: Review all active and completed document requests.
- **Real-Time Communication Auditor**: Select any active patient conversation thread to inspect live message exchanges between patients and hospital staff.

---

## 🏛️ System Architecture

```text
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│ Discharged Patient Desk │  │    Hospital MRD Desk    │  │   Super Admin Auditor   │
│  (Onboard, Request,     │  │ (Audit, Seal, Upload,   │  │   (User Registry, Live  │
│   Download, Chat)       │  │       Reply Chat)       │  │     Chat Monitoring)    │
└────────────┬────────────┘  └────────────┬────────────┘  └────────────┬────────────┘
             │                            │                            │
             └────────────────────────────┼────────────────────────────┘
                                          │
                                   HTTP REST APIs
                                          │
                                          ▼
                             Flask Backend (flask_app.py)
                                          │
               ┌──────────────────────────┴──────────────────────────┐
               ▼                                                     ▼
     Relational Storage                                    Hybrid RAG Engine
    (SQLite / caresync.db)                               - ChromaDB (Semantic Vector Search)
   - Users (Patients, MRD, Admin)                        - BM25 (Exact Keyword Match)
   - Claims & Requisitions Checklists                    - Google Gemini 1.5 Flash (AI Extraction)
   - Real-Time Message Threads                           - Fast Regex Local Fallback
```

---

## 🧠 How the Hybrid RAG Architecture Works

When a patient describes their admission details in free-text:
1. **Semantic Search (ChromaDB)**: Converts the patient query into vector embeddings to find semantically relevant insurer policies (e.g., matching medical conditions to claim rules).
2. **Keyword Match (BM25)**: Evaluates exact term frequencies for insurance companies (Star Health, HDFC Ergo, IRDAI timelines).
3. **Prompt Augmentation**: Merges and deduplicates the top retrieved guidelines and augments the LLM prompt.
4. **Structured Entity Extraction**: Google Gemini extracts a clean JSON payload (`patient_name`, `insurer`, `policy_number`, `hospital`, `diagnosis`, `bill_amount`) with zero manual form-filling.

---

## 🚀 Getting Started (Local Setup)

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/CareSync-AI.git
cd CareSync-AI
```

### 3. Create & Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
*(Optional: Add your `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey) for live LLM extraction. The app runs fine without it using local regex extraction).*

### 6. Run the Application
```bash
python flask_app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 🧪 Demo & Testing Accounts (For Evaluators)

> [!NOTE]
> The following pre-seeded credentials are mock test accounts intended solely for evaluating and demonstrating the portal's role-based workflows. Real secrets and API keys should always be loaded via `.env` and environment variables.

| Role | Portal Tab | Email | Password | Access Level |
| :--- | :--- | :--- | :--- | :--- |
| **Discharged Patient** | Patient Login | `raja@hospital.com` | `pass123` | Request records, live chat, download sealed files |
| **Hospital MRD Staff** | MRD Staff | `mrd@hospital.com` | `password123` | Digital seal (SHA-256), upload PDFs, reply to chats |
| **Super Admin** | Admin | `admin@hospital.com` | `admin123` | System audit, user directory, message monitor |

*(New visitors can also self-register at any time using the **Signup** tab).*

---

## ☁️ Deployment (Render)

This repository includes a pre-configured `render.yaml` ready for one-click deployment on Render:

1. Push this repository to your **GitHub** account.
2. Go to **[dashboard.render.com](https://dashboard.render.com)** and create a new **Web Service**.
3. Connect your GitHub repository.
4. Set the following build options:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn flask_app:app`
5. *(Optional)* Under **Environment Variables**, add `GOOGLE_API_KEY`.
6. Click **Deploy Web Service** to launch your live instance.

---

## 🛠️ Tech Stack

- **Backend**: Python 3, Flask, Gunicorn
- **Database**: SQLite (ACID-compliant relational storage)
- **AI / NLP**: Google Gemini 1.5 Flash, Google Generative AI SDK
- **RAG & Search**: ChromaDB (Vector Search), `rank-bm25` (BM25Okapi)
- **Security**: SHA-256 Cryptographic Hash Generation, Strict Role-Based Access Control (RBAC)
- **Frontend**: HTML5, Vanilla JavaScript (ES6+), Bootstrap 5.3 (Dark Theme)

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
