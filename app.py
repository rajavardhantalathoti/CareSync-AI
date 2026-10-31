"""
app.py - Advanced Healthcare & Mediclaim Agentic Platform (Interactive Web UI)
=============================================================================
Run with:
    python app.py

Features for Recruiters:
- Multi-Stakeholder UI: Patient Portal, ER Doctor Desk, Hospital Records Desk, Insurer Adjudication
- Live LangGraph Execution with Memory Checkpoint
- Dual Human-In-The-Loop: Doctor Emergency Acknowledge + Hospital Desk E-Sign
- Hybrid RAG Document Retrieval & Rule-Based Fraud Detection Engine
"""

import os
import json
import uuid
import datetime
import gradio as gr
from langchain_core.messages import HumanMessage

from core_platform import (
    healthcare_agent_graph,
    UnifiedAgentState,
    ClaimFraudEngine,
    InsuranceClaimRecord,
    hybrid_rag
)

# In-memory shared state across demo sessions
GLOBAL_SESSION_ID = "demo-session-001"
CURRENT_STATE = {
    "messages": [],
    "intake_record": {
        "patient_name": "Rajesh Sharma",
        "age": 45,
        "gender": "Male",
        "primary_complaint": "Acute severe chest pain radiating to left jaw, sweating",
        "symptom_severity": 9,
        "triage_severity": "CRITICAL",
        "recommended_dept": "Emergency Cardiology / Cath Lab",
        "triage_reasoning": "Classic acute coronary syndrome indicators with radiation to left jaw and diaphoresis.",
        "escalated_to_doctor": True,
        "doctor_acknowledgment": None
    },
    "claim_record": {
        "claim_id": "CLM-A892F1",
        "patient_name": "Priya Sharma",
        "insurer_name": "Star Health",
        "policy_number": "SH-2024-88129",
        "claim_type": "HOSPITALIZATION",
        "hospital_name": "Apollo Super Specialty Hospital",
        "treating_doctor": "Dr. Ramesh Kumar (Cardiologist, Reg #MMC-4412)",
        "admission_date": "2024-09-20",
        "discharge_date": "2024-09-24",
        "diagnosis": "Acute Appendicitis with laparoscopic appendectomy",
        "total_bill_amount": 185000.0,
        "claim_amount": 185000.0,
        "fraud_risk_score": 14,
        "fraud_risk_tier": "LOW_RISK",
        "fraud_anomalies": ["None detected: Itemized billing and clinical stay match standard surgical guidelines."],
        "status": "PENDING_DOCUMENTS",
        "documents": [
            {
                "doc_id": "DOC-H01",
                "doc_name": "Hospital Discharge Summary",
                "doc_description": "Comprehensive admission to discharge clinical report with doctor signature",
                "is_mandatory": True,
                "provided_by": "hospital",
                "status": "REQUESTED_FROM_HOSPITAL",
                "file_path": None,
                "hospital_signature_hash": None
            },
            {
                "doc_id": "DOC-H02",
                "doc_name": "Itemized Final Hospital Bill",
                "doc_description": "Detailed breakup of room, OT, nursing, surgery, and medications",
                "is_mandatory": True,
                "provided_by": "hospital",
                "status": "REQUESTED_FROM_HOSPITAL",
                "file_path": None,
                "hospital_signature_hash": None
            },
            {
                "doc_id": "DOC-P01",
                "doc_name": "Signed Mediclaim Claim Form",
                "doc_description": "Duly filled Star Health Form C with policyholder signature",
                "is_mandatory": True,
                "provided_by": "patient",
                "status": "VALIDATED",
                "file_path": "/uploads/claim_form_signed.pdf",
                "hospital_signature_hash": None
            },
            {
                "doc_id": "DOC-P02",
                "doc_name": "Govt Photo ID (Aadhaar / PAN)",
                "doc_description": "Identity verification proof for insurance processing",
                "is_mandatory": True,
                "provided_by": "patient",
                "status": "VALIDATED",
                "file_path": "/uploads/aadhaar_verified.pdf",
                "hospital_signature_hash": None
            },
            {
                "doc_id": "DOC-P03",
                "doc_name": "Cancelled Cheque / Bank Passbook",
                "doc_description": "Bank account confirmation for direct reimbursement deposit",
                "is_mandatory": True,
                "provided_by": "patient",
                "status": "VALIDATED",
                "file_path": "/uploads/bank_cheque.pdf",
                "hospital_signature_hash": None
            }
        ]
    },
    "audit_trail": [
        {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "node": "system_init", "action": "Platform initialized with LangGraph checkpointer"}
    ]
}


# ---------------------------------------------------------------------------
# Backend Action Handlers
# ---------------------------------------------------------------------------
def patient_chat_interact(user_message, chat_history):
    if not user_message.strip():
        return "", chat_history, get_status_overview(), get_documents_table()

    chat_history = chat_history or []

    # Run LangGraph engine
    config = {"configurable": {"thread_id": GLOBAL_SESSION_ID}}
    input_state: UnifiedAgentState = {
        "messages": [HumanMessage(content=user_message)],
        "active_intent": "general_help",
        "intake_record": CURRENT_STATE["intake_record"],
        "claim_record": CURRENT_STATE["claim_record"],
        "missing_fields": [],
        "rag_context": "",
        "hospital_requests": [],
        "audit_trail": CURRENT_STATE["audit_trail"],
        "session_id": GLOBAL_SESSION_ID,
        "human_interrupt_data": None,
        "final_output": None
    }

    try:
        result = healthcare_agent_graph.invoke(input_state, config=config)
        agent_reply = result.get("final_output", "I processed your request.")
        CURRENT_STATE["intake_record"] = result.get("intake_record", CURRENT_STATE["intake_record"])
        CURRENT_STATE["claim_record"] = result.get("claim_record", CURRENT_STATE["claim_record"])
        CURRENT_STATE["audit_trail"] = result.get("audit_trail", CURRENT_STATE["audit_trail"])
    except Exception as e:
        agent_reply = f"⚠️ Agent encountered an issue: {str(e)}"

    chat_history.append((user_message, agent_reply))
    return "", chat_history, get_status_overview(), get_documents_table()


def request_hospital_docs_online():
    """Simulates the core feature: patient requests hospital documents digitally without visiting."""
    docs = CURRENT_STATE["claim_record"]["documents"]
    count = 0
    for d in docs:
        if d["provided_by"] == "hospital" and d["status"] != "FULFILLED_BY_HOSPITAL":
            d["status"] = "REQUESTED_FROM_HOSPITAL"
            count += 1

    audit_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "node": "hospital_dispatch_gateway",
        "action": f"Sent digital requisition for {count} hospital document(s) directly to Apollo Hospital Records Desk."
    }
    CURRENT_STATE["audit_trail"].append(audit_entry)
    msg = f"✅ **Digital Requisition Sent to Hospital Records Desk!**\n" \
          f"Patient does NOT need to visit the hospital.\n" \
          f"- **Hospital**: {CURRENT_STATE['claim_record']['hospital_name']}\n" \
          f"- **Documents Requested**: {count} items\n" \
          f"- **Notification**: SMS and Portal alert dispatched to Hospital Medical Records Desk."
    return msg, get_documents_table(), get_hospital_desk_view()


def hospital_fulfill_and_sign():
    """Simulates hospital staff uploading and cryptographically signing requested documents."""
    docs = CURRENT_STATE["claim_record"]["documents"]
    fulfilled_count = 0
    for d in docs:
        if d["provided_by"] == "hospital":
            d["status"] = "FULFILLED_BY_HOSPITAL"
            d["file_path"] = f"/hospital_vault/{d['doc_id']}_signed_certified.pdf"
            d["hospital_signature_hash"] = f"SHA256:{uuid.uuid4().hex[:16].upper()}"
            fulfilled_count += 1

    # Check if all mandatory docs ready
    all_ready = all(d["status"] in ["VALIDATED", "FULFILLED_BY_HOSPITAL"] for d in docs if d["is_mandatory"])
    if all_ready:
        CURRENT_STATE["claim_record"]["status"] = "READY_FOR_ADJUDICATION"

    audit_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "node": "hospital_records_portal",
        "action": f"Dr. Ramesh Kumar signed & uploaded {fulfilled_count} hospital document(s) to claim package."
    }
    CURRENT_STATE["audit_trail"].append(audit_entry)

    res = f"🖋️ **Documents Digitally Certified & Uploaded by Hospital Staff!**\n" \
          f"- **Documents Signed**: {fulfilled_count}\n" \
          f"- **Digital Signature**: SHA-256 Verified by Hospital Attending Authority\n" \
          f"- **Patient Status**: Updated in real-time. Patient saved 1 physical trip to hospital!"
    return res, get_hospital_desk_view(), get_documents_table(), get_adjudication_view()


def doctor_acknowledge_alert(doctor_notes):
    """Doctor acknowledges critical emergency alert."""
    notes = doctor_notes or "Physician notified, Cath Lab on 10-minute standby."
    CURRENT_STATE["intake_record"]["doctor_acknowledgment"] = notes

    audit_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "node": "doctor_er_console",
        "action": f"Physician acknowledged critical alert: {notes}"
    }
    CURRENT_STATE["audit_trail"].append(audit_entry)

    return f"✅ **Acknowledged by On-Duty Physician!**\nClinical Order: {notes}", get_doctor_alert_card()


def insurer_adjudicate_claim(decision, reason):
    claim = CURRENT_STATE["claim_record"]
    claim["status"] = decision
    claim["insurer_reference"] = f"TPA-SETTLE-{uuid.uuid4().hex[:8].upper()}"
    claim["submission_timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    audit_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "node": "tpa_adjudication",
        "action": f"Claim {claim['claim_id']} {decision}. Ref: {claim['insurer_reference']}"
    }
    CURRENT_STATE["audit_trail"].append(audit_entry)

    msg = f"⚖️ **Claim Decision Executed: {decision}**\n" \
          f"- **Insurer Reference**: `{claim['insurer_reference']}`\n" \
          f"- **Adjudication Notes**: {reason or 'Standard settlement approval'}\n" \
          f"- **Payout Timeline**: Direct NEFT to patient bank account within 3 business days."
    return msg, get_adjudication_view()


# ---------------------------------------------------------------------------
# View Generators
# ---------------------------------------------------------------------------
def get_status_overview():
    claim = CURRENT_STATE["claim_record"]
    intake = CURRENT_STATE["intake_record"]
    docs = claim.get("documents", [])
    ready_count = sum(1 for d in docs if d["status"] in ["VALIDATED", "FULFILLED_BY_HOSPITAL"])

    return f"""### 📌 Live Session Telemetry
- **Active Patient**: `{claim.get('patient_name', 'N/A')}`
- **Clinical Priority**: `{intake.get('triage_severity', 'UNKNOWN')}` ({intake.get('recommended_dept', 'OPD')})
- **Claim ID**: `{claim.get('claim_id', 'N/A')}` | **Insurer**: `{claim.get('insurer_name', 'N/A')}`
- **Total Bill**: ₹{claim.get('total_bill_amount', 0):,.2f}
- **Document Progress**: **{ready_count}/{len(docs)} Ready**
- **Fraud Risk Score**: `{claim.get('fraud_risk_score', 0)}/100` ({claim.get('fraud_risk_tier', 'LOW')})
- **Claim Status**: `{claim.get('status', 'DRAFT')}`
"""

def get_documents_table():
    docs = CURRENT_STATE["claim_record"].get("documents", [])
    data = []
    for d in docs:
        status_icon = "⏳ Pending"
        if d["status"] == "VALIDATED":
            status_icon = "✅ Validated"
        elif d["status"] == "REQUESTED_FROM_HOSPITAL":
            status_icon = "📨 Requested from Hospital"
        elif d["status"] == "FULFILLED_BY_HOSPITAL":
            status_icon = "🏥 Uploaded & Signed by Hospital"

        who = "🏥 Hospital" if d["provided_by"] == "hospital" else "👤 Patient"
        mand = "Yes (Mandatory)" if d["is_mandatory"] else "Optional"
        data.append([d["doc_name"], who, mand, status_icon, d.get("hospital_signature_hash") or "N/A"])
    return data

def get_doctor_alert_card():
    intake = CURRENT_STATE["intake_record"]
    ack = intake.get("doctor_acknowledgment")
    status_str = f"✅ Acknowledged: {ack}" if ack else "🚨 UNACKNOWLEDGED EMERGENCY ALERT"
    return f"""### 🚨 Emergency Patient Monitor
- **Patient**: {intake.get('patient_name')} ({intake.get('age')}y / {intake.get('gender')})
- **Symptom Severity**: **{intake.get('symptom_severity')}/10**
- **Chief Complaint**: {intake.get('primary_complaint')}
- **Triage Level**: **{intake.get('triage_severity')}**
- **Target Unit**: {intake.get('recommended_dept')}
- **Clinical Reasoning**: {intake.get('triage_reasoning')}
- **Doctor Action Status**: `{status_str}`
"""

def get_hospital_desk_view():
    claim = CURRENT_STATE["claim_record"]
    docs = claim.get("documents", [])
    hosp_docs = [d for d in docs if d["provided_by"] == "hospital"]
    pending = [d["doc_name"] for d in hosp_docs if d["status"] != "FULFILLED_BY_HOSPITAL"]

    return f"""### 🏥 Hospital Medical Records & Attending Doctor Desk
- **Hospital**: {claim.get('hospital_name')}
- **Patient**: {claim.get('patient_name')}
- **Admission**: `{claim.get('admission_date')}` to `{claim.get('discharge_date')}`
- **Diagnosis**: {claim.get('diagnosis')}
- **Pending Requisitions**: **{len(pending)} item(s) pending** ({", ".join(pending) if pending else "All Signed"})
- **Action**: Hospital records staff can upload & attach cryptographic signature without requiring patient presence.
"""

def get_adjudication_view():
    claim = CURRENT_STATE["claim_record"]
    return f"""### ⚖️ TPA / Insurer Claim Adjudication Overview
- **Claim ID**: `{claim.get('claim_id')}` | **Insurer**: `{claim.get('insurer_name')}`
- **Policy Number**: `{claim.get('policy_number')}`
- **Claim Amount**: ₹{claim.get('total_bill_amount', 0):,.2f}
- **Current Lifecycle Status**: `{claim.get('status')}`
- **Automated Fraud Risk Metric**: **{claim.get('fraud_risk_score')}/100** ({claim.get('fraud_risk_tier')})
- **Discrepancy Audit Findings**: {", ".join(claim.get('fraud_anomalies', []))}
- **Settlement Reference**: `{claim.get('insurer_reference', 'Pending Final Adjudication')}`
"""

def get_audit_trail_text():
    logs = CURRENT_STATE.get("audit_trail", [])
    lines = [f"[{l['timestamp']}] ({l['node']}) -> {l['action']}" for l in logs]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI Construction (Multi-Stakeholder Experience)
# ---------------------------------------------------------------------------
with gr.Blocks(title="AI Healthcare & Mediclaim Agentic Platform", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥🏦 Autonomous Healthcare & Mediclaim Agentic Platform
    ### *Production-Grade Multi-Agent System with LangGraph State Machine, Hybrid RAG & Multi-Stakeholder HITL*
    """)

    with gr.Tabs():
        # TAB 1: PATIENT PORTAL
        with gr.TabItem("👤 1. Patient Portal (Intake & Mediclaim)"):
            with gr.Row():
                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(
                        value=[
                            ("Hello! How can I assist you today?",
                             "I am your AI Healthcare & Mediclaim Agent. You can describe symptoms for instant clinical triage, or enter your hospitalization & insurance details to file a claim without hospital visits.")
                        ],
                        height=420
                    )
                    with gr.Row():
                        user_input = gr.Textbox(
                            show_label=False,
                            placeholder="Type symptoms (e.g. 'Severe chest pain radiating to jaw') or claim info (e.g. 'Star Health claim for appendectomy at Apollo')...",
                            scale=8
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=2)

                    with gr.Row():
                        req_hosp_btn = gr.Button("📨 Request Hospital Documents Digitally (No Physical Visit Needed!)", variant="secondary")

                    patient_action_output = gr.Markdown("")

                with gr.Column(scale=4):
                    status_card = gr.Markdown(get_status_overview())
                    gr.Markdown("#### 📋 Live Document Status Matrix")
                    doc_table = gr.Dataframe(
                        headers=["Document Name", "Source", "Mandatory", "Status", "Digital Signature Hash"],
                        value=get_documents_table(),
                        interactive=False
                    )

        # TAB 2: DOCTOR ER CONSOLE
        with gr.TabItem("🚨 2. Doctor ER & Triage Console"):
            with gr.Row():
                with gr.Column(scale=6):
                    doc_alert_card = gr.Markdown(get_doctor_alert_card())
                with gr.Column(scale=4):
                    gr.Markdown("#### 👨‍⚕️ Emergency Clinical Override & Acknowledgment")
                    doc_notes_input = gr.Textbox(
                        label="Physician Orders / Instructions",
                        placeholder="e.g. 'Direct admit to ICU, prepare heparin, 12-lead ECG stat.'"
                    )
                    ack_btn = gr.Button("🚨 Acknowledge Critical Alert & Dispatch Care", variant="stop")
                    doc_ack_status = gr.Markdown("")

        # TAB 3: HOSPITAL RECORDS DESK
        with gr.TabItem("🏥 3. Hospital Records Desk (Digital E-Sign)"):
            with gr.Row():
                with gr.Column(scale=6):
                    hosp_desk_info = gr.Markdown(get_hospital_desk_view())
                with gr.Column(scale=4):
                    gr.Markdown("#### 🖋️ Digital Document Certification")
                    gr.Markdown("Hospital administrative staff can fulfill patient document requests directly into the claim package with verified digital certification.")
                    fulfill_btn = gr.Button("✅ Sign & Upload Discharge Summary & Itemized Bill", variant="primary")
                    hosp_fulfill_status = gr.Markdown("")

        # TAB 4: INSURER ADJUDICATION
        with gr.TabItem("⚖️ 4. Insurer / TPA Adjudication & Fraud Audit"):
            with gr.Row():
                with gr.Column(scale=6):
                    adj_view = gr.Markdown(get_adjudication_view())
                with gr.Column(scale=4):
                    gr.Markdown("#### 🛡️ Adjudication Actions")
                    adj_reason = gr.Textbox(label="Adjudication Rationale", placeholder="e.g., 'Documents verified against clinical protocol. Approved.'")
                    with gr.Row():
                        approve_btn = gr.Button("✅ Approve Claim & Settle", variant="primary")
                        reject_btn = gr.Button("❌ Reject Claim", variant="stop")
                    adj_status = gr.Markdown("")

        # TAB 5: SYSTEM ARCHITECTURE & TELEMETRY
        with gr.TabItem("🧠 5. Architecture & State Telemetry"):
            gr.Markdown("""
            ### 🏗️ Production Architecture Specification
            - **Orchestration**: LangGraph State Machine with SQLite Session Persistence
            - **Supervisor Node**: Autonomous routing across Clinical Triage, Mediclaim Intake, and Hospital Gateway
            - **Hybrid RAG**: Dense Embeddings (all-MiniLM-L6-v2) + Sparse BM25 Keyword Search
            - **Fraud Engine**: Deterministic LOS validation + Unbundled billing anomaly detection
            - **Multi-Stakeholder HITL**:
                - *Interrupt 1*: Emergency Doctor escalation for CRITICAL triage
                - *Interrupt 2*: Hospital staff digital signature fulfillment without patient visits
                - *Interrupt 3*: Insurer final adjudication settlement
            """)
            refresh_logs_btn = gr.Button("🔄 Refresh Audit Logs")
            audit_log_display = gr.Code(label="Real-Time LangGraph Execution Audit Trail", value=get_audit_trail_text(), language="shell")

    # Wire event handlers
    send_btn.click(
        patient_chat_interact,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot, status_card, doc_table]
    )
    user_input.submit(
        patient_chat_interact,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot, status_card, doc_table]
    )
    req_hosp_btn.click(
        request_hospital_docs_online,
        outputs=[patient_action_output, doc_table, hosp_desk_info]
    )
    ack_btn.click(
        doctor_acknowledge_alert,
        inputs=[doc_notes_input],
        outputs=[doc_ack_status, doc_alert_card]
    )
    fulfill_btn.click(
        hospital_fulfill_and_sign,
        outputs=[hosp_fulfill_status, hosp_desk_info, doc_table, adj_view]
    )
    approve_btn.click(
        lambda r: insurer_adjudicate_claim("APPROVED", r),
        inputs=[adj_reason],
        outputs=[adj_status, adj_view]
    )
    reject_btn.click(
        lambda r: insurer_adjudicate_claim("REJECTED", r),
        inputs=[adj_reason],
        outputs=[adj_status, adj_view]
    )
    refresh_logs_btn.click(
        get_audit_trail_text,
        outputs=[audit_log_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
