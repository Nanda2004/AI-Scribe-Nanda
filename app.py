import os
import io
import time
import json
import requests
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

AAI_KEY = os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("Assemby_api_key") or os.getenv("assemblyai_api_key")
BASE_URL = "https://api.assemblyai.com/v2"
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key")

def auth_headers():
    return {"authorization": AAI_KEY}

def upload_bytes(data: bytes) -> str:
    url = f"{BASE_URL}/upload"
    res = requests.post(url, headers=auth_headers(), data=data)
    res.raise_for_status()
    return res.json()["upload_url"]

def start_transcript(audio_url: str, speaker_labels: bool = True) -> str:
    payload = {
        "audio_url": audio_url,
        "speaker_labels": speaker_labels,
        "format_text": True,
        "punctuate": True,
        "speech_model": "universal",
        "language_detection": True
    }
    res = requests.post(f"{BASE_URL}/transcript", headers={**auth_headers(), "content-type": "application/json"}, json=payload)
    res.raise_for_status()
    return res.json()["id"]

def poll_transcript(tid: str) -> dict:
    endpoint = f"{BASE_URL}/transcript/{tid}"
    while True:
        res = requests.get(endpoint, headers=auth_headers())
        res.raise_for_status()
        body = res.json()
        status = body.get("status")
        if status == "completed":
            return body
        if status == "error":
            raise RuntimeError(body.get("error"))
        time.sleep(2)

def format_utterances(utt: list) -> list:
    items = []
    for u in utt or []:
        spk = u.get("speaker", "Unknown")
        txt = u.get("text", "")
        start = u.get("start", 0) / 1000.0 if isinstance(u.get("start"), (int, float)) else 0
        end = u.get("end", 0) / 1000.0 if isinstance(u.get("end"), (int, float)) else 0
        items.append({"speaker": spk, "text": txt, "start": start, "end": end})
    return items

def generate_note(transcript_text: str, fmt: str) -> str:
    if fmt == "SOAP":
        prompt = (
            "You are a medical documentation assistant. \n"
            "Your task is to convert the provided clinical transcript into a concise, accurate, and well-structured SOAP note.\n"
            "Only use information explicitly stated in the transcript. \n"
            "Do NOT add, infer, or assume any data not mentioned.\n"
            "If a section has no data in the transcript, write: “Not mentioned.”\n\n"
            "Format exactly as follows:\n\n"
            "SOAP NOTE\n"
            "Patient Name:\n"
            "DOB:\n"
            "Clinician:\n"
            "Date:\n"
            "Setting: (telemedicine / in-person) — based on transcript\n\n"
            "S – Subjective\n"
            "• Chief Complaint:\n"
            "• History of Present Illness:\n"
            "• Review of Systems (only items mentioned):\n"
            "• Past Medical History:\n"
            "• Medications:\n"
            "• Allergies:\n"
            "• Family History:\n"
            "• Social History:\n\n"
            "O – Objective\n"
            "• Exam findings from transcript\n"
            "(If telehealth and no exam provided, write: “No physical exam performed; assessment based on verbal report.”)\n"
            "• Vitals if mentioned\n\n"
            "A – Assessment\n"
            "• List all clinician-stated assessments or inferred concerns explicitly stated in the transcript\n"
            "• Do NOT generate diagnoses that were not discussed\n\n"
            "P – Plan\n"
            "• Diagnostics ordered or recommended\n"
            "• Treatments/medications advised\n"
            "• Work restrictions\n"
            "• Safety netting / follow-up advice\n\n"
            "Now generate the SOAP note based on the following transcript:\n\n"
            "[TRANSCRIPT]\n{{TRANSCRIPT_HERE}}\n\n".replace("{{TRANSCRIPT_HERE}}", transcript_text)
        )
    else:
        prompt = (
            "You are a medical documentation assistant. \n"
            "Convert the provided transcript into a structured History & Physical (H&P) note.\n"
            "Use only information explicitly stated. Do NOT guess or add details.\n"
            "If a section is missing information, mark it as “Not mentioned.”\n\n"
            "Format exactly as follows:\n\n"
            "HISTORY & PHYSICAL (H&P)\n\n"
            "Patient Name:\n"
            "DOB:\n"
            "Clinician:\n"
            "Date:\n"
            "Setting:\n\n"
            "HISTORY\n"
            "Chief Complaint:\n"
            "History of Present Illness:\n"
            "Past Medical History:\n"
            "Past Surgical History:\n"
            "Medications:\n"
            "Allergies:\n"
            "Family History:\n"
            "Social History:\n"
            "Review of Systems:\n"
            "(Only list items explicitly found in the transcript.)\n\n"
            "PHYSICAL EXAM\n"
            "• If no exam data exists, write: “Not performed in transcript.”\n\n"
            "ASSESSMENT\n"
            "• Summarize the clinician’s diagnostic thinking exactly as discussed.\n"
            "• Do NOT generate new differentials unless mentioned.\n\n"
            "PLAN\n"
            "• Document investigations ordered or recommended\n"
            "• Treatment recommendations\n"
            "• Follow-up instructions\n"
            "• Any disposition (e.g., clinic referral, ER recommendation)\n\n"
            "Now generate the H&P note using the transcript below:\n\n"
            "[TRANSCRIPT]\n{{TRANSCRIPT_HERE}}\n\n".replace("{{TRANSCRIPT_HERE}}", transcript_text)
        )
    if not GEMINI_KEY:
        return ""
    genai.configure(api_key=GEMINI_KEY)
    base_names = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
    ]
    candidates = []
    for n in base_names:
        candidates.append(n)
        candidates.append("models/" + n)
    for name in candidates:
        try:
            m = genai.GenerativeModel(name)
            m.count_tokens("")
            r = m.generate_content(prompt)
            st.session_state["note_model"] = name
            return r.text or ""
        except Exception:
            continue
    return ""

def fallback_note(transcript_text: str, fmt: str) -> str:
    t = transcript_text.strip()
    if fmt == "SOAP":
        return (
            "SOAP NOTE\n"
            "Patient Name: Not mentioned.\n"
            "DOB: Not mentioned.\n"
            "Clinician: Not mentioned.\n"
            "Date: Not mentioned.\n"
            "Setting: Not mentioned.\n\n"
            "S – Subjective\n"
            "• Chief Complaint: Not mentioned.\n"
            "• History of Present Illness: " + (t if t else "Not mentioned.") + "\n"
            "• Review of Systems (only items mentioned): Not mentioned.\n"
            "• Past Medical History: Not mentioned.\n"
            "• Medications: Not mentioned.\n"
            "• Allergies: Not mentioned.\n"
            "• Family History: Not mentioned.\n"
            "• Social History: Not mentioned.\n\n"
            "O – Objective\n"
            "• Exam findings from transcript\n"
            "No physical exam performed; assessment based on verbal report.\n"
            "• Vitals if mentioned\n"
            "Not mentioned.\n\n"
            "A – Assessment\n"
            "• Not mentioned.\n\n"
            "P – Plan\n"
            "• Not mentioned.\n"
        )
    else:
        return (
            "HISTORY & PHYSICAL (H&P)\n\n"
            "Patient Name: Not mentioned.\n"
            "DOB: Not mentioned.\n"
            "Clinician: Not mentioned.\n"
            "Date: Not mentioned.\n"
            "Setting: Not mentioned.\n\n"
            "HISTORY\n"
            "Chief Complaint: Not mentioned.\n"
            "History of Present Illness: " + (t if t else "Not mentioned.") + "\n"
            "Past Medical History: Not mentioned.\n"
            "Past Surgical History: Not mentioned.\n"
            "Medications: Not mentioned.\n"
            "Allergies: Not mentioned.\n"
            "Family History: Not mentioned.\n"
            "Social History: Not mentioned.\n"
            "Review of Systems: Not mentioned.\n\n"
            "PHYSICAL EXAM\n"
            "• Not performed in transcript.\n\n"
            "ASSESSMENT\n"
            "• Not mentioned.\n\n"
            "PLAN\n"
            "• Not mentioned.\n"
        )

def beautify_note(note_text: str) -> str:
    lines = (note_text or "").splitlines()
    out = []
    for raw in lines:
        s = raw.strip()
        if not s:
            out.append("")
            continue
        if s in ("SOAP NOTE", "HISTORY & PHYSICAL (H&P)"):
            out.append(f"# {s}")
        elif s in ("S – Subjective", "O – Objective", "A – Assessment", "P – Plan",
                    "HISTORY", "PHYSICAL EXAM", "ASSESSMENT", "PLAN"):
            out.append(f"## {s}")
        elif s.endswith(":") and not s.startswith("•"):
            out.append(f"**{s}**")
        else:
            out.append(s)
    return "\n\n".join(out)

def main():
    st.set_page_config(page_title="AI Scribe", layout="wide")
    st.markdown(
        """
        <style>
        body {background: radial-gradient(circle at top left,#e0f2fe,#f5f3ff 40%,#fef9c3 80%);}        
        .main .block-container{max-width:1100px;padding-top:2rem;padding-bottom:4rem;}
        h1{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-weight:800;letter-spacing:-0.04em;}
        .app-subtitle{color:#6b7280;font-size:0.95rem;margin-bottom:1.5rem;}
        .note-card{padding:20px 22px;border:1px solid rgba(148,163,184,0.4);border-radius:18px;background:linear-gradient(135deg,#f9fafb,#eef2ff);box-shadow:0 18px 45px rgba(15,23,42,0.12);}        
        .section-header{font-weight:700; margin: 8px 0 4px 0}
        .pill{display:inline-block;padding:2px 8px;border:1px solid #3b82f6;border-radius:999px;color:#1e3a8a;font-weight:600;background:#eff6ff;font-size:0.75rem;}
        .spk-a{background:#eef2ff;border-left:4px solid #6366f1;padding:8px;border-radius:8px}
        .spk-b{background:#ecfdf5;border-left:4px solid #10b981;padding:8px;border-radius:8px}
        .spk-x{background:#f3f4f6;border-left:4px solid #9ca3af;padding:8px;border-radius:8px}
        .stTabs [data-baseweb="tab-list"]{gap:1.5rem;border-bottom:1px solid #e5e7eb;margin-bottom:0.5rem;}
        .stTabs [data-baseweb="tab"]{font-weight:600;font-size:0.9rem;color:#6b7280;}
        .stTabs [data-baseweb="tab"][aria-selected="true"]{color:#2563eb;border-bottom:2px solid #2563eb;}
        .stButton>button{width:100%;border-radius:999px;background:linear-gradient(90deg,#2563eb,#7c3aed);color:white;font-weight:600;border:none;padding:0.7rem 1.1rem;box-shadow:0 12px 30px rgba(37,99,235,0.45);letter-spacing:0.02em;}        
        .stButton>button:hover{filter:brightness(1.05);}        
        .stRadio>div{gap:0.75rem;}        
        .audio-card{border-radius:16px;border:1px solid rgba(148,163,184,0.35);background:rgba(255,255,255,0.9);padding:16px 18px;box-shadow:0 10px 30px rgba(15,23,42,0.08);}        
        .toggle-label{font-size:0.85rem;color:#4b5563;font-weight:500;margin-top:0.5rem;}        
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("# AI Scribe")
    st.markdown("<div class='app-subtitle'>Upload a clinical encounter, view diarized transcript, and generate structured SOAP or H&P notes.</div>", unsafe_allow_html=True)
    if not AAI_KEY:
        st.error("AssemblyAI API key not found in .env. Set ASSEMBLYAI_API_KEY or Assemby_api_key.")
        return
    if not GEMINI_KEY:
        st.warning("GEMINI_API_KEY not set; Note generation will be skipped.")
    if "note_model" in st.session_state:
        st.caption("Note model: " + st.session_state.get("note_model"))

    tabs = st.tabs(["Transcript", "Code", "API Response"])

    with tabs[0]:
        left_cfg, right_cfg = st.columns([1, 1])
        with left_cfg:
            st.subheader("Audio")
            src_choice = st.radio("Source", ["Upload", "URL"], horizontal=True)
            uploaded = None
            url_input = None
            if src_choice == "Upload":
                uploaded = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a", "mp4", "aac", "ogg", "flac"]) 
            else:
                url_input = st.text_input("Audio URL", value="")
            diar = st.toggle("Speaker labels", value=True)
        with right_cfg:
            st.subheader("Note Options")
            note_format = st.radio("Choose format", ["SOAP", "H&P"], horizontal=True)
        run = st.button("Transcribe & Generate", use_container_width=True)
        c2 = st.container()
        with c2:
            audio_bytes = None
            if uploaded is not None:
                audio_bytes = uploaded.read()
                st.audio(audio_bytes)
            elif url_input:
                try:
                    pr = requests.get(url_input, timeout=10)
                    if pr.ok:
                        audio_bytes = pr.content
                        st.audio(audio_bytes)
                except Exception:
                    st.info("Audio will stream from URL if supported.")

        st.divider()
        transcript_json = st.session_state.get("aai_result")
        note_text = st.session_state.get("note_text")
        if run:
            audio_url = None
            if uploaded is not None:
                audio_url = upload_bytes(audio_bytes)
            elif url_input:
                audio_url = url_input
            if audio_url:
                tid = start_transcript(audio_url, speaker_labels=diar)
                with st.spinner("Transcribing..."):
                    result = poll_transcript(tid)
                st.session_state["aai_result"] = result
                transcript_json = result
                full_text = result.get("text") or ""
                if GEMINI_KEY and full_text:
                    with st.spinner("Generating note..."):
                        nt = generate_note(full_text, note_format)
                    st.session_state["note_text"] = nt
                    note_text = nt
                else:
                    note_text = None
        if transcript_json and GEMINI_KEY and not note_text:
            ft = transcript_json.get("text") or ""
            if ft:
                with st.spinner("Generating note..."):
                    nt = generate_note(ft, note_format)
                note_text = nt or fallback_note(ft, note_format)
                st.session_state["note_text"] = note_text
        if transcript_json and not note_text:
            ft = transcript_json.get("text") or ""
            if ft:
                note_text = fallback_note(ft, note_format)
                st.session_state["note_text"] = note_text
        if transcript_json:
            left, right = st.columns([3, 4])
            with left:
                st.markdown("## Note")
                md = beautify_note(note_text or "")
                st.markdown(f"<div class='note-card'>{st.markdown(md, unsafe_allow_html=False) if False else ''}</div>", unsafe_allow_html=True)
                st.markdown(md)
                cdl1, cdl2 = st.columns([1,1])
                with cdl1:
                    st.download_button("Download .md", data=md, file_name="note.md", mime="text/markdown")
                with cdl2:
                    st.download_button("Download .txt", data=(note_text or md), file_name="note.txt", mime="text/plain")
            with right:
                utter = transcript_json.get("utterances")
                if not utter and transcript_json.get("text"):
                    utter = [{"speaker": "Speaker", "text": transcript_json.get("text")}] 
                utter = format_utterances(utter)
                st.markdown("## Transcript")
                cont = st.container(height=480, border=True)
                spk_map = {}
                palette = ["spk-a", "spk-b"]
                for u in utter:
                    spk = str(u.get("speaker", "")).strip() or "X"
                    if spk not in spk_map and len(spk_map) < 2:
                        spk_map[spk] = palette[len(spk_map)]
                    cls = spk_map.get(spk, "spk-x")
                    block = f"<div class='{cls}'><div class='pill' style='margin-bottom:6px'>SPEAKER {spk}</div><div>{u['text']}</div></div>"
                    cont.markdown(block, unsafe_allow_html=True)
                lc, rc = st.columns([1, 9])
                with lc:
                    like = st.button("👍")
                    dislike = st.button("👎")
                with rc:
                    pass

    with tabs[1]:
        code_str = (
            "import requests\n"
            "BASE='https://api.assemblyai.com/v2'\n"
            "H={'authorization': 'YOUR_KEY'}\n"
            "u=requests.post(BASE+'/upload',headers=H,data=open('audio.mp3','rb')).json()['upload_url']\n"
            "tid=requests.post(BASE+'/transcript',headers={**H,'content-type':'application/json'},json={"
            "'audio_url':u,'speaker_labels':True,'format_text':True,'punctuate':True}).json()['id']\n"
            "import time\n"
            "while True:\n"
            " r=requests.get(BASE+'/transcript/'+tid,headers=H).json()\n"
            " if r['status']=='completed': break\n"
            " if r['status']=='error': raise SystemExit(r['error'])\n"
            " time.sleep(2)\n"
            "print(r)\n"
        )
        st.code(code_str, language="python")

    with tabs[2]:
        data = st.session_state.get("aai_result")
        if data:
            st.json(data)
        else:
            st.info("No API response yet.")

if __name__ == "__main__":
    main()