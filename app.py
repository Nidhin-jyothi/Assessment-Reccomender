"""
app.py - Streamlit chat UI for the SHL Assessment Recommender.

Calls the FastAPI backend at BACKEND_URL (default: http://localhost:8000).
Run alongside the FastAPI server:
    uvicorn main:app --port 8000 &
    streamlit run app.py
"""

import os
import httpx
import streamlit as st

#  Config 
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MAX_TURNS   = 8

TEST_TYPE_LABELS = {
    "A": ("", "Ability & Aptitude"),
    "B": ("", "Biodata & SJT"),
    "C": ("", "Competencies"),
    "D": ("", "Development & 360"),
    "E": ("", "Assessment Exercises"),
    "K": ("", "Knowledge & Skills"),
    "P": ("", "Personality & Behavior"),
    "S": ("", "Simulations"),
}

#  Page config 
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="",
    layout="centered",
)

#  Styling 
st.markdown("""
<style>
    .main { max-width: 780px; margin: 0 auto; }

    .rec-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1a56db;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .rec-card a {
        color: #1a56db;
        text-decoration: none;
        font-weight: 600;
        font-size: 15px;
    }
    .rec-card a:hover { text-decoration: underline; }
    .type-badge {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 4px;
    }
    .eoc-banner {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 10px 16px;
        color: #166534;
        font-size: 14px;
        margin-top: 8px;
    }
    div[data-testid="stChatMessage"] { padding: 4px 0; }
</style>
""", unsafe_allow_html=True)

#  Session state 
if "messages"     not in st.session_state: st.session_state.messages     = []
if "conv_ended"   not in st.session_state: st.session_state.conv_ended   = False
if "last_recs"    not in st.session_state: st.session_state.last_recs    = []
if "turn_count"   not in st.session_state: st.session_state.turn_count   = 0


#  Helpers 
def call_backend(messages: list[dict]) -> dict | None:
    try:
        resp = httpx.post(
            f"{BACKEND_URL}/chat",
            json={"messages": messages},
            timeout=35.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.TimeoutException:
        return {"reply": " Request timed out. Please try again.", "recommendations": [], "end_of_conversation": False}
    except Exception as e:
        return {"reply": f" Could not reach the backend: {e}", "recommendations": [], "end_of_conversation": False}


def render_recommendations(recs: list[dict]):
    if not recs:
        return
    st.markdown(f"**{len(recs)} Assessment{'s' if len(recs) > 1 else ''} Recommended**")
    for i, rec in enumerate(recs, 1):
        tt = rec.get("test_type", "K")
        icon, label = TEST_TYPE_LABELS.get(tt, ("", tt))
        st.markdown(
            f"""<div class="rec-card">
                <span style="color:#64748b;font-size:12px">#{i}</span>&nbsp;
                <a href="{rec['url']}" target="_blank">{rec['name']}</a><br>
                <span class="type-badge">{icon} {label}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def reset_conversation():
    st.session_state.messages   = []
    st.session_state.conv_ended = False
    st.session_state.last_recs  = []
    st.session_state.turn_count = 0


#  Header 
st.title(" SHL Assessment Recommender")
st.caption("Describe the role you're hiring for and I'll recommend the right SHL assessments.")

# Backend status check
@st.cache_data(ttl=30)
def check_health():
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False

if not check_health():
    st.error(f" Backend not reachable at `{BACKEND_URL}`. Make sure `uvicorn main:app` is running.")
    st.stop()

#  Chat history 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("recommendations"):
            render_recommendations(msg["recommendations"])
        if msg.get("end_of_conversation"):
            st.markdown(
                '<div class="eoc-banner"> Conversation complete. Use the button below to start over.</div>',
                unsafe_allow_html=True,
            )

#  Input 
col1, col2 = st.columns([5, 1])
with col2:
    if st.button(" Reset", use_container_width=True):
        reset_conversation()
        st.rerun()

if st.session_state.conv_ended:
    st.info("Conversation complete. Click **Reset** to start a new one.")
    st.stop()

if st.session_state.turn_count >= MAX_TURNS:
    st.warning(f"Turn limit ({MAX_TURNS}) reached. Please reset to start over.")
    st.stop()

user_input = st.chat_input("e.g. I'm hiring a mid-level Java developer...")

if user_input and user_input.strip():
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    st.session_state.turn_count += 1

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input.strip())

    # Build API payload (role + content only)
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = call_backend(api_messages)

        reply = result.get("reply", "")
        recs  = result.get("recommendations", [])
        eoc   = result.get("end_of_conversation", False)

        st.markdown(reply)
        if recs:
            render_recommendations(recs)
        if eoc:
            st.markdown(
                '<div class="eoc-banner"> Conversation complete. Use Reset to start over.</div>',
                unsafe_allow_html=True,
            )

    # Store assistant message with metadata
    st.session_state.messages.append({
        "role":              "assistant",
        "content":           reply,
        "recommendations":   recs,
        "end_of_conversation": eoc,
    })
    st.session_state.last_recs  = recs
    st.session_state.conv_ended = eoc
    st.session_state.turn_count += 1
