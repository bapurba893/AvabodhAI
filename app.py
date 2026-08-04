"""
app.py
------
Streamlit UI for Avabodh document summarisation pipeline.
Run with: streamlit run app.py
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"

import time
import streamlit as st

st.set_page_config(
    page_title="Avabodh — Document Summariser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(__file__))

# @st.cache_resource loads these ONCE and reuses across all sessions
# Without this — every page load reimports all heavy ML packages
@st.cache_resource
def load_pipeline():
    from db.database import init_db, get_db_session
    from db.models import DocumentSummary
    from pipeline.loader import load_single_document
    from pipeline.splitter import split_documents
    from pipeline.summariser import summarise_document
    from pipeline.storage import validate_summary, check_duplicate, save_summary
    init_db()
    return {
        "get_db_session":   get_db_session,
        "DocumentSummary":  DocumentSummary,
        "load_single_document": load_single_document,
        "split_documents":  split_documents,
        "summarise_document": summarise_document,
        "validate_summary": validate_summary,
        "check_duplicate":  check_duplicate,
        "save_summary":     save_summary,
    }

# Load pipeline — shows spinner on first load only
with st.spinner("Loading Avabodh..."):
    _pipeline = load_pipeline()

# Unpack for use throughout app
get_db_session      = _pipeline["get_db_session"]
DocumentSummary     = _pipeline["DocumentSummary"]
load_single_document = _pipeline["load_single_document"]
split_documents     = _pipeline["split_documents"]
summarise_document  = _pipeline["summarise_document"]
validate_summary    = _pipeline["validate_summary"]
check_duplicate     = _pipeline["check_duplicate"]
save_summary        = _pipeline["save_summary"]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }
    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #64748b;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    .summary-box {
        background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%);
        border-left: 4px solid #0f3460;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem 2rem;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #1e293b;
        margin: 1rem 0;
    }
    .meta-chip {
        display: inline-block;
        background: #e8f0fe;
        color: #1a56db;
        border-radius: 999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .step-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #475569;
    }
    .stButton > button {
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        color: white;
        padding: 0.55rem 1.5rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(15, 52, 96, 0.3);
    }
    .new-doc-btn > button {
        background: linear-gradient(135deg, #065f46, #047857) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "current_summary" not in st.session_state:
    st.session_state["current_summary"] = None

if "upload_key" not in st.session_state:
    # Changing this key forces file uploader to reset
    st.session_state["upload_key"] = 0

if "db_ready" not in st.session_state:
    st.session_state["db_ready"] = True

# ── Helper: reset for new document upload ────────────────────────────────────
def new_document():
    """Clear current summary and reset file uploader."""
    st.session_state["current_summary"] = None
    st.session_state["_last_streamed_id"] = None
    # Incrementing key forces Streamlit to destroy and recreate the uploader
    # This is the official Streamlit way to clear a file uploader
    st.session_state["upload_key"] += 1

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Avabodh")
    st.markdown("Document Intelligence Platform")
    st.divider()

    # NEW — tenant_id. This UI has no gateway in front of it, so there's
    # no header to read it from; the operator types it in directly. Every
    # document processed in this session is attributed to this tenant.
    tenant_id = st.text_input(
        "Tenant ID",
        value=st.session_state.get("tenant_id", ""),
        help="Documents processed in this session are scoped to this tenant.",
    )
    st.session_state["tenant_id"] = tenant_id
    if not tenant_id.strip():
        st.warning("Enter a Tenant ID above to process or view documents.")

    st.divider()

    # New Document button — taken from reference code's "New Chat" pattern
    if st.button("+ New Document", use_container_width=True):
        new_document()
        st.rerun()

    st.divider()

    if st.session_state.get("db_ready"):
        st.success("DB connected", icon="✅")
    else:
        st.error(f"DB error: {st.session_state.get('db_error', 'unknown')}", icon="❌")

    st.markdown("#### Previously Processed")

    if st.session_state.get("db_ready") and tenant_id.strip():
        try:
            with get_db_session() as session:
                records = (
                    session.query(DocumentSummary)
                    .filter(DocumentSummary.tenant_id == tenant_id)
                    .order_by(DocumentSummary.created_at.desc())
                    .limit(20)
                    .all()
                )
                if records:
                    for rec in records:
                        if st.button(f"📄 {rec.doc_name}", key=f"sidebar_{rec.id}",
                                     use_container_width=True):
                            # Load from DB and show — also clears uploader
                            st.session_state["current_summary"] = {
                                "doc_name":   rec.doc_name,
                                "summary":    rec.summary_text,
                                "chunks":     rec.chunk_count,
                                "pages":      rec.page_count,
                                "model":      rec.model_used,
                                "created_at": str(rec.created_at)[:19],
                                "id":         str(rec.id),
                            }
                            # Reset uploader so old file doesn't show
                            st.session_state["upload_key"] += 1
                            st.rerun()
                else:
                    st.caption("No documents yet — upload one!")
        except Exception as e:
            st.caption(f"Could not load history: {e}")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Avabodh</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a document. Get an intelligent summary. Stored forever.</div>',
            unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1], gap="large")

# ── Left column: Upload ───────────────────────────────────────────────────────
with col_upload:
    st.markdown("#### Upload Document")

    # upload_key changes when new_document() is called — forces uploader to reset
    uploaded_file = st.file_uploader(
        "Choose a PDF, TXT, DOCX or CSV file",
        type=["pdf", "txt", "docx", "csv"],
        help="File is processed locally — your data stays private",
        key=f"uploader_{st.session_state['upload_key']}",
    )

    if uploaded_file:
        st.markdown(f"""
        <div style='background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
                    padding:0.8rem 1rem;margin:0.5rem 0;font-size:0.9rem;color:#166534'>
            <b>Selected:</b> {uploaded_file.name} &nbsp;|&nbsp;
            <b>Size:</b> {uploaded_file.size / 1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)

    process_btn = st.button(
        "Generate Summary",
        disabled=uploaded_file is None or not tenant_id.strip(),
        use_container_width=True,
    )
    if uploaded_file is not None and not tenant_id.strip():
        st.caption("⚠️ Enter a Tenant ID in the sidebar before processing.")

    if process_btn and uploaded_file:
        temp_path = os.path.join("documents", uploaded_file.name)
        os.makedirs("documents", exist_ok=True)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.status("Processing document...", expanded=True) as status:

            # Step 1: Load
            st.markdown('<span class="step-label">Step 1 — Loading document</span>',
                       unsafe_allow_html=True)
            try:
                docs = load_single_document(temp_path)
                file_hash = docs[0].metadata.get("file_hash", "")
                page_count = len(docs)
                st.write(f"Loaded {page_count} page(s)")
            except Exception as e:
                status.update(label="Failed at loading", state="error")
                st.error(f"Load failed: {e}")
                st.stop()

            # Dedup check
            if file_hash:
                existing = check_duplicate(file_hash, tenant_id)
                if existing:
                    status.update(label="Already processed — showing cached result",
                                  state="complete")
                    st.info("This document was already summarised. Showing cached result.")
                    st.session_state["current_summary"] = {
                        "doc_name":   existing.doc_name,
                        "summary":    existing.summary_text,
                        "chunks":     existing.chunk_count,
                        "pages":      existing.page_count,
                        "model":      existing.model_used,
                        "created_at": str(existing.created_at)[:19],
                        "id":         str(existing.id),
                    }
                    # Clear uploader after processing
                    st.session_state["upload_key"] += 1
                    st.rerun()

            # Step 2: Split
            st.markdown('<span class="step-label">Step 2 — Splitting into chunks</span>',
                       unsafe_allow_html=True)
            try:
                chunks = split_documents(docs)
                del docs
                st.write(f"Created {len(chunks)} semantic chunks")
            except Exception as e:
                status.update(label="Failed at splitting", state="error")
                st.error(f"Split failed: {e}")
                st.stop()

            # Step 3: Summarise
            st.markdown('<span class="step-label">Step 3 — Summarising (Map + Reduce)</span>',
                       unsafe_allow_html=True)
            from config.settings import get_settings as _get_settings
            _settings = _get_settings()
            st.write(f"Processing {len(chunks)} chunks with {_settings.MAP_MODEL}...")

            try:
                raw_result = summarise_document(chunks, doc_name=uploaded_file.name)
                chunk_count = len(chunks)
                del chunks
            except Exception as e:
                status.update(label="Failed at summarisation", state="error")
                st.error(f"Summarisation failed: {e}")
                st.stop()

            # Step 4: Save
            st.markdown('<span class="step-label">Step 4 — Saving to PostgreSQL</span>',
                       unsafe_allow_html=True)
            try:
                metadata = {
                    "doc_name":    uploaded_file.name,
                    "source_path": temp_path,
                    "page_count":  page_count,
                }
                validated = validate_summary(raw_result, metadata)
                record = save_summary(validated, file_hash, chunk_count, tenant_id=tenant_id)
                st.write(f"Saved — ID: {str(record.id)[:8]}...")
            except Exception as e:
                status.update(label="Failed at saving", state="error")
                st.error(f"Save failed: {e}")
                st.stop()

            status.update(label="Done! Summary ready.", state="complete")

        st.session_state["current_summary"] = {
            "doc_name":   uploaded_file.name,
            "summary":    raw_result["summary_text"],
            "chunks":     chunk_count,
            "pages":      page_count,
            "model":      raw_result["reduce_model"],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "id":         str(record.id),
        }

        # Clear file uploader after successful processing
        st.session_state["upload_key"] += 1
        st.rerun()

# ── Right column: Summary ─────────────────────────────────────────────────────
with col_result:
    st.markdown("#### Summary")

    if st.session_state["current_summary"] is None:
        st.markdown("""
        <div style='height:300px;display:flex;align-items:center;justify-content:center;
                    border:2px dashed #e2e8f0;border-radius:12px;color:#94a3b8;font-size:0.95rem;'>
            Upload a document to see its summary here
        </div>
        """, unsafe_allow_html=True)

    else:
        data = st.session_state["current_summary"]

        st.markdown(f"""
        <div style='margin-bottom:1rem'>
            <span class='meta-chip'>📄 {data['doc_name']}</span>
            <span class='meta-chip'>📃 {data['pages']} pages</span>
            <span class='meta-chip'>🧩 {data['chunks']} chunks</span>
            <span class='meta-chip'>🤖 {data.get('model','phi3.5')}</span>
            <span class='meta-chip'>🕐 {data['created_at']}</span>
        </div>
        """, unsafe_allow_html=True)

        summary_text = data["summary"]

        def stream_summary():
            for word in summary_text.split():
                yield word + " "
                time.sleep(0.03)

        if st.session_state.get("_last_streamed_id") != data["id"]:
            streamed = st.write_stream(stream_summary())
            # After streaming, replace with formatted version
            formatted = streamed
            for heading in ["DOCUMENT OVERVIEW", "KEY FINDINGS", "MAIN TOPICS COVERED", "CONCLUSION"]:
                formatted = formatted.replace(
                    heading,
                    f"\n\n**{heading}**\n"
                )
            st.session_state["_last_streamed_id"] = data["id"]
        else:
            # Format headings bold and add line breaks
            formatted = summary_text
            for heading in ["DOCUMENT OVERVIEW", "KEY FINDINGS", "MAIN TOPICS COVERED", "CONCLUSION"]:
                formatted = formatted.replace(
                    heading,
                    f"<br><strong style='font-size:1rem;color:#0f3460;letter-spacing:0.05em'>{heading}</strong><br>"
                )
            st.markdown(
                f'<div class="summary-box">{formatted}</div>',
                unsafe_allow_html=True,
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                label="Download Summary",
                data=summary_text,
                file_name=f"{data['doc_name']}_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_b:
            if st.button("New Document", use_container_width=True):
                new_document()
                st.rerun()