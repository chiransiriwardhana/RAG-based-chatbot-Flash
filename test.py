import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from PIL import Image
import io
import hashlib
import base64

# =====================================================
# Page config
# =====================================================
st.set_page_config(page_title="Flash", layout="centered")

# =====================================================
# Session state
# =====================================================
st.session_state.setdefault("text_messages", [])
st.session_state.setdefault("image_threads", [])
st.session_state.setdefault("last_image_hash", None)
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("pending_image", None)  # NEW

# =====================================================
# Sidebar
# =====================================================
st.sidebar.markdown("### Settings")

TEXT_MODELS = ["llama3.2", "mistral", "qwen2.5"]
text_model = st.sidebar.selectbox("Text model", TEXT_MODELS)

MODES = ["Chat", "Image Q&A", "About"]
mode = st.sidebar.selectbox("Mode", MODES)

ANSWER_SOURCE = ["Auto", "Documents only", "Model knowledge only"]
answer_source = st.sidebar.radio("Answer source", ANSWER_SOURCE)

if st.sidebar.button("Clear chats"):
    st.session_state.text_messages.clear()
    st.session_state.image_threads.clear()
    st.session_state.last_image_hash = None
    st.session_state.pending_image = None
    st.session_state.uploader_key += 1
    st.rerun()

# =====================================================
# Vectorstore (RAG)
# =====================================================
@st.cache_resource
def load_vectorstore(pdf_paths):
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="llama3.2")
    return Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")

PDF_FILES = [
    "globalwarming.pdf",
    "/Users/chiransiriwardena/Documents/Rag_chatbot/pdf/pop.pdf"
]

vectorstore = load_vectorstore(PDF_FILES)

# =====================================================
# LLM helpers
# =====================================================
def image_answer(image_file, question: str) -> str:
    llm = OllamaLLM(model="llava")
    img = Image.open(image_file)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return llm.invoke(question, images=[buf.getvalue()])

# =====================================================
# Header
# =====================================================
with open("icons/logo.png", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <h1 style="display:flex;align-items:center;gap:5px;">
        <img src="data:image/png;base64,{encoded}" width="45">
        Flash
    </h1>
    """,
    unsafe_allow_html=True
)
st.divider()

# =====================================================
# IMAGE Q&A MODE
# =====================================================
if mode == "Image Q&A":
    st.subheader("Image Q&A")

    # Display existing threads
    for thread in st.session_state.image_threads:
        st.image(thread["image"], width=350)
        for msg in thread["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    st.divider()

    # ===============================
    # ChatGPT-style input row
    # ===============================
    col1, col2 = st.columns([1, 6])

    with col1:
        uploaded = st.file_uploader(
            "📎",
            type=["png", "jpg", "jpeg"],
            key=f"attach_{st.session_state.uploader_key}",
            label_visibility="collapsed"
        )
        if uploaded:
            st.session_state.pending_image = uploaded
            st.image(uploaded, width=80)

    with col2:
        user_input = st.text_input("Ask about the image", key="image_input")

    if st.button("Send") and user_input:
        if not st.session_state.pending_image:
            st.warning("Attach an image first.")
        else:
            image_file = st.session_state.pending_image

            # New image thread
            st.session_state.image_threads.append({
                "image": image_file,
                "messages": [{"role": "user", "content": user_input}]
            })

            with st.spinner("Analyzing image..."):
                ans = image_answer(image_file, user_input)

            st.session_state.image_threads[-1]["messages"].append(
                {"role": "assistant", "content": ans}
            )

            # Reset attach state
            st.session_state.pending_image = None
            st.session_state.uploader_key += 1
            st.rerun()

# =====================================================
# ABOUT
# =====================================================
elif mode == "About":
    st.markdown("""
### About AmazBot 🤖
- Text chat with RAG
- Image-based Q&A (LLaVA)
- Offline inference via Ollama
""")
