import streamlit as st
import io
import hashlib
import base64
from modules.vectorstore_utils import load_vectorstore, retrieve_context
from modules.llm_utils import answer_from_own_knowledge, answer_from_document, image_answer

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

# =====================================================
# Sidebar (always visible)
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
    st.session_state.uploader_key += 1
    st.rerun()

# =====================================================
# Vectorstore (RAG)
# =====================================================
PDF_FILES = [
    "globalwarming.pdf",
    # Add more PDF paths as needed
]
vectorstore = load_vectorstore(PDF_FILES)

# =====================================================
# LLM helpers (imported)
# =====================================================

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
# CHAT MODE
# =====================================================
if mode == "Chat":
    st.subheader("Chat")
    for msg in st.session_state.text_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# =====================================================
# IMAGE Q&A MODE
# =====================================================
elif mode == "Image Q&A":
    st.subheader("Image Q&A")

    # --- File uploader ---
    uploaded_file = st.file_uploader(
        "",
        type=["png", "jpg", "jpeg"],
        key=f"image_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image_hash = hashlib.md5(image_bytes).hexdigest()

        if st.session_state.last_image_hash != image_hash:
            st.session_state.image_threads.append({
                "image_bytes": image_bytes, 
                "messages": []
            })
            st.session_state.last_image_hash = image_hash

    # --- Render threads safely ---
    for thread in st.session_state.image_threads:
        st.image(thread["image_bytes"], width=250)
        for msg in thread["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

# =====================================================
# ABOUT
# =====================================================
else:
    st.markdown("""
### About AmazBot 🤖
- Text chat with RAG
- Image-based Q&A (LLaVA)
- Offline inference via Ollama
""")

# =====================================================
# CHAT INPUT
# =====================================================
if mode != "About":
    user_input = st.chat_input("Ask a question")

    if user_input:

        # ---------------- TEXT CHAT ----------------
        if mode == "Chat":
            st.session_state.text_messages.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)


            if answer_source == "Model knowledge only":
                with st.spinner("Thinking..."):
                    final_answer = answer_from_own_knowledge(user_input, text_model)

            elif answer_source == "Documents only":
                with st.spinner("Searching documents..."):
                    context = retrieve_context(vectorstore, user_input)
                with st.spinner("Thinking..."):
                    final_answer = (
                        answer_from_document(user_input, context, text_model)
                        if context.strip()
                        else "ANSWER NOT IN DOCUMENT"
                    )

            else:  # Auto
                with st.spinner("Thinking..."):
                    own_answer = answer_from_own_knowledge(user_input, text_model)

                if own_answer != "I DON'T KNOW":
                    final_answer = own_answer
                else:
                    with st.spinner("Searching documents..."):
                        context = retrieve_context(vectorstore, user_input)
                    with st.spinner("Thinking..."):
                        final_answer = (
                            answer_from_document(user_input, context, text_model)
                            if context.strip()
                            else "ANSWER NOT IN DOCUMENT"
                        )

            st.session_state.text_messages.append(
                {"role": "assistant", "content": final_answer}
            )
            st.rerun()

        # ---------------- IMAGE CHAT ----------------
        elif mode == "Image Q&A":
            with st.chat_message("user"):
                st.markdown(user_input)

            if not st.session_state.image_threads:
                st.warning("Upload an image first.")
            else:
                thread = st.session_state.image_threads[-1]
                thread["messages"].append(
                    {"role": "user", "content": user_input}
                )

                with st.spinner("Analyzing image..."):
                    # Convert bytes back to file-like object
                    image_file_like = io.BytesIO(thread["image_bytes"])
                    ans = image_answer(image_file_like, user_input)

                thread["messages"].append(
                    {"role": "assistant", "content": ans}
                )
                st.rerun()