# app.py
import streamlit as st
import requests
import os
import io
import tempfile
from typing import List

# LangChain & Google Generative AI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# PDF parsing
import pdfplumber

# ---- Helper: GitHub file listing & raw download ----
GITHUB_OWNER = "vita-lombokdatasci"
GITHUB_REPO = "FinalProject_DataSci"
GITHUB_API_BASE = "https://api.github.com"

def list_repo_files(path=""):
    """Return list of files (dicts) in repo path via GitHub API (public repo)."""
    url = f"{GITHUB_API_BASE}/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{path}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()  # list of file dicts

def raw_url_from_fileinfo(fileinfo):
    """Given GitHub API file object, return raw URL."""
    return fileinfo.get("download_url")

def download_bytes(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.content

# ---- Helper: PDF -> text ----
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text_chunks = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n\n".join(text_chunks)

# ---- Build documents from PDFs in repo ----
def build_documents_from_repo():
    st.info("Mengambil daftar file dari GitHub...")
    try:
        contents = list_repo_files()
    except Exception as e:
        st.error(f"Gagal mengakses GitHub: {e}")
        return [], []

    pdf_files = [f for f in contents if f["name"].lower().endswith(".pdf")]
    image_files = [f for f in contents if f["name"].lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))]

    docs: List[Document] = []
    images: List[dict] = []

    # Progress bar untuk UX yang lebih baik
    progress_bar = st.progress(0)
    total_files = len(pdf_files)

    for idx, p in enumerate(pdf_files):
        raw = raw_url_from_fileinfo(p)
        # st.write(f"Memproses PDF: {p['name']}") # Optional: kurangi clutter UI
        try:
            b = download_bytes(raw)
            text = extract_text_from_pdf_bytes(b)
            if not text.strip():
                st.warning(f"Tidak ada teks diekstrak dari {p['name']}")
                continue
            
            # Create Document with metadata
            docs.append(Document(page_content=text, metadata={"source": p["name"], "url": raw}))
        except Exception as e:
            st.warning(f"Gagal memproses {p['name']}: {e}")
        
        # Update progress
        progress_bar.progress((idx + 1) / total_files)

    progress_bar.empty() # Hapus bar setelah selesai

    for im in image_files:
        images.append({"name": im["name"], "raw_url": raw_url_from_fileinfo(im)})

    return docs, images

# ---- Create vectorstore (FAISS) with Google Embeddings ----
def create_vectorstore(docs: List[Document], google_api_key: str):
    st.info("Membangun embeddings & index dengan Google AI...")
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Menggunakan model embedding Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = []
    metadatas = []
    for d in docs:
        pieces = splitter.split_text(d.page_content)
        for i, p in enumerate(pieces):
            texts.append(p)
            md = d.metadata.copy()
            md.update({"chunk_index": i})
            metadatas.append(md)
            
    # Init FAISS
    vect = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vect

# ---- Simple intent: detect if user asks for images ----
IMAGE_KEYWORDS = ["gambar","foto","foto nya","lihat gambar","lihat foto","show image","show images","pictures","images","foto-foto","gallery","galeri", "visual"]

def wants_images(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in IMAGE_KEYWORDS)

# ---- Streamlit UI ----
st.set_page_config(page_title="Luna — Sky Villa Assistant", layout="wide")

# CSS
st.markdown("""
<style>
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Sky Villa logo
logo_url = "https://raw.githubusercontent.com/vita-lombokdatasci/FinalProject_DataSci/main/logo_SkyVilla_tetebatu.jpg"
try:
    cols = st.columns([1, 4, 1])
    with cols[0]:
        st.image(logo_url, width=150)
except Exception:
    st.warning("Gagal memuat logo Sky Villa dari GitHub.")

st.title("Luna — Sky Villa Assistant")
st.markdown("**Marketing Automation**")

st.sidebar.header("Settings")
google_key = st.sidebar.text_input("Google API Key", type="password", help="Dapatkan API Key di aistudio.google.com")
use_index = st.sidebar.checkbox("Update Index dari GitHub", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**WhatsApp Booking**")
st.sidebar.markdown("[Chat via WhatsApp](https://wa.me/6285253628371)")

# Logic Building Index
if st.sidebar.button("Build / Rebuild Index"):
    use_index = True

if use_index:
    if not google_key:
        st.sidebar.error("Masukkan Google API Key terlebih dahulu.")
    else:
        with st.spinner("Mendownload PDF dari GitHub dan membuat Index..."):
            try:
                docs, images = build_documents_from_repo()
                if not docs:
                    st.error("Tidak ada PDF yang bisa diindeks atau repo kosong.")
                else:
                    vect = create_vectorstore(docs, google_key)
                    
                    # Save to temp files
                    tmpdir = tempfile.gettempdir()
                    store_path = os.path.join(tmpdir, "luna_faiss_index")
                    vect.save_local(store_path)
                    
                    st.success(f"Berhasil mengindeks {len(docs)} dokumen!")
                    
                    # Store context in session
                    st.session_state["vectorstore_path"] = store_path
                    st.session_state["images"] = images
            except Exception as e:
                st.error(f"Terjadi Error saat indexing: {e}")

# Restore index if present
vect = None
if "vectorstore_path" in st.session_state and google_key:
    try:
        os.environ["GOOGLE_API_KEY"] = google_key
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        # allow_dangerous_deserialization=True diperlukan untuk local load di versi baru langchain
        vect = FAISS.load_local(st.session_state["vectorstore_path"], embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.warning("Index lama tidak valid. Silakan tekan 'Build / Rebuild Index'.")

# ---- Chat Area ----
st.subheader("Tanya Luna tentang Sky Villa")
st.caption("Luna akan menjawab berdasarkan dokumen PDF yang ada di GitHub Anda.")

# Image Pre-fetch fallback
if "images" not in st.session_state:
    try:
        contents = list_repo_files()
        image_files = [f for f in contents if f["name"].lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))]
        st.session_state["images"] = [{"name": im["name"], "raw_url": raw_url_from_fileinfo(im)} for im in image_files]
    except Exception:
        st.session_state["images"] = []

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (Optional, but good for UX)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input User
if prompt := st.chat_input("Ketik pertanyaan anda (misal: 'Apa fasilitas unggulan Sky Villa?')"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Logic Jawaban
    if wants_images(prompt):
        with st.chat_message("assistant"):
            st.info("Luna sedang mencari foto di galeri...")
            images = st.session_state.get("images", [])
            if not images:
                st.warning("Maaf, saya tidak menemukan file gambar di repository GitHub tersebut.")
                response_text = "Maaf, tidak ada gambar ditemukan."
            else:
                # Tampilkan gambar dalam grid
                cols = st.columns(3)
                for i, im in enumerate(images):
                    c = cols[i % 3]
                    c.image(im["raw_url"], caption=im["name"], use_column_width=True)
                response_text = "Berikut adalah koleksi foto Sky Villa yang saya temukan di database."
                st.markdown(response_text)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})

    else:
        if vect is None:
            with st.chat_message("assistant"):
                st.error("Index pengetahuan belum siap. Masukkan Google API Key di sidebar dan klik 'Build Index'.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Luna sedang membaca dokumen..."):
                    # Konfigurasi Google Gemini LLM
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash", 
                        temperature=0.3,
                        google_api_key=google_key,
                        convert_system_message_to_human=True 
                    )
                    
                    retriever = vect.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                    qa = RetrievalQA.from_chain_type(
                        llm=llm, 
                        chain_type="stuff", 
                        retriever=retriever, 
                        return_source_documents=True
                    )
                    
                    try:
                        res = qa.invoke(prompt) # Gunakan .invoke() bukan qa() di versi baru
                        answer = res["result"]
                        docs = res.get("source_documents", [])
                        
                        st.markdown(answer)
                        
                        # Tampilkan sumber (Expandable untuk kerapian)
                        with st.expander("Lihat Sumber Referensi PDF"):
                            for d in docs:
                                src = d.metadata.get("source", "unknown")
                                st.markdown(f"- **{src}**: ...{d.page_content[:100]}...")
                                
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        st.error(f"Terjadi kesalahan koneksi ke Google AI: {e}")