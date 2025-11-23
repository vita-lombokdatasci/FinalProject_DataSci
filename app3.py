"""
Untuk jalankan,

streamlit run app3.py
"""

import os

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Bikin judul
st.title("My ChatBot")

# Cek apakah API key sudah ada
if "GOOGLE_API_KEY" not in os.environ:
    # Jika belum, minta user buat masukin API key
    google_api_key = st.text_input("Google API Key", type="password")
    # User harus klik Start untuk save API key
    start_button = st.button("Start")
    if start_button:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        st.rerun()
    # Jangan tampilkan chat dulu kalau belum pencet start
    st.stop()

# Inisiasi client LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Cek apakah data sebelumnya ttg message history sudah ada
if "messages_history" not in st.session_state:
    # Jika belum, bikin datanya, isinya hanya system message dulu
    st.session_state["messages_history"] = [
        SystemMessage(
            "You are a comedian that knows a lot about Bali. Always response in less than 3 sentences in a chat style. Reply in bahasa indonesia"
        )
    ]
# Jika messages_history sudah ada, tinggal di load aja
messages_history = st.session_state["messages_history"]

# Tampilkan messages history selama ini
for message in messages_history:
    # Tdk perlu tampilkan system message
    if type(message) is SystemMessage:
        continue
    # Pilih role, apakah user/AI
    role = "User" if type(message) is HumanMessage else "AI"
    # Tampikan chatnya!
    with st.chat_message(role):
        st.markdown(message.content)

# Baca prompt terbaru dari user
prompt = st.chat_input("Chat with AI")
if not prompt:
    st.stop()
# Jika user ada prompt, tampilkan promptnya langsung
with st.chat_message("User"):
    st.markdown(prompt)
# Masukin prompt ke message history, dan kirim ke LLM
messages_history.append(HumanMessage(prompt))
response = llm.invoke(messages_history)

# Simpan jawaban LLKM ke message history
messages_history.append(response)
# Tampilkan langsung jawaban LLM
with st.chat_message("AI"):
    st.markdown(response.content)