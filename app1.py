import streamlit as st

# Bikin judul
st.title("My ChatBot Assistant")

# text input
nama = st.text_input("Masukkan nama anda")
# Tampilkan nama jika namanya tidak kosong
if nama != "":
    st.markdown(f"Halo {nama}")

# Minta umur dalam bentuk slider dan tampilkan
umur = st.slider("Umur", min_value=10, max_value=50)
st.markdown(f"Umur: {umur}")

# Munculin balon jika pencet submit
submit_pressed = st.button("Submit")
if submit_pressed:
    st.balloons()