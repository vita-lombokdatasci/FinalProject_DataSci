import getpass
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Minta user masukkan API key
GOOGLE_API_KEY = getpass.getpass("Enter your API key: ")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print()

# Inisiasi client LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Inisiasi chat history dengan hanya system message
messages_history: list = [
    SystemMessage(
        "You are a comedian that knows a lot about Bali. Always response in less than 3 sentences in a chat style. Reply in bahasa indonesia"
    )
]

# Lakukan ini berulang2
while True:
    # Minta prompt terbaru dari user
    user_prompt = input("User: ")
    # Tambahkan prompt ke chat history dan tanyakan ke LLM
    messages_history.append(HumanMessage(user_prompt))
    response = llm.invoke(messages_history)
    # Simpan jawaban LLM ke chat history dan tampilkan jawabannya
    messages_history.append(response)
    print(f"AI: {response.content}")