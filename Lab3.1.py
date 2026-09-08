#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab03

import streamlit as st
from openai import OpenAI

model_to_use = "gpt-5.4-mini"

st.title(":blue[Lab 3:] :grey[Deep] Chatbot")
st.write("Ask me anything!")

if "client" not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    
    #try:
    #    api_key = st.secrets["OPENAI_API_KEY"]
    #except Exception:
    #    api_key = None

    if not api_key:
        st.error("OPENAI_API_KEY invalid!")
        st.stop()

    st.session_state.client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client

    try:
        stream = client.chat.completions.create(
            model = model_to_use,
            messages=st.session_state.messages,
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)

    except Exception as e:
        st.error(f"This request has failed: {e}")
        st.stop()

    st.session_state.messages.append({"role": "assistant", "content": response})