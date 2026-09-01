#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab02

import streamlit as st
from openai import OpenAI

###
# ----------------- INSERTED SIDEBAR (VISUAL ONLY) -----------------EXAMPLE STARTS/PLEASE DELETE
#st.sidebar.header("Controls & Info")
#st.sidebar.caption("Visual prototype only — does not affect execution.")
#st.sidebar.radio("View Mode", ["Standard", "Detailed Analysis", "Debug"])
#st.sidebar.selectbox("Filter Category", ["All Sections", "Executive Summary", "Key Points"])
#st.sidebar.checkbox("Highlight Citations", value=True)
#st.sidebar.slider("Chunk Window Size", min_value=100, max_value=1000, value=500, step=50)
#st.sidebar.button("Clear Cache")
# -------------------------------------------------------------------EXAMPLE ENDS/PLEASE DELETE

with st.sidebar:
    add_radio = st.radio("Shipping", ("hahah","lololol"))#Delete

# Show title and description.
st.title(":blue[Lab 2:] :grey[Deep] Scan Protocol")  #Updated title

st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
)


@st.cache_data  #Caches result
def is_valid_key(key: str) -> bool:  #Validation function
    try:
        OpenAI(api_key=key).models.list()  #Checks key
        return True
    except Exception:
        return False


# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")  #Key read from .streamlit/secrets.toml (or App settings > Secrets)

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
elif not is_valid_key(openai_api_key):  #Validate the API key when entered
    st.error("Invalid API key. Please try again.")  #Error displayed
else:
    st.success("Access granted!")  #Confirmation

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        model_options = ["gpt-3.5-turbo", "gpt-4.1", "gpt-5-chat-latest", "gpt-5-nano"]  #Available models
        selected_model = st.selectbox("Model",
            model_options,
            index=None, #Nothing preselected
            placeholder="Choose a model", #Shown while the selectbox is empty
        )

        if selected_model: #No generation until user sleects a model

            # Generate an answer using the OpenAI API.
            stream = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                stream=True,
            )

            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)
