#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab02

import streamlit as st
from openai import OpenAI

st.sidebar.header("**Settings:**")
st.sidebar.caption("Configure Output Format & AI Model")
st.sidebar.selectbox("**Specify Output Format**", ["100-Word Summary", "2 Paragraph Summary", "5-Bullet Summary"])
model_options = [
    "gpt-3.5-turbo", 
    "gpt-5-nano", 
    "gpt-4o-mini", 
    "gpt-4.1", 
    "gpt-5.6-luna", 
    "gpt-5.6-terra", 
    "gpt-5.6-sol"
]

model_labels = {
    "gpt-3.5-turbo": "gpt-3.5-turbo (Legacy)",
    "gpt-5-nano": "gpt-5-nano (Micro)",
    "gpt-4o-mini": "gpt-4o-mini (Budget)",
    "gpt-4.1": "gpt-4.1 (Advanced)",
    "gpt-5.6-luna": "gpt-5.6-luna (Modern Efficiency)",
    "gpt-5.6-terra": "gpt-5.6-terra (Balanced Frontier)",
    "gpt-5.6-sol": "gpt-5.6-sol (Absolute Best)",
}

selected_model = st.sidebar.selectbox(
    "**Select Model**",
    options=model_options,
    format_func=lambda model_id: model_labels.get(model_id, model_id),
    index=2  # Defaults to "gpt-4o-mini"
)
#st.sidebar.selectbox("**Specify AI Model**", ["                  "])
st.sidebar.checkbox("Use Advanced Model", value=False)
st.sidebar.button("Clear Cache")


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

        ###model_options = ["gpt-3.5-turbo", "gpt-4o-mini","gpt-4.1", "gpt-5-nano"]  #Available models ##################################################

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
