#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab02

import streamlit as st
from openai import OpenAI

st.sidebar.header("**Settings:**")
st.sidebar.caption("Configure Output Format & AI Model")

#Summary Type
summary_type = st.sidebar.selectbox("**Specify Output Format**", ["100-Word Summary", "2 Paragraph Summary", "5-Bullet Summary"],
    index=None,
    placeholder="Choose a format",
)  #Stored

#Model Selection
base_model = st.sidebar.selectbox("**Select Model**", ["gpt-3.5-turbo", "gpt-5-nano", "gpt-4o-mini", "gpt-4.1",],
    index=None,
    placeholder="Choose a model",
)  #Stored

use_advanced = st.sidebar.checkbox("Use Advanced Model", value=False)  #When checked the advanced model is used instead
advanced_model = "gpt-4.1"
selected_model = advanced_model if use_advanced else base_model  #Model selection sent to the API

if st.sidebar.button("Clear Cache"):  #Clears the cached key validation
    st.cache_data.clear()

generate = st.sidebar.button("Generate Summary", type="primary")  #Nothing is sent to the API until this is clicked
inputs_ready = bool(summary_type) and bool(selected_model)  #A format is always required, and a model must come from the dropdown or the checkbox

# Show title and description.
st.title(":blue[Lab 2:] :grey[Deep] Scan Protocol")  #Updated title

st.write(
    "Upload a document below, then select summary format and model. "
)

if generate and not inputs_ready:  #Error shown when either sidebar selection is missing
    st.error("Error! Please choose a summary format and a model (or check Use Advanced Model) before generating.")

@st.cache_data  #Caches result
def is_valid_key(key: str) -> bool:  #Validation function
    try:
        OpenAI(api_key=key).models.list()  #Checks key
        return True
    except Exception:
        return False

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")  #Key read from .streamlit/secrets.toml (or App settings > Secrets)

summary_instructions = {
    "100-Word Summary": "Summarize the document in about 100 words.",
    "2 Paragraph Summary": "Summarize the document in 2 connecting paragraphs.",
    "5-Bullet Summary": "Summarize the document in 5 bullet points.",
}  #Maps selection to the instruction sent to the LLM

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

    if uploaded_file and generate and inputs_ready:  #Runs once selections are made
        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {summary_instructions[summary_type]}", #Summary format is now the instruction
            }
        ]

        if selected_model: #No generation until user selects a model

            # Generate an answer using the OpenAI API.
            stream = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                stream=True,
            )

            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream) #Karisa is BB