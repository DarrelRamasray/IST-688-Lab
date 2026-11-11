#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab08

import base64
from pathlib import Path
import streamlit as st
from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  #Step 2: same secret name as previous labs

def prepare_image(upload):  #Step 3: packages the image for the Responses API
    formats = {".png": "image/png", ".jpg": "image/jpeg",
               ".jpeg": "image/jpeg"}
    mime = formats[Path(upload.name).suffix.lower()]
    encoded = base64.b64encode(upload.getvalue()).decode("utf-8")
    return {"type": "input_image",
            "image_url": f"data:{mime};base64,{encoded}"}

def transcribe_audio(upload):  #Step 3: sends audio to the transcription endpoint
    transcript = client.audio.transcriptions.create(
        model="gpt-transcribe",
        file=(upload.name, upload.getvalue()))
    return transcript.text

def analyze_content(content, prompt):  #Step 3: sends content and instructions to the model
    response = client.responses.create(
        model="gpt-4.1-mini", instructions=prompt,
        input=[{"role": "user", "content": content}],
        max_output_tokens=600)
    return response.output_text

st.title("Lab 8 - The See and Hear Assistant")  #Page title
st.write(
    "Choose Image or Audio in the sidebar, upload a file, optionally ask a question, then click Analyze. "
)  #Step 5a: page description, same pattern as Lab 2

mode = st.sidebar.radio("Input type", ["Image", "Audio"], key="lab8_mode")  #Step 5a: Image or Audio selector

if mode == "Image":
    file_types = ["png", "jpg", "jpeg"]  #Step 5b-i: allowed image formats
    upload_label = "Upload an image (PNG, JPG, or JPEG, 5 MB max)"
else:
    file_types = ["mp3", "wav", "m4a"]  #Step 5b-i: allowed audio formats
    upload_label = "Upload an audio clip (MP3, WAV, or M4A, 5 MB max, 60 seconds max)"

upload = st.file_uploader(
    upload_label,
    type=file_types,
    key=f"lab8_upload_{mode}",
)  #Step 5a: file uploader; a separate key per mode clears the file when you switch modes

if upload:
    if mode == "Image":
        st.image(upload, caption=upload.name)  #Step 5b: image preview
    else:
        st.audio(upload, format=upload.type)  #Step 5b: audio playback

question = st.text_area(
    "Ask a question about the upload (optional)",
    placeholder="Leave blank to get a summary only.",
    disabled=not upload,
    key="lab8_question",
)  #Step 5a: optional question, same disabled-until-upload pattern as Lab 2

analyze = st.button("Analyze", type="primary")  #Step 5a: Analyze button

if analyze:  #Temporary: Step 7 replaces this block
    st.info("The Analyze button gets connected in Step 7.")
