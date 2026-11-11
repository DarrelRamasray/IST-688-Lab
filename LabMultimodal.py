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

st.title("Lab 8 - The See and Hear Assistant")  #Temporary title so the page shows something; Part B builds the rest
