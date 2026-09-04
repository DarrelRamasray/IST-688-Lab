#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab02
import streamlit as st
from openai import OpenAI
import time  #Used to pace the loader steps

##***
st.markdown(  #Tightens the default spacing around sidebar dividers and headings
    """
    <style>
    section[data-testid="stSidebar"] hr { margin: 0.75rem 0; }
    section[data-testid="stSidebar"] h3 { margin-top: 0.25rem; margin-bottom: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
##***

#st.sidebar.header("**Settings:**")
##***
st.sidebar.header(":material/settings: **Settings:**")  #Header with icon
##***
st.sidebar.caption("Configure Output Format & AI Model")
##***
st.sidebar.divider()  #Separates the settings caption from the first section
##***

#Output Language
st.sidebar.subheader(":material/translate: Language")  #Section heading with icon
st.sidebar.caption("Select Language")  #Caption sits above the dropdown
#language = st.sidebar.selectbox("**Language**", ["English", "Mandarin Chinese", "Hindi", "Spanish", "French"],
#    index=0,  #English preselected
#)  #Stored
#st.sidebar.caption("Select Language")  #Caption for the dropdown above
language = st.sidebar.selectbox("Language", ["English", "Mandarin Chinese", "Hindi", "Spanish", "French"],
    index=0,  #English preselected
    label_visibility="collapsed",  #Hidden so the caption above acts as the label
)  #Stored

##***
st.sidebar.divider()  #Separates the language section from the format section
##***

#Summary Type
st.sidebar.subheader(":material/description: Specify Output Format")  #Section heading with icon
st.sidebar.caption("Select type of summary")  #Caption sits above the dropdown
#summary_type = st.sidebar.selectbox("**Specify Output Format**", ["100-Word Summary", "2 Paragraph Summary", "5-Bullet Summary"],
#    index=None,
#    placeholder="Choose a format",
#)  #Stored
summary_type = st.sidebar.selectbox("Specify Output Format", ["100-Word Summary", "2 Paragraph Summary", "5-Bullet Summary"],
    index=None,
    placeholder="Choose a format",
    label_visibility="collapsed",  #Hidden so the caption above acts as the label
)  #Stored

st.sidebar.divider()  #Separates the model section

#Model Selection
st.sidebar.subheader(":material/computer: Model Selection")  #Section heading with icon

use_advanced = st.sidebar.checkbox("Use Advanced Model", value=False)  #Switches between the two models below
#base_model = st.sidebar.selectbox("**Select AI Model**", ["gpt-3.5-turbo", "gpt-5-nano", "gpt-4o-mini",],
#    index=None,
#    placeholder="Choose a model",
#    disabled=use_advanced,  #Greyed out when the advanced model is in use
#)  #Stored
#advanced_model = "gpt-4.1"
#selected_model = advanced_model if use_advanced else base_model  #Model selection sent to the API
basic_model = "gpt-5.4-nano"  #Model used by default
advanced_model = "gpt-5.4-mini"  #Model used when the box above is checked
selected_model = advanced_model if use_advanced else basic_model  #Model selection sent to the API
st.sidebar.caption("_Now using model GPT-5.4 Mini._" if use_advanced else "_You are using model GPT-5.4 Nano._")  #Shows which model is active

#if st.sidebar.button("Clear Cache"):  #Clears the cached key validation
#    st.cache_data.clear()

generate = st.sidebar.button("Generate Summary", type="primary")  #Nothing is sent to the API until this is clicked

inputs_ready = bool(summary_type)  #A model is always set now, so only the format has to be chosen

# Show title and description.
st.title(":blue[Lab 2:] :grey[Deep] Scan Protocol")  #Updated title
st.write(
    "Upload a document below, then select summary format and model. "
)

if generate and not inputs_ready:  #Error shown when either sidebar selection is missing
    st.error("Error! Please choose a summary format before generating.")  #Model no longer needs selecting

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
                "content": f"Here's a document: {document} \n\n---\n\n {summary_instructions[summary_type]} Write the entire summary in {language}.", #Summary format and output language are now the instruction
            }
        ]
        if selected_model: #No generation until user selects a model
            # Generate an answer using the OpenAI API.
            with st.status(":material/radar: Initializing Deep Scan Protocol...", expanded=True) as scan:  #Live loader while the request runs
                st.write(f":material/description: Parsing document — {len(document.split())} words detected")
                time.sleep(0.4)
                st.write(f":material/memory: Routing request to {selected_model}")
                time.sleep(0.4)
                st.write(f":material/translate: Output language set to {language}")
                time.sleep(0.4)
                st.write(":material/bolt: Establishing stream...")
                stream = client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    stream=True,
                )
                scan.update(label=":material/check_circle: Scan complete", state="complete", expanded=False)  #Collapses once the stream is open
#            stream = client.chat.completions.create(
#                model=selected_model,
#                messages=messages,
#                stream=True,
#            )
            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)
