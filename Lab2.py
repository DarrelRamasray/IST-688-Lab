#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab02
import streamlit as st
from openai import OpenAI

basic_model = "gpt-5.4-nano" #Model used by default
basic_model_name = "GPT-5.4 Nano" #Caption
advanced_model = "gpt-5.4-mini"  #Model used when the advanced box is checked
advanced_model_name = "GPT-5.4 Mini"  #Caption

UI = {
    "English": {
        "header": "**Settings:**",
        "caption": "Configure Output Format & AI Model",
        "lang_label": "**Language**",
        "lang_caption": "Select Language",
        "format_label": "**Specify Output Format**",
        "format_placeholder": "Choose a format",
        "formats": {
            "100-Word Summary": "100-Word Summary",
            "2 Paragraph Summary": "2 Paragraph Summary",
            "5-Bullet Summary": "5-Bullet Summary",
        },
        "model_label": "**Model Selection**",
        "using_basic": f"_You are using {basic_model_name}_",
        "using_advanced": f"_Now using {advanced_model_name}_",
        "advanced": "Use Advanced Model",
        "clear_cache": "Clear Cache",
        "generate": "Generate Summary",
        "dark_mode": "Dark Mode",
        "description": "Upload a document below, then select summary format and model.",
        "error": "Error! Please choose a summary format before generating.",
        "no_key": "Please add your OpenAI API key to continue.",
        "bad_key": "Invalid API key. Please try again.",
        "granted": "Access granted!",
        "uploader": "Upload a document (.txt or .md)",
    },
    "Mandarin Chinese": {
        "header": "**设置：**",
        "caption": "配置输出格式和 AI 模型",
        "lang_label": "**语言**",
        "lang_caption": "选择语言",
        "format_label": "**指定输出格式**",
        "format_placeholder": "选择格式",
        "formats": {
            "100-Word Summary": "100 字摘要",
            "2 Paragraph Summary": "两段连贯摘要",
            "5-Bullet Summary": "5 个要点摘要",
        },
        "model_label": "**模型选择**",
        "using_basic": f"_您正在使用 {basic_model_name}_",
        "using_advanced": f"_现在使用 {advanced_model_name}_",
        "advanced": "使用高级模型",
        "clear_cache": "清除缓存",
        "generate": "生成摘要",
        "dark_mode": "深色模式",
        "description": "请在下方上传文档，然后选择摘要格式。",
        "error": "错误！请先选择摘要格式再生成。",
        "no_key": "请添加您的 OpenAI API 密钥以继续。",
        "bad_key": "API 密钥无效，请重试。",
        "granted": "访问已授权！",
        "uploader": "上传文档 (.txt 或 .md)",
    },
    "Hindi": {
        "header": "**सेटिंग्स:**",
        "caption": "आउटपुट प्रारूप और AI मॉडल कॉन्फ़िगर करें",
        "lang_label": "**भाषा**",
        "lang_caption": "भाषा चुनें",
        "format_label": "**आउटपुट प्रारूप चुनें**",
        "format_placeholder": "एक प्रारूप चुनें",
        "formats": {
            "100-Word Summary": "100 शब्दों का सारांश",
            "2 Paragraph Summary": "2 जुड़े हुए अनुच्छेदों में सारांश",
            "5-Bullet Summary": "5 बुलेट बिंदुओं में सारांश",
        },
        "model_label": "**मॉडल चयन**",
        "using_basic": f"_आप {basic_model_name} का उपयोग कर रहे हैं_",
        "using_advanced": f"_अब {advanced_model_name} का उपयोग हो रहा है_",
        "advanced": "उन्नत मॉडल का उपयोग करें",
        "clear_cache": "कैश साफ़ करें",
        "generate": "सारांश बनाएँ",
        "dark_mode": "डार्क मोड",
        "description": "नीचे एक दस्तावेज़ अपलोड करें, फिर सारांश प्रारूप चुनें।",
        "error": "त्रुटि! बनाने से पहले कृपया सारांश प्रारूप चुनें।",
        "no_key": "जारी रखने के लिए कृपया अपनी OpenAI API कुंजी जोड़ें।",
        "bad_key": "अमान्य API कुंजी। कृपया पुनः प्रयास करें।",
        "granted": "पहुँच प्रदान की गई!",
        "uploader": "एक दस्तावेज़ अपलोड करें (.txt या .md)",
    },
    "Spanish": {
        "header": "**Ajustes:**",
        "caption": "Configura el formato de salida y el modelo de IA",
        "lang_label": "**Idioma**",
        "lang_caption": "Selecciona el idioma",
        "format_label": "**Especifica el formato de salida**",
        "format_placeholder": "Elige un formato",
        "formats": {
            "100-Word Summary": "Resumen de 100 palabras",
            "2 Paragraph Summary": "Resumen en 2 párrafos conectados",
            "5-Bullet Summary": "Resumen en 5 viñetas",
        },
        "model_label": "**Selección de modelo**",
        "using_basic": f"_Estás usando {basic_model_name}_",
        "using_advanced": f"_Ahora usando {advanced_model_name}_",
        "advanced": "Usar modelo avanzado",
        "clear_cache": "Borrar caché",
        "generate": "Generar resumen",
        "dark_mode": "Modo oscuro",
        "description": "Sube un documento abajo y luego selecciona el formato del resumen.",
        "error": "¡Error! Elige un formato de resumen antes de generar.",
        "no_key": "Añade tu clave de API de OpenAI para continuar.",
        "bad_key": "Clave de API no válida. Inténtalo de nuevo.",
        "granted": "¡Acceso concedido!",
        "uploader": "Sube un documento (.txt o .md)",
    },
    "French": {
        "header": "**Paramètres :**",
        "caption": "Configurez le format de sortie et le modèle d'IA",
        "lang_label": "**Langue**",
        "lang_caption": "Sélectionnez la langue",
        "format_label": "**Spécifiez le format de sortie**",
        "format_placeholder": "Choisissez un format",
        "formats": {
            "100-Word Summary": "Résumé de 100 mots",
            "2 Paragraph Summary": "Résumé en 2 paragraphes liés",
            "5-Bullet Summary": "Résumé en 5 puces",
        },
        "model_label": "**Sélection du modèle**",
        "using_basic": f"_Vous utilisez {basic_model_name}_",
        "using_advanced": f"_Vous utilisez maintenant {advanced_model_name}_",
        "advanced": "Utiliser le modèle avancé",
        "clear_cache": "Vider le cache",
        "generate": "Générer le résumé",
        "dark_mode": "Mode sombre",
        "description": "Téléversez un document ci-dessous, puis sélectionnez le format du résumé.",
        "error": "Erreur ! Veuillez choisir un format de résumé avant de générer.",
        "no_key": "Veuillez ajouter votre clé API OpenAI pour continuer.",
        "bad_key": "Clé API invalide. Veuillez réessayer.",
        "granted": "Accès autorisé !",
        "uploader": "Téléversez un document (.txt ou .md)",
    },
}

language = st.sidebar.selectbox("**Language**", ["English", "Mandarin Chinese", "Hindi", "Spanish", "French"],
    index=0,  #English selected by default
)
T = UI[language]

st.sidebar.header(T["header"])
st.sidebar.caption(T["caption"])
st.sidebar.caption(T["lang_caption"])

summary_type = st.sidebar.selectbox(T["format_label"], ["100-Word Summary", "2 Paragraph Summary", "5-Bullet Summary"],
    index=None,
    placeholder=T["format_placeholder"],
    format_func=lambda key: T["formats"][key],
)  #Stored

st.sidebar.markdown(T["model_label"])  #Replaces the old dropdown label
model_caption = st.sidebar.empty()  #Reserves the caption spot so it can sit above the checkbox
##***

#use_advanced = st.sidebar.checkbox("Use Advanced Model", value=False)  #Read before the dropdown so it can disable it
#base_model = st.sidebar.selectbox("**Select AI Model**", ["gpt-3.5-turbo", "gpt-5-nano", "gpt-4o-mini",],
#    index=None,
#    placeholder="Choose a model",
#    disabled=use_advanced,  #Greyed out when the advanced model is in use
#)  #Stored
#advanced_model = "gpt-4.1"
#selected_model = advanced_model if use_advanced else base_model  #Model selection sent to the API
##***
use_advanced = st.sidebar.checkbox(T["advanced"], value=False)  #Switches between the two fixed models
selected_model = advanced_model if use_advanced else basic_model  #Model selection sent to the API
model_caption.caption(T["using_advanced"] if use_advanced else T["using_basic"])  #Italic caption reflects the model in use
##***

#if st.sidebar.button("Clear Cache"):  #Clears the cached key validation
##***
if st.sidebar.button(T["clear_cache"]):  #Clears the cached key validation
##***
    st.cache_data.clear()

#generate = st.sidebar.button("Generate Summary", type="primary")  #Nothing is sent to the API until this is clicked
##***
generate = st.sidebar.button(T["generate"], type="primary")  #Nothing is sent to the API until this is clicked

dark_mode = st.sidebar.toggle(T["dark_mode"], value=False)  #Last item in the sidebar
if dark_mode:  #Applies a dark palette over the default theme
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        section[data-testid="stSidebar"] { background-color: #1a1c24; }
        .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3 { color: #fafafa; }
        </style>
        """,
        unsafe_allow_html=True,
    )
##***

#inputs_ready = bool(summary_type) and bool(selected_model)  #A format is always required, and a model must come from the dropdown or the checkbox
##***
inputs_ready = bool(summary_type)  #A model is always set now, so only the format has to be chosen
##***

# Show title and description.
st.title(":blue[Lab 2:] :grey[Deep] Scan Protocol")  #Updated title
#st.write(
#    "Upload a document below, then select summary format and model. "
#)
##***
st.write(T["description"])  #Description now follows the chosen language
##***

if generate and not inputs_ready:  #Error shown when either sidebar selection is missing
#    st.error("Error! Please choose a summary format and a model (or check Use Advanced Model) before generating.")
##***
    st.error(T["error"])  #Error now follows the chosen language
##***

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
#    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
##***
    st.info(T["no_key"], icon="🗝️")  #Message now follows the chosen language
##***
elif not is_valid_key(openai_api_key):  #Validate the API key when entered
#    st.error("Invalid API key. Please try again.")  #Error displayed
##***
    st.error(T["bad_key"])  #Error displayed, now follows the chosen language
##***
else:
#    st.success("Access granted!")  #Confirmation
##***
    st.success(T["granted"])  #Confirmation, now follows the chosen language
##***
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)
    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
#        "Upload a document (.txt or .md)", type=("txt", "md")
##***
        T["uploader"], type=("txt", "md")  #Label now follows the chosen language
##***
    )
    if uploaded_file and generate and inputs_ready:  #Runs once selections are made
        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
#                "content": f"Here's a document: {document} \n\n---\n\n {summary_instructions[summary_type]}", #Summary format is now the instruction
##***
                "content": f"Here's a document: {document} \n\n---\n\n {summary_instructions[summary_type]} Write the entire summary in {language}.",  #Summary format and output language are now the instruction
##***
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
            st.write_stream(stream)
