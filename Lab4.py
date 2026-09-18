#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab04

import streamlit as st
from openai import OpenAI
import sys
import re
import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

# A fix for working with ChromaDB on Streamlit Community Cloud
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

COLLECTION_NAME = 'Lab4Collection'
EMBEDDING_MODEL = 'text-embedding-3-small'
CHROMA_PATH = './ChromaDB_for_Lab'
DATA_SUBFOLDER = Path('data') / 'Lab04'

REBUILD_COLLECTION = False #True for ONE run to wipe and re-embed, then back to False

MAX_EMBED_CHARS = 28000 #text-embedding-3-small errors above 8,191 tokens

CHAT_MODEL = 'gpt-5.4-mini'
N_RESULTS = 5
HISTORY_MESSAGES = 8

#Part B, Retrieved syllabus text is appended to this.
RAG_SYSTEM_PROMPT = """You are a course information assistant for Syracuse University
iSchool students. You answer questions about a small set of IST course syllabi.

HOW TO USE THE COURSE CONTEXT
- Syllabus text is provided below under SOURCE FILE headings. It was retrieved from
a vector database by matching the student's question.
- When your answer draws on that text, say so plainly and name the syllabus you used,
for example: "Based on the IST 387 syllabus, ...".
- When the context does not contain the answer, say clearly that the retrieved
syllabi do not cover it. You may then answer from general knowledge, but label that
part as coming from outside the course documents.
- Never invent a course policy, date, grade weight, instructor name, textbook, or
assignment. If a detail is not in the context, say it is not in the syllabus.
- Only give exact figures such as percentages, dates, or credit hours when they
appear in the context.

ATTRIBUTION
- Every fact you state must be attributed to the course named in the COURSE: header
of the block it came from. Read that header before naming a course.
- Do not attribute a detail to any course whose block is not present in the context,
even if that course was mentioned earlier in the conversation.

STYLE
- Be concise and direct. Short paragraphs, or a short list when comparing courses.
- If the question is about several courses, answer course by course."""


def parse_file_name(file_name):
    stem = Path(file_name).stem

    match = re.match(r'\s*IST[\s\-]*(\d{3})', stem, re.IGNORECASE)
    course_code = match.group(1) if match else ''

    if ' - ' in stem:
        course_title = stem.split(' - ', 1)[1].strip()
    else:
        course_title = stem

    return course_code, course_title


#### EXTRACT TEXT FROM PDF ####
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or '') #None on a page with no text layer

    return '\n'.join(pages).strip()


#### USING CHROMA DB WITH OPENAI EMBEDDINGS ####
def add_to_collection(collection, text, file_name):
    course_code, course_title = parse_file_name(file_name)

    #Header fix
    document = (
        f'COURSE: IST {course_code}\n'
        f'COURSE TITLE: {course_title}\n'
        f'SOURCE FILE: {file_name}\n\n'
        + text
    )

    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=document,
        model=EMBEDDING_MODEL
    )

    embedding = response.data[0].embedding

    collection.add(
        documents=[document],
        ids=file_name, #Part A, Step 2: the filename is the key
        embeddings=[embedding],
        metadatas=[{
            'filename': file_name,
            'course_code': course_code,
            'course_title': course_title
        }]
    )


#### POPULATE COLLECTION WITH PDFs ####
def load_pdfs_to_collection(folder_path, collection):
    folder = Path(folder_path)
    pdf_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() == '.pdf')

    existing = set(collection.get()['ids']) #Interrupted load resumes instead of duplicating

    loaded = []

    for pdf_path in pdf_paths:
        if pdf_path.name in existing:
            continue

        text = extract_text_from_pdf(pdf_path)

        if not text:
            st.warning(f'No extractable text in {pdf_path.name} - skipped')
            continue

        if len(text) > MAX_EMBED_CHARS:
            text = text[:MAX_EMBED_CHARS]

        add_to_collection(collection, text, pdf_path.name)
        loaded.append(pdf_path.name)

    return loaded


def find_data_folder(): #Walks up so this works from the repo root or a pages/ folder
    start = Path(__file__).resolve().parent

    for base in [start, *start.parents]:
        candidate = base / DATA_SUBFOLDER

        if candidate.is_dir():
            return candidate

    return None


def create_lab4_vectordb():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    if REBUILD_COLLECTION:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        st.session_state.pop('Lab4_CourseCodes', None)

    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    st.session_state.Lab4_ChromaClient = chroma_client #Held so the client is not garbage collected

    folder = find_data_folder()

    if folder is None:
        if collection.count() == 0:
            st.error(f'Could not find the folder {DATA_SUBFOLDER} containing the syllabus PDFs.')
            st.stop()

        return collection

    pdf_total = len([p for p in folder.iterdir() if p.suffix.lower() == '.pdf'])

    if collection.count() < pdf_total: #Covers an empty collection AND a load interrupted part way
        with st.spinner('Embedding the syllabus PDFs (one time only)...'):
            loaded = load_pdfs_to_collection(folder, collection)

        if loaded:
            st.success(f'Added {len(loaded)} PDFs to {COLLECTION_NAME}: ' + ', '.join(loaded))
        elif collection.count() == 0:
            st.error(f'No PDFs were loaded from {folder}.')

    return collection


def course_codes_in_collection(collection):
    if 'Lab4_CourseCodes' not in st.session_state:
        stored = collection.get(include=['metadatas'])['metadatas']

        codes = set()

        for meta in stored:
            if meta and meta.get('course_code'):
                codes.add(meta['course_code'])

        st.session_state.Lab4_CourseCodes = sorted(codes)

    return st.session_state.Lab4_CourseCodes


def detect_course_codes(query, known_codes):
    found = re.findall(r'IST[\s\-]*(\d{3})', query, re.IGNORECASE) #"IST 387", "IST-387", "IST387"
    found += re.findall(r'\b\d{3}\b', query) #A bare code, e.g. "and 488?"

    return sorted(set(code for code in found if code in known_codes))


#### RETRIEVE RELEVANT SYLLABI ####
def get_info_from_vectorDB(collection, query, n_results=N_RESULTS):
    client = st.session_state.openai_client

    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )

    query_embedding = response.data[0].embedding


    codes = detect_course_codes(query, course_codes_in_collection(collection)) #Metadata filter

    where = None

    if len(codes) == 1:
        where = {'course_code': codes[0]}
    elif len(codes) > 1:
        where = {'course_code': {'$in': codes}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where
    )

    documents = results['documents'][0]
    ids = results['ids'][0]

    blocks = []

    for doc_id, doc in zip(ids, documents):
        blocks.append(f'--- SOURCE FILE: {doc_id} ---\n{doc}') #Headings let the LLM name its source

    return '\n\n'.join(blocks), list(ids), codes


#### MAIN APP ####
st.title(":blue[Lab 4:] :grey[Deep] Chatbot | RAG")

if 'openai_client' not in st.session_state: #Built first, because add_to_collection uses it
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

# Part A, Step 2: only build the vector DB if it is not already in session_state
if 'Lab4_VectorDB' not in st.session_state:
    st.session_state.Lab4_VectorDB = create_lab4_vectordb()

collection = st.session_state.Lab4_VectorDB

st.sidebar.caption(f'Documents in {COLLECTION_NAME}: {collection.count()}')


#### QUERYING A COLLECTION - ONLY USED FOR TESTING ####
#### PART A TEST BLOCK - COMMENTED OUT FOR PART B, Step 4 ####
# topic = st.sidebar.text_input('Topic', placeholder='Type your topic (e.g., GenAI)...')
#
# if topic:
#     client = st.session_state.openai_client
#     response = client.embeddings.create(
#         input=topic,
#         model=EMBEDDING_MODEL)
#
#     # Get the embedding
#     query_embedding = response.data[0].embedding
#
#     # Get the text related to this question (this prompt)
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=3  # The number of closest documents to return
#     )
#
#     # Display the results
#     st.subheader(f'Results for: {topic}')
#
#     for i in range(len(results['documents'][0])):
#         doc = results['documents'][0][i]
#         doc_id = results['ids'][0][i]
#
#         st.write(f'**{i+1}. {doc_id}**')
#
# else:
#     st.info('Enter a topic in the sidebar to search the collection')
#### PART A TEST BLOCK - END ####


#### PART B: COURSE INFORMATION CHATBOT ####
if 'messages' not in st.session_state:
    codes_available = course_codes_in_collection(collection)

    if codes_available:
        greeting = 'Ask me questions about IST ' + ', '.join(codes_available) + ' syllabi.'
    else:
        greeting = 'Ask me questions about the IST course syllabi.'

    st.session_state['messages'] = [
        {'role': 'assistant', 'content': greeting}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input('Ask about the IST courses...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    client = st.session_state.openai_client

    with st.spinner('Searching the syllabi...'):
        extra_info, sources, codes = get_info_from_vectorDB(collection, prompt)

    #Rebuilt every turn and never appended to st.session_state.messages
    system_msg = {
        'role': 'system',
        'content': RAG_SYSTEM_PROMPT + '\n\nCOURSE CONTEXT\n' + extra_info
    }

    recent_history = st.session_state.messages[-HISTORY_MESSAGES:]
    messages_to_send = [system_msg] + recent_history

    st.session_state.last_sources = sources
    st.session_state.last_course_filter = codes

    try:
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages_to_send,
            stream=True,
        )

        with st.chat_message('assistant'):
            response = st.write_stream(stream)

    except Exception as e:
        st.error(f'This request has failed: {e}')
        st.stop()

    st.session_state.messages.append({'role': 'assistant', 'content': response})

if 'last_sources' in st.session_state:
    note = ''

    if st.session_state.get('last_course_filter'):
        note = ' (filtered to IST ' + ', '.join(st.session_state.last_course_filter) + ')'

    st.caption('Syllabi retrieved for the last question: ' + ', '.join(st.session_state.last_sources) + note)
