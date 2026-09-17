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
# The try/except is a small addition to the slide code: pysqlite3-binary has no
# Windows wheel, so an unguarded import crashes the app when run locally.
# On Streamlit Cloud the swap happens exactly as the slide shows.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass #Local sqlite3 is usually new enough for Chroma

COLLECTION_NAME = 'Lab4Collection' #Part A, Step 2: required collection name
EMBEDDING_MODEL = 'text-embedding-3-small' #Part A, Step 2: OpenAI embeddings model
CHROMA_PATH = './ChromaDB_for_Lab' #Where PersistentClient writes the DB files
DATA_SUBFOLDER = Path('data') / 'Lab04' #Repo folder holding the 7 syllabus PDFs

#Set to True for ONE run to wipe and re-embed the collection, then set back to
#False. Needed after changing what text gets embedded, because an already
#populated collection is otherwise reused untouched.
REBUILD_COLLECTION = False

#text-embedding-3-small errors above 8,191 tokens. A full syllabus is embedded as
#ONE document in Part A (no chunking yet), so long text is capped here at roughly
#7,000 tokens' worth of characters to keep the API call from failing.
MAX_EMBED_CHARS = 28000

CHAT_MODEL = 'gpt-5.4-mini' #Part B, Step 5: same model as Lab 3
N_RESULTS = 3 #Part B: how many syllabi are retrieved per question
HISTORY_MESSAGES = 8 #How many past messages travel with each request

#Part B, Step 5: prompt engineering. The retrieved syllabus text is appended to
#this before it is sent as the 'system' message, and the rules below are what make
#the bot state when it is answering from the RAG context.
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

STYLE
- Be concise and direct. Short paragraphs, or a short list when comparing courses.
- If the question is about several courses, answer course by course."""


def parse_file_name(file_name): #Pulls the course code and title out of the filename
    stem = Path(file_name).stem

    match = re.match(r'\s*IST[\s\-]*(\d{3})', stem, re.IGNORECASE)
    course_code = match.group(1) if match else ''

    if ' - ' in stem:
        course_title = stem.split(' - ', 1)[1].strip() #Text after the dash, e.g. "Data in Society"
    else:
        course_title = stem

    return course_code, course_title


#### EXTRACT TEXT FROM PDF ####
# This function extracts text from each syllabus
# to pass to add_to_collection
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or '') #extract_text() returns None on a page with no text layer

    return '\n'.join(pages).strip()


#### USING CHROMA DB WITH OPENAI EMBEDDINGS ####
# A function that will add documents to collection
# collection = ChromaDB collection, already established
# text = extracted text from PDF files
# Embeddings inserted into the collection from OpenAI
def add_to_collection(collection, text, file_name):
    course_code, course_title = parse_file_name(file_name)

    #A course code appears under once per 1,000 characters inside its own syllabus,
    #so it barely registers in a whole-document embedding. This header puts the
    #course identity at the front of the embedded text, and the LLM sees it too.
    document = (
        f'COURSE: IST {course_code}\n'
        f'COURSE TITLE: {course_title}\n'
        f'SOURCE FILE: {file_name}\n\n'
        + text
    )

    # Create an embedding
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=document,
        model=EMBEDDING_MODEL
    )

    # Get the embedding
    embedding = response.data[0].embedding

    # Add embedding and document to ChromaDB
    collection.add(
        documents=[document],
        ids=file_name, #Part A, Step 2: the filename is the key
        embeddings=[embedding],
        metadatas=[{ #Part A, Step 2: "use metadata as needed"
            'filename': file_name,
            'course_code': course_code, #Used by the course-code filter in get_info_from_vectorDB
            'course_title': course_title
        }]
    )


#### POPULATE COLLECTION WITH PDFs ####
# This function uses extract_text_from_pdf
# and add_to_collection to put syllabi in ChromaDB collection
def load_pdfs_to_collection(folder_path, collection):
    folder = Path(folder_path)
    pdf_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() == '.pdf')

    loaded = []

    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)

        if not text:
            st.warning(f'No extractable text in {pdf_path.name} - skipped') #A scanned PDF would need OCR
            continue

        if len(text) > MAX_EMBED_CHARS:
            text = text[:MAX_EMBED_CHARS] #Truncate the body; the header is added after this, so it always survives

        add_to_collection(collection, text, pdf_path.name) #Key/id is the filename
        loaded.append(pdf_path.name)

    return loaded


def find_data_folder(): #Walks up from this file so the app works whether Lab4.py sits at the repo root or in a pages/ folder
    start = Path(__file__).resolve().parent

    for base in [start, *start.parents]:
        candidate = base / DATA_SUBFOLDER

        if candidate.is_dir():
            return candidate

    return None


def create_lab4_vectordb(): #Part A, Step 2: the one function that builds the whole vector DB
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    if REBUILD_COLLECTION: #Forces a fresh embed; runs once per session because of the Lab4_VectorDB guard
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass #Nothing to delete on a first run

        st.session_state.pop('Lab4_CourseCodes', None) #Cached code list is stale after a rebuild

    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    st.session_state.Lab4_ChromaClient = chroma_client #Held so the client is not garbage collected

    # Check if collection is empty and load PDFs
    if collection.count() == 0: #Embeds once; a populated DB on disk is reused as-is
        folder = find_data_folder()

        if folder is None:
            st.error(f'Could not find the folder {DATA_SUBFOLDER} containing the syllabus PDFs.')
            st.stop()

        with st.spinner('Embedding the syllabus PDFs (one time only)...'):
            loaded = load_pdfs_to_collection(folder, collection)

        if loaded:
            st.success(f'Added {len(loaded)} PDFs to {COLLECTION_NAME}: ' + ', '.join(loaded))
        else:
            st.error(f'No PDFs were loaded from {folder}.')

    return collection


def course_codes_in_collection(collection): #The codes actually stored, read once per session
    if 'Lab4_CourseCodes' not in st.session_state:
        stored = collection.get(include=['metadatas'])['metadatas']

        codes = set()

        for meta in stored:
            if meta and meta.get('course_code'):
                codes.add(meta['course_code'])

        st.session_state.Lab4_CourseCodes = sorted(codes)

    return st.session_state.Lab4_CourseCodes


def detect_course_codes(query, known_codes): #Finds 3-digit codes in the question that exist in the collection
    found = re.findall(r'IST[\s\-]*(\d{3})', query, re.IGNORECASE) #"IST 387", "IST-387", "IST387"
    found += re.findall(r'\b\d{3}\b', query) #A bare code, e.g. a follow-up like "and 488?"

    return sorted(set(code for code in found if code in known_codes))


#### RETRIEVE RELEVANT SYLLABI ####
# Part B, Step 5: embeds the student's question, finds the closest syllabi, and
# returns their text formatted for the prompt, the filenames used, and any course
# codes the question named.
def get_info_from_vectorDB(collection, query, n_results=N_RESULTS):
    client = st.session_state.openai_client

    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL #Same model used to embed the documents, so the vectors are comparable
    )

    query_embedding = response.data[0].embedding

    #A named course code is an exact request, not a similarity guess, so it is
    #answered with a metadata filter instead of being left to the vector search.
    codes = detect_course_codes(query, course_codes_in_collection(collection))

    where = None

    if len(codes) == 1:
        where = {'course_code': codes[0]}
    elif len(codes) > 1:
        where = {'course_code': {'$in': codes}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where #None means a normal similarity search across all 7 syllabi
    )

    documents = results['documents'][0]
    ids = results['ids'][0]

    blocks = []

    for doc_id, doc in zip(ids, documents):
        blocks.append(f'--- SOURCE FILE: {doc_id} ---\n{doc}') #Headings let the LLM name its source

    return '\n\n'.join(blocks), list(ids), codes


#### MAIN APP ####
st.title('Lab 4: Chatbot using RAG')

# Create OpenAI client
if 'openai_client' not in st.session_state: #Built before the vector DB, because add_to_collection uses it
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

# Part A, Step 2: only build the vector DB if it is not already in session_state
if 'Lab4_VectorDB' not in st.session_state:
    st.session_state.Lab4_VectorDB = create_lab4_vectordb()

collection = st.session_state.Lab4_VectorDB

st.sidebar.caption(f'Documents in {COLLECTION_NAME}: {collection.count()}') #Quick check that all 7 loaded


#### QUERYING A COLLECTION -- ONLY USED FOR TESTING ####
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
if 'messages' not in st.session_state: #Guard stops the history being wiped on every rerun
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': 'Ask me about the IST course syllabi.'}
    ]

for msg in st.session_state.messages: #Redraws the full visible history
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input('Ask about the IST courses...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    client = st.session_state.openai_client

    with st.spinner('Searching the syllabi...'):
        extra_info, sources, codes = get_info_from_vectorDB(collection, prompt) #Part B, Step 5: RAG lookup

    #Rebuilt every turn and never appended to st.session_state.messages, so the
    #syllabus text is sent once per request instead of accumulating in the history
    system_msg = {
        'role': 'system',
        'content': RAG_SYSTEM_PROMPT + '\n\nCOURSE CONTEXT\n' + extra_info
    }

    recent_history = st.session_state.messages[-HISTORY_MESSAGES:] #Simple bound; the context is the expensive part
    messages_to_send = [system_msg] + recent_history #Prepended AFTER slicing

    st.session_state.last_sources = sources
    st.session_state.last_course_filter = codes

    try:
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages_to_send,
            stream=True,
        )

        with st.chat_message('assistant'):
            response = st.write_stream(stream) #Streams the reply and returns the finished text

    except Exception as e:
        st.error(f'This request has failed: {e}')
        st.stop()

    st.session_state.messages.append({'role': 'assistant', 'content': response})

if 'last_sources' in st.session_state: #Shows which files the RAG actually pulled
    note = ''

    if st.session_state.get('last_course_filter'):
        note = ' (filtered to IST ' + ', '.join(st.session_state.last_course_filter) + ')'

    st.caption('Syllabi retrieved for the last question: ' + ', '.join(st.session_state.last_sources) + note)