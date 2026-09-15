#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Lab04 - Part A: Create and test a ChromaDB database

import streamlit as st
from openai import OpenAI
import sys
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

#text-embedding-3-small errors above 8,191 tokens. A full syllabus is embedded as
#ONE document in Part A (no chunking yet), so long text is capped here at roughly
#7,000 tokens' worth of characters to keep the API call from failing.
MAX_EMBED_CHARS = 28000


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

    # Create an embedding
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )

    # Get the embedding
    embedding = response.data[0].embedding

    # Add embedding and document to ChromaDB
    collection.add(
        documents=[text],
        ids=file_name, #Part A, Step 2: the filename is the key
        embeddings=[embedding],
        metadatas=[{'filename': file_name}] #Part A, Step 2: "use metadata as needed"
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
            text = text[:MAX_EMBED_CHARS] #Truncate rather than let the embeddings call error out

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
#### PART A TEST BLOCK - BEGIN ####
#### Comment out everything down to PART A TEST BLOCK - END when starting Part B
topic = st.sidebar.text_input('Topic', placeholder='Type your topic (e.g., GenAI)...')

if topic:
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=topic,
        model=EMBEDDING_MODEL)

    # Get the embedding
    query_embedding = response.data[0].embedding

    # Get the text related to this question (this prompt)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # The number of closest documents to return
    )

    # Display the results
    st.subheader(f'Results for: {topic}')

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i] #Retrieved text; Part A only displays the id, Part B will use this
        doc_id = results['ids'][0][i]

        st.write(f'**{i+1}. {doc_id}**')

else:
    st.info('Enter a topic in the sidebar to search the collection')
#### PART A TEST BLOCK - END ####


#### PART B: chatbot goes here (not implemented yet) ####
