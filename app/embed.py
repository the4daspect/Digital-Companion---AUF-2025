import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# chunk the texts
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True,
)

all_chunks = []

for file in glob.glob("streamlit/rag_docs/career_*.pdf"):
    loader = PyPDFLoader(file)

    if "career_1.pdf" in file:
        docs = loader.load()[12:440]
    else:
        docs = loader.load()

    chunks = text_splitter.split_documents(docs)
    all_chunks.extend(chunks)

    print(f"{file}: split into {len(chunks)} chunks.")

print(f"Total chunks across all docs: {len(all_chunks)}")

# store in vectorstore
vectorstore = FAISS.from_documents(
    all_chunks,
    embedding=embeddings,
)

vectorstore.save_local("streamlit/vector_stores/UvA_AUG_YAG_chatbot")
