from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st

name = "the_Garden_of_sinners"

CHROMA_DB_PATH = f"vector_database/{name}"


def embed_data():
    loader = CSVLoader(file_path=f"{name}.csv", encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embed = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vector_db = Chroma.from_documents(texts, embed, persist_directory=CHROMA_DB_PATH)
    vector_db.persist()


embed_data()
