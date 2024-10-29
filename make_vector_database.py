from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st

st.title("Make Vector Database")

name = "the_Garden_of_sinners"

CHROMA_DB_PATH = f"vector_database/{name}"

loader = TextLoader(f"{name}.txt", encoding="utf-8")
data = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n", chunk_size=900, chunk_overlap=0, length_function=len
)
documents = text_splitter.create_documents([doc.page_content for doc in data])

with open("text_chunks.txt", "w", encoding="utf-8") as file:
    for text in documents:
        file.write(text.page_content)
        file.write("\n--------------------------------------\n")

key = st.secrets["OPENAI_API_KEY"]
embed = OpenAIEmbeddings(
    openai_api_key=key,
    model="text-embedding-3-large",
)
db = Chroma.from_documents(
    documents=documents,
    embedding=embed,
    persist_directory=CHROMA_DB_PATH,
)
if db:
    db.persist()
    print(f"db:{db._collection.count()}")
    db = None
else:
    st.write("Chroma DB has not been initialized.")
