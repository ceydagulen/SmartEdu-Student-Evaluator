from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return embeddings


def create_vectorstore(chunks: list, persist_directory="data/vectorstore") -> Chroma:
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Vektör veritabanı oluşturuldu. {len(chunks)} parça kaydedildi.")
    return vectorstore


def load_vectorstore(persist_directory="data/vectorstore") -> Chroma:
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print("Vektör veritabanı yüklendi.")
    return vectorstore


def ask_question(question: str, vectorstore: Chroma) -> str:
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Aşağıdaki ders transkriptine dayanarak soruyu cevapla.
    
Transkript:
{context}

Soru: {question}

Cevap:"""
    
    response = llm.invoke(prompt)
    return response.content