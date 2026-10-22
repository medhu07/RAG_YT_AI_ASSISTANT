
import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)


def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "/shorts/" in url:
        return url.split("/shorts/")[1].split("?")[0]
    return None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def fetch_transcript(video_id):
    api_key = os.getenv("YOUTUBE_TRANSCRIPT_API_KEY")

    if not api_key:
        raise ValueError(
            "YOUTUBE_TRANSCRIPT_API_KEY is missing in Render Environment."
        )

    response = requests.get(
        "https://getyoutubetranscript.com/api/v1/transcript",
        params={"v": video_id, "language": "en"},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=45
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Transcript API error {response.status_code}: "
            f"{response.text[:300]}"
        )

    result = response.json()
    data = result.get("data", {})
    transcript = data.get("transcript")

    if not result.get("success") or not transcript:
        raise RuntimeError(
            "No transcript was returned. Try another video with captions."
        )

    if isinstance(transcript, list):
        transcript = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in transcript
        )

    return transcript


@st.cache_resource
def build_rag_chain(video_id):
    transcript = fetch_transcript(video_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(transcript)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the answer is not in the transcript, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    return parallel_chain | prompt | llm | StrOutputParser()
