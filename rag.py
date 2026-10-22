
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

    try:
        response = requests.post(
            "https://www.youtubetranscript.dev/api/v2/transcribe",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            json={
                "video": video_id,
                "mode": "captions_only",
                "language": "en"
            },
            timeout=60
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to transcript API: {exc}"
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Transcript API error {response.status_code}: "
            f"{response.text[:300]}"
        )

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Transcript API returned invalid JSON."
        ) from exc

    transcript_data = result.get("data", {}).get("transcript", {})
    transcript = transcript_data.get("text", "")

    if result.get("status") != "completed" or not transcript.strip():
        raise RuntimeError(
            "Transcript is unavailable. The video may lack accessible "
            "captions, or transcription may still be processing."
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
