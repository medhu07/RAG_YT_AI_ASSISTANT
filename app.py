import streamlit as st
from rag import build_rag_chain, extract_video_id

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube RAG Chatbot")

st.markdown("""
Paste a YouTube video URL and ask questions about the video transcript.
Supports English and Hindi videos.
""")

st.sidebar.title("Controls")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("### Model")
st.sidebar.write("Llama 3.1 8B Instant (Groq)")

st.sidebar.markdown("### Embeddings")
st.sidebar.write("Multilingual MiniLM")

url = st.text_input("Paste YouTube URL")

if st.button("Process Video"):

    video_id = extract_video_id(url)

    if not video_id:
        st.error("Invalid YouTube URL")

    else:
        try:
            with st.spinner("Processing Video..."):

                chain = build_rag_chain(video_id)

                st.session_state.chain = chain
                st.session_state.messages = []

            st.success("Video Processed Successfully!")

        except Exception as e:
            st.error(f"Error: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" in st.session_state:

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask anything about the video..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = st.session_state.chain.invoke(
                    question
                )

                st.markdown(response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

st.divider()

st.caption(
    "Built with LangChain • FAISS • Groq • Streamlit"
)