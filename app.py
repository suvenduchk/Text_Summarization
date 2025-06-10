import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_core.documents import Document
from pytube import YouTube
import sys
import traceback
import time
import re
import requests

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    model_choice = st.selectbox(
        "Select Model", 
        ["llama3-8b-8192", "llama3-70b-8192"],
        index=0,
        help="Faster models are listed first"
    )
    
generic_url = st.text_input("URL", label_visibility="collapsed")

prompt_template = """
Provide a concise summary of the following content in 4000 words:
Content:{text}

Focus on the main points and key information only.

If this is a YouTube video with minimal information, please respond with:
"This appears to be a YouTube video. To summarize this video, I would need access to its transcript or detailed description. Unfortunately, this video doesn't have accessible captions or detailed metadata that I can use for summarization."
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    youtube_regex = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(youtube_regex, url)
    return match.group(1) if match else None

def safe_load_docs(url):
    """
    Attempts to load documents from the given URL using the appropriate loader.
    Returns the loaded docs or raises a RuntimeError with a user-friendly message.
    """
    try:
        if "youtube.com" in url or "youtu.be" in url:
            video_id = extract_video_id(url)
            
            # Display video thumbnail if we have the ID
            if video_id:
                st.image(f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg", 
                        width=400, 
                        caption="Video Thumbnail")
            
            # Try multiple methods to get video content
            try:
                # First try: Use YoutubeLoader
                try:
                    loader = YoutubeLoader.from_youtube_url(
                        url, 
                        add_video_info=True,
                        language='en'
                    )
                    docs = loader.load()
                    
                    # Show transcript length for context
                    if docs:
                        word_count = len(docs[0].page_content.split())
                        st.info(f"📺 **Video Title:** {docs[0].metadata.get('title', 'Unknown')}")
                        st.info(f"👤 **Channel:** {docs[0].metadata.get('author', 'Unknown')}")
                        st.info(f"📝 **Transcript Length:** {word_count} words")
                        return docs
                except Exception as caption_exc:
                    # Second try: Use pytube directly
                    try:
                        yt = YouTube(url)
                        st.info(f"📺 **Video Title:** {yt.title}")
                        st.info(f"👤 **Channel:** {yt.author}")
                        
                        # If captions aren't available, use video description and metadata
                        st.warning("⚠️ **This video doesn't have accessible captions/subtitles.** Using video description for summarization instead.")
                        
                        # Create a document with video metadata and description
                        description = yt.description if yt.description else "No description available."
                        content = f"Video Title: {yt.title}\n\nChannel: {yt.author}\n\nDescription: {description}"
                        
                        doc = Document(
                            page_content=content,
                            metadata={"title": yt.title, "author": yt.author, "source": url}
                        )
                        
                        return [doc]
                    except Exception as pytube_exc:
                        # Third try: Use YouTube API to get video info
                        import requests
                        try:
                            st.warning("⚠️ **Unable to access video details through standard methods.** Attempting to fetch video information from YouTube API.")
                            # Try to get video info from YouTube oEmbed API
                            oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
                            response = requests.get(oembed_url)
                            if response.status_code == 200:
                                video_data = response.json()
                                title = video_data.get('title', 'Unknown Title')
                                author = video_data.get('author_name', 'Unknown Author')
                                st.info(f"📺 **Video Title:** {title}")
                                st.info(f"👤 **Channel:** {author}")
                                
                                # Create document with the information we have
                                content = f"Video Title: {title}\nChannel: {author}\nVideo ID: {video_id}\nURL: {url}"
                                doc = Document(
                                    page_content=content,
                                    metadata={"title": title, "author": author, "source": url, "video_id": video_id}
                                )
                                return [doc]
                            else:
                                # If all else fails, create minimal document with video ID
                                st.warning("⚠️ **Unable to access video details.** Creating minimal summary from video ID.")
                                content = f"YouTube Video ID: {video_id}\nURL: {url}\nThis is a YouTube video that requires captions or metadata for proper summarization."
                                doc = Document(
                                    page_content=content,
                                    metadata={"source": url, "video_id": video_id}
                                )
                                return [doc]
                        except Exception as api_exc:
                            # If all else fails, create minimal document with video ID
                            st.warning("⚠️ **Unable to access video details.** Creating minimal summary from video ID.")
                            content = f"YouTube Video ID: {video_id}\nURL: {url}\nThis is a YouTube video that requires captions or metadata for proper summarization."
                            doc = Document(
                                page_content=content,
                                metadata={"source": url, "video_id": video_id}
                            )
                            return [doc]
                        else:
                            raise RuntimeError("Failed to extract any information from this YouTube video.")
            except Exception as yt_exc:
                raise RuntimeError(f"Failed to load YouTube video: {yt_exc}")
        else:
            try:
                # Using WebBaseLoader instead of UnstructuredURLLoader for better performance
                loader = WebBaseLoader(
                    web_paths=[url],
                    header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"}
                )
                docs = loader.load()
                
                # Show webpage info
                if docs:
                    word_count = len(docs[0].page_content.split())
                    st.info(f"📄 **Web Page Content:** {word_count} words")
                
                return docs
            except Exception as url_exc:
                raise RuntimeError(f"Failed to initialize URL loader: {url_exc}")
        
        if not docs:
            raise RuntimeError("No content could be loaded from the provided URL.")
        return docs
    except Exception as e:
        raise RuntimeError(str(e))

if st.button("Summarize the Content from YouTube Videos or Any Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or website URL.")
    else:
        try:
            start_time = time.time()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Initialize LLM (10%)
            status_text.text("Initializing AI model...")
            try:
                llm = ChatGroq(model=model_choice, groq_api_key=groq_api_key)
                progress_bar.progress(10)
            except Exception as llm_exc:
                st.error(f"Failed to initialize LLM: {llm_exc}")
                st.stop()
            
            # Step 2: Load content (40%)
            status_text.text("Loading content from URL...")
            try:
                docs = safe_load_docs(generic_url)
                progress_bar.progress(40)
            except RuntimeError as doc_exc:
                st.error(str(doc_exc))
                st.stop()
            
            # Step 3: Prepare summarization (60%)
            status_text.text("Preparing summarization...")
            try:
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                progress_bar.progress(60)
            except Exception as chain_exc:
                st.error(f"Summarization preparation failed: {chain_exc}")
                st.stop()
            
            # Step 4: Generate summary (100%)
            status_text.text("Generating summary...")
            try:
                output_summary = chain.run(docs)
                progress_bar.progress(100)
                
                # Display results
                elapsed_time = time.time() - start_time
                st.success(f"Summary completed in {elapsed_time:.2f} seconds")
                
                # Format the summary nicely
                st.markdown("### Summary")
                st.markdown(output_summary)
            except Exception as chain_exc:
                st.error(f"Summarization failed: {chain_exc}")
                st.text(traceback.format_exc())
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.text(traceback.format_exc())
