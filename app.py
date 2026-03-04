import streamlit as st
import asyncio
import edge_tts
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from typing import List

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="E-Paper News Reader", layout="wide")
st.title("📰 Bangla E-Paper News Reader (Text → Audio)")

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(" ").strip()

def chunk_text(text: str, max_len: int = 2500) -> List[str]:
    """Split text into smaller chunks for TTS"""
    words = text.split()
    chunks, current = [], []
    current_len = 0
    
    for word in words:
        word_len = len(word) + 1
        if current_len + word_len < max_len:
            current.append(word)
            current_len += word_len
        else:
            if current:
                chunks.append(" ".join(current))
            current = [word]
            current_len = word_len
    
    if current:
        chunks.append(" ".join(current))
    return chunks

def get_page_id_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parse_qs(parsed.query).get("pgid", [None])[0]
    except:
        return None

def truncate_for_preview(text: str, max_chars: int = 1500) -> str:
    """Truncate text for quick audio preview"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + "..."

# ------------------------------------------------------------
# API FETCHING (CACHED)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_json(url: str):
    try:
        res = requests.get(url.strip(), timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return None

# ------------------------------------------------------------
# SCRAPING (CACHED)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_final_page_list(page_id: str):
    news_raw = get_json(f"https://epaper.prothomalo.com/Home/getStoriesOnPage?pageid={page_id}")
    if not news_raw:
        return []
    
    news_list = []
    for story in news_raw:
        region_id = story.get("OrgId")
        if not region_id:
            continue
        
        story_details = get_json(f"https://epaper.prothomalo.com/User/ShowArticleView?OrgId={region_id}")
        if not story_details:
            continue
        
        content_list = story_details.get("StoryContent", [])
        if not content_list:
            continue
        
        # Title
        title = content_list[0].get("Headlines")
        if isinstance(title, list):
            title = " ".join(title)
        title = title.strip()
        
        # Body
        body = content_list[0].get("Body", "")
        
        # Linked article
        linked_id = story_details.get("LinkedStoryId", 0)
        if linked_id:
            detail = get_json(f"https://epaper.prothomalo.com/Home/getstorydetail?Storyid={linked_id}")
            detail_list = detail.get("StoryContent", [])
            if detail_list:
                body += " " + detail_list[0].get("Body", "")
        
        final_content = clean_html(body)
        news_list.append({
            "title": title,
            "content": final_content
        })
    
    return news_list

# ------------------------------------------------------------
# TTS GENERATION (OPTIMIZED)
# ------------------------------------------------------------
async def generate_audio_chunk(text: str, voice: str = "bn-IN-BashkarNeural") -> bytes:
    """Generate audio for a single chunk"""
    communicate = edge_tts.Communicate(text, voice)
    output = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            output += chunk["data"]
    return output

async def generate_all_chunks_async(chunks: List[str], voice: str, progress_bar) -> bytes:
    """Process all chunks with progress tracking"""
    audio_bytes = b""
    total = len(chunks)
    
    for i, chunk in enumerate(chunks):
        audio_data = await generate_audio_chunk(chunk, voice)
        audio_bytes += audio_data
        progress_bar.progress((i + 1) / total)
    
    return audio_bytes

def generate_audio(text: str, use_preview: bool = False, voice: str = "bn-IN-BashkarNeural"):
    """Generate audio with optional preview mode"""
    if use_preview:
        text = truncate_for_preview(text, max_chars=1500)
    
    chunks = chunk_text(text, max_len=2500)
    
    if not chunks:
        return b""
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    try:
        audio_bytes = asyncio.run(generate_all_chunks_async(chunks, voice, progress_bar))
        return audio_bytes
    finally:
        progress_bar.empty()

# Cache audio by content hash
@st.cache_data(show_spinner=False)
def get_cached_audio(text: str, use_preview: bool = False, voice: str = "bn-IN-BashkarNeural"):
    """Cache audio generation by content hash"""
    return generate_audio(text, use_preview, voice)

# ------------------------------------------------------------
# SESSION STATE FOR AUDIO
# ------------------------------------------------------------
if 'audio_cache' not in st.session_state:
    st.session_state.audio_cache = {}

# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------
url = st.text_input("Paste the e-paper news page URL", "")

# Audio settings sidebar
with st.sidebar:
    st.header("⚙️ Audio Settings")
    voice_option = st.selectbox(
        "Voice",
        ["bn-IN-BashkarNeural", "bn-IN-TanishaaNeural"],
        index=0
    )
    preview_mode = st.checkbox("🚀 Quick Preview (faster)", value=False, 
                               help="Generate audio for first 1500 characters only")
    st.info("💡 Tip: Use Quick Preview for long articles to reduce wait time")

if url:
    page_id = get_page_id_from_url(url)
    if not page_id:
        st.error("❌ Invalid URL. Could not extract pgid.")
        st.stop()
    
    with st.spinner("🔍 Scraping news (cached)…"):
        news_list = get_final_page_list(page_id)
    
    if not news_list:
        st.error("❌ No news found on this page.")
        st.stop()
    
    st.success(f"✅ Found {len(news_list)} news articles.")
    st.divider()
    
    for idx, item in enumerate(news_list):
        with st.container():
            st.subheader(f"📰 {item['title']}")
            
            # Show content length warning
            content_len = len(item['content'])
            if content_len > 3000:
                st.warning(f"⚠️ Long article ({content_len} chars). Consider using Quick Preview mode!")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                play_key = f"play_{idx}"
                if st.button(f"▶ Play Audio #{idx+1}", key=play_key):
                    with st.spinner("🎧 Generating audio…"):
                        audio_data = get_cached_audio(item['content'], preview_mode, voice_option)
                    
                    if audio_data:
                        # Store in session state with unique key per article
                        st.session_state.audio_cache[idx] = audio_data
                        st.success("✅ Audio ready!")
            
            with col2:
                st.metric("Length", f"{content_len} chars")
            
            # Display audio player (only if audio exists for this article)
            if idx in st.session_state.audio_cache:
                st.audio(st.session_state.audio_cache[idx], format="audio/mp3")
            
            with st.expander("📄 Read News Text"):
                st.write(item['content'])
            
            st.divider()

# Footer
st.markdown("---")
st.caption("💡 **Performance Tips:** Use Quick Preview for articles >3000 characters | Audio is cached for repeated plays")