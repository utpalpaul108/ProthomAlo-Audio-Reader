import streamlit as st
import asyncio
import edge_tts
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import re
import demjson3
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="প্রথম আলো · Audio Reader",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# LOAD EXTERNAL CSS
# ------------------------------------------------------------
def load_css(filepath) -> None:
    css = Path(filepath).read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css(Path(__file__).parent / "style.css")

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def clean_html(html: str) -> str:
    """Extract text from HTML, preserving paragraph breaks as double newlines."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Block-level elements → paragraph separator
    for tag in soup.find_all(["p", "div", "br", "h1", "h2", "h3", "h4", "li"]):
        tag.insert_before("\n\n")
    return soup.get_text(" ").strip()

# Noise patterns that appear in scraped e-paper text and sound bad when read aloud
_NOISE_PATTERNS = [
    r'\(?\s*ফাইল\s*ছবি\s*\)?',          # "(ফাইল ছবি)" – file photo caption
    r'\(?\s*ছবি\s*[:：]\s*[^)।\n]*\)?',  # "ছবি: photographer name"
    r'\(?\s*প্রতিবেদক\s*\)?',            # reporter credit
    r'\(?\s*সংবাদদাতা\s*\)?',            # correspondent credit
    r'\(?\s*বিশেষ\s*প্রতিনিধি\s*\)?',   # special correspondent
    r'\(?\s*স্টাফ\s*রিপোর্টার\s*\)?',   # staff reporter
    r'https?://\S+',                       # raw URLs
    r'\*{2,}',                             # repeated asterisks
]

def preprocess_for_tts(title: str, content: str) -> str:
    """
    Combine title + content and clean for natural, realistic TTS output.
    - Removes noise (captions, bylines, URLs)
    - Converts paragraph breaks into sentence-ending pauses
    - Normalises punctuation spacing
    """
    # Remove noise from content before combining
    clean = content.strip()
    for pattern in _NOISE_PATTERNS:
        clean = re.sub(pattern, '', clean, flags=re.IGNORECASE)

    # Paragraph breaks → sentence-ending pause (। causes natural TTS pause)
    clean = re.sub(r'\n\n+', '। ', clean)
    clean = re.sub(r'\n', ' ', clean)

    full_text = f"{title.strip()}। {clean}"

    # Spacing after sentence endings
    full_text = re.sub(r'([।!?])([^\s])', r'\1 \2', full_text)
    # Collapse repeated punctuation
    full_text = re.sub(r'([।!?,;])\1+', r'\1', full_text)
    # Collapse whitespace
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    return full_text

def chunk_text(text: str, max_len: int = 2500) -> List[str]:
    """Split text into chunks at sentence boundaries for natural TTS flow."""
    sentences = re.split(r'(?<=[।!?])\s+', text)
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if current_len + sentence_len <= max_len:
            current.append(sentence)
            current_len += sentence_len
        else:
            if current:
                chunks.append(" ".join(current))
            if sentence_len > max_len:
                # Sentence too long: split at commas
                parts = re.split(r'(?<=[,;])\s+', sentence)
                sub_current, sub_len = [], 0
                for part in parts:
                    part_len = len(part) + 1
                    if sub_len + part_len <= max_len:
                        sub_current.append(part)
                        sub_len += part_len
                    else:
                        if sub_current:
                            chunks.append(" ".join(sub_current))
                        sub_current, sub_len = [part], part_len
                current, current_len = sub_current, sub_len
            else:
                current, current_len = [sentence], sentence_len
    if current:
        chunks.append(" ".join(current))
    return chunks

def truncate_for_preview(text: str, max_chars: int = 1500) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Try to end at a sentence boundary
    last_sentence_end = max(cut.rfind('।'), cut.rfind('?'), cut.rfind('!'))
    if last_sentence_end > max_chars // 2:
        return cut[:last_sentence_end + 1] + "..."
    return cut.rsplit(' ', 1)[0] + "..."

def format_date_for_api(date_obj: datetime) -> str:
    return date_obj.strftime("%d/%m/%Y")

# ------------------------------------------------------------
# FETCH PAGE LIST FROM PROTHOM ALO
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def fetch_prothomalo_pglist(edate: str):
    url = f"https://epaper.prothomalo.com/Home/DIndex?edate={edate}"
    headers = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "content-type": "application/json; charset=utf-8",
        "x-requested-with": "XMLHttpRequest",
        "referer": f"https://epaper.prothomalo.com/Home/DIndex?eid=1&edate={edate}&sedId=1&pgid=498326&isProductPanel=true&MagazineEdID=0&MagEdDate={edate}&isIssueRefresh=False&uemail=",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
    }
    cookies = {"ViewType_": "6", "currentPubId": "1", "EditionId": "1"}
    try:
        response = requests.get(url, headers=headers, cookies=cookies, timeout=30)
        response.raise_for_status()
        html_content = response.text
    except requests.RequestException:
        return None
    pattern = r'var\s+pglist_\s*=\s*(\[.*?\]);\s*(?:\/\/Generate|$)'
    match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    try:
        pglist_data = demjson3.decode(match.group(1))
        return pglist_data if isinstance(pglist_data, list) else None
    except Exception:
        return None

def build_page_options(pglist: List[Dict[str, Any]]) -> Dict[str, str]:
    options = {}
    for page in pglist:
        pgid = page.get("PageId")
        pgname = page.get("NewsProPageTitle", "Unknown")
        if pgid:
            options[f"Page {pgid}: {pgname}"] = str(pgid)
    return options

# ------------------------------------------------------------
# API FETCHING (CACHED)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_json(url: str):
    try:
        res = requests.get(url.strip(), timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception:
        return None

# ------------------------------------------------------------
# SCRAPING (CACHED)
# ------------------------------------------------------------
def _fetch_one_story(region_id: str) -> Dict[str, Any] | None:
    """Fetch and parse a single story by its OrgId. Returns {title, content} or None."""
    story_details = get_json(f"https://epaper.prothomalo.com/User/ShowArticleView?OrgId={region_id}")
    if not story_details:
        return None
    content_list = story_details.get("StoryContent", [])
    if not content_list:
        return None
    title = content_list[0].get("Headlines", "")
    if isinstance(title, list):
        title = " ".join(title)
    title = title.strip()
    if not title:
        return None
    body = content_list[0].get("Body", "")
    linked_id = story_details.get("LinkedStoryId", 0)
    if linked_id:
        detail = get_json(f"https://epaper.prothomalo.com/Home/getstorydetail?Storyid={linked_id}")
        if detail:
            detail_list = detail.get("StoryContent", [])
            if detail_list:
                body += " " + detail_list[0].get("Body", "")
    return {"title": title, "content": clean_html(body)}

@st.cache_data(show_spinner=False, ttl=3600)
def get_final_page_list(page_id: str):
    news_raw = get_json(f"https://epaper.prothomalo.com/Home/getStoriesOnPage?pageid={page_id}")
    if not news_raw:
        return []
    region_ids = [str(s["OrgId"]) for s in news_raw if s.get("OrgId")]
    if not region_ids:
        return []

    # Fetch all stories in parallel using a thread pool
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results_map: Dict[int, Dict] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {executor.submit(_fetch_one_story, rid): i for i, rid in enumerate(region_ids)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            if result:
                results_map[idx] = result

    # Return in original page order, skip empty
    return [results_map[i] for i in sorted(results_map) if results_map[i]["content"].strip()]

# ------------------------------------------------------------
# TTS GENERATION
# ------------------------------------------------------------
async def generate_audio_chunk(text: str, voice: str, rate: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    output = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            output += chunk["data"]
    return output

async def generate_all_chunks_async(chunks: List[str], voice: str, rate: str, progress_bar) -> bytes:
    """Generate all chunks in parallel, then stitch in order."""
    total = len(chunks)
    completed = 0

    async def tracked(coro):
        nonlocal completed
        result = await coro
        completed += 1
        progress_bar.progress(completed / total)
        return result

    tasks = [tracked(generate_audio_chunk(chunk, voice, rate)) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    return b"".join(results)

def estimate_reading_time(text: str, speed_pct: int) -> str:
    """Estimate TTS reading duration from character count and speed."""
    # Bangla TTS reads ~10–12 chars/sec at normal speed
    base_chars_per_sec = 11.0
    multiplier = 1 + (speed_pct / 100)
    chars_per_sec = base_chars_per_sec * max(multiplier, 0.1)
    total_seconds = len(text) / chars_per_sec
    mins = int(total_seconds // 60)
    secs = int(total_seconds % 60)
    if mins == 0:
        return f"~{secs}s"
    return f"~{mins}m {secs}s"

def generate_audio(text: str, use_preview: bool = False, voice: str = "bn-IN-BashkarNeural", rate: str = "+0%"):
    if use_preview:
        text = truncate_for_preview(text, max_chars=1500)
    chunks = chunk_text(text, max_len=2500)
    if not chunks:
        return b""
    progress_bar = st.progress(0)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_bytes = loop.run_until_complete(
            generate_all_chunks_async(chunks, voice, rate, progress_bar)
        )
        return audio_bytes
    finally:
        loop.close()
        progress_bar.empty()

@st.cache_data(show_spinner=False)
def get_cached_audio(title: str, content: str, use_preview: bool = False, voice: str = "bn-IN-BashkarNeural", rate: str = "+0%"):
    text = preprocess_for_tts(title, content)
    return generate_audio(text, use_preview, voice, rate)

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if 'audio_cache' not in st.session_state:
    st.session_state.audio_cache = {}

# ------------------------------------------------------------
# HERO HEADER
# ------------------------------------------------------------
today = datetime.now()
today_date_str = format_date_for_api(today)
formatted_date = today.strftime("%d %B, %Y")

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1a1040 0%, #0e153a 50%, #0a0c18 100%);
    border: 1px solid rgba(100,108,255,0.2);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
">
  <!-- Glow blob -->
  <div style="
    position:absolute; top:-60px; right:-60px;
    width:280px; height:280px;
    background: radial-gradient(circle, rgba(61,58,248,0.25) 0%, transparent 70%);
    border-radius:50%; pointer-events:none;
  "></div>

  <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.6rem;">
    <div style="
      background: linear-gradient(135deg,#3d3af8,#6c63ff);
      border-radius:14px; padding:10px 14px;
      font-size:1.8rem; line-height:1;
      box-shadow: 0 4px 20px rgba(61,58,248,0.45);
    ">🎙️</div>
    <div>
      <div style="
        font-size:1.85rem; font-weight:800; letter-spacing:-0.02em;
        background: linear-gradient(90deg, #ffffff 0%, #a5b4fc 60%, #818cf8 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        background-clip:text;
        font-family:'Inter',sans-serif;
      ">প্রথম আলো · Audio Reader</div>
      <div style="color:#6a72a8; font-size:0.9rem; margin-top:2px; font-weight:400;">
        Bangla E-Paper → Text-to-Speech, powered by Edge TTS
      </div>
    </div>
  </div>

  <div style="
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(61,58,248,0.12); border:1px solid rgba(61,58,248,0.3);
    border-radius:20px; padding:4px 14px; margin-top:0.8rem;
  ">
    <span style="font-size:0.75rem; color:#818cf8;">📅</span>
    <span style="font-size:0.8rem; color:#a5b4fc; font-weight:500;">{formatted_date}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    # Brand mark
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(61,58,248,0.15), rgba(108,99,255,0.08));
        border: 1px solid rgba(61,58,248,0.25);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 1.2rem;
        text-align: center;
    ">
        <div style="font-size:2rem; margin-bottom:4px;">📰</div>
        <div style="font-size:0.95rem; font-weight:700; color:#c8ceff; letter-spacing:0.01em;">
            আজকের সংস্করণ
        </div>
        <div style="font-size:0.75rem; color:#6a72a8; margin-top:2px;">Today's Edition</div>
    </div>
    """, unsafe_allow_html=True)

    # Page list
    with st.spinner("📋 Loading pages…"):
        pglist = fetch_prothomalo_pglist(edate=today_date_str)

    if pglist:
        page_options = build_page_options(pglist)
        if page_options:
            st.markdown('<p style="font-size:0.78rem;color:#6a72a8;font-weight:600;letter-spacing:0.06em;margin-bottom:4px;">SELECT PAGE</p>', unsafe_allow_html=True)
            selected_label = st.selectbox(
                "Page",
                options=list(page_options.keys()),
                index=0,
                label_visibility="collapsed",
                help="Choose an e-paper page"
            )
            selected_page_id = page_options[selected_label]
            st.session_state.selected_page_id = selected_page_id
            st.session_state.selected_page_label = selected_label

            st.markdown(f"""
            <div style="
                background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.25);
                border-radius:10px; padding:8px 14px; margin-top:8px;
                font-size:0.8rem; color:#4ade80; font-weight:500;
            ">✅ {len(page_options)} pages available</div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ No pages available for today.")
            st.session_state.selected_page_id = None
    else:
        st.error("❌ Failed to load page list.")
        st.session_state.selected_page_id = None

    st.markdown("<div style='margin:1.4rem 0; border-top:1px solid rgba(100,108,255,0.15);'></div>", unsafe_allow_html=True)

    # Audio settings card
    st.markdown("""
    <div style="
        font-size:0.78rem; color:#6a72a8; font-weight:600;
        letter-spacing:0.06em; margin-bottom:10px;
    ">⚙️ AUDIO SETTINGS</div>
    """, unsafe_allow_html=True)

    voice_option = st.selectbox(
        "🎤 Voice",
        ["bn-IN-BashkarNeural", "bn-IN-TanishaaNeural"],
        index=0,
        help="Choose TTS voice"
    )

    # Speed slider: -50% (slow) → +100% (fast), default 0%
    speed_pct = st.slider(
        "🚀 Reading Speed",
        min_value=-50,
        max_value=100,
        value=0,
        step=10,
        format="%d%%",
        help="Adjust speech rate. 0% = normal, negative = slower, positive = faster."
    )
    speed_option = f"+{speed_pct}%" if speed_pct >= 0 else f"{speed_pct}%"

    # Speed label badge
    if speed_pct < 0:
        speed_color, speed_label = "#60a5fa", "Slow"
    elif speed_pct == 0:
        speed_color, speed_label = "#a5b4fc", "Normal"
    elif speed_pct <= 50:
        speed_color, speed_label = "#f59e0b", "Fast"
    else:
        speed_color, speed_label = "#ef4444", "Very Fast"

    st.markdown(f"""
    <div style="
        background:{speed_color}18; border:1px solid {speed_color}55;
        border-radius:8px; padding:5px 12px; margin-top:4px;
        font-size:0.78rem; color:{speed_color}; font-weight:600; text-align:center;
    ">🎵 {speed_label} · {speed_option}</div>
    """, unsafe_allow_html=True)

    preview_mode = st.checkbox(
        "⚡ Quick Preview mode",
        value=False,
        help="Generate audio for first 1 500 characters only – much faster for long articles"
    )

    st.markdown("""
    <div style="
        background:rgba(61,58,248,0.07); border:1px solid rgba(61,58,248,0.18);
        border-radius:10px; padding:10px 14px; margin-top:14px;
        font-size:0.78rem; color:#6a72a8; line-height:1.6;
    ">
        💡 <strong style="color:#a5b4fc;">Tip:</strong> Enable Quick Preview for articles longer than 3 000 characters to reduce wait time.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# MAIN CONTENT
# ------------------------------------------------------------
if st.session_state.get('selected_page_id'):
    page_id = st.session_state.selected_page_id

    # Page header
    page_label = st.session_state.get('selected_page_label', 'Unknown')
    st.markdown(f"""
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        margin-bottom:1.5rem; flex-wrap:wrap; gap:0.75rem;
    ">
        <div>
            <div style="font-size:1.25rem;font-weight:700;color:#e8eaff;">
                📄 {page_label}
            </div>
            <div style="font-size:0.8rem;color:#6a72a8;margin-top:4px;">
                Edition · {today.strftime('%d/%m/%Y')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔍 Fetching articles…"):
        news_list = get_final_page_list(page_id)

    if not news_list:
        st.error("❌ No news found on this page.")
        st.stop()

    # Summary chip
    st.markdown(f"""
    <div style="
        display:inline-flex; align-items:center; gap:8px;
        background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.2);
        border-radius:20px; padding:6px 16px; margin-bottom:1.5rem;
        font-size:0.83rem; color:#4ade80; font-weight:500;
    ">📋 {len(news_list)} articles found on this page</div>
    """, unsafe_allow_html=True)

    # ── Article cards ──
    for idx, item in enumerate(news_list):

        # Card wrapper
        st.markdown(f"""
        <div style="
            background: linear-gradient(145deg, #0f1228 0%, #111530 100%);
            border: 1px solid rgba(100,108,255,0.15);
            border-radius: 16px;
            padding: 1.5rem 1.75rem 1.25rem;
            margin-bottom: 1.25rem;
            transition: border-color 0.3s;
            position: relative;
            overflow: hidden;
        ">
          <!-- Subtle left accent bar -->
          <div style="
            position:absolute; left:0; top:16px; bottom:16px;
            width:3px; border-radius:0 3px 3px 0;
            background: linear-gradient(180deg,#3d3af8,#6c63ff);
          "></div>

          <!-- Title -->
          <div style="padding-left:10px;">
            <h3 style="
              font-size:1.05rem; font-weight:700;
              color:#e8eaff; line-height:1.45;
              font-family:'Inter','Noto Sans Bengali',sans-serif;
              margin:0;
            ">{item['title']}</h3>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Article length + reading time indicator
        char_count = len(item['content'])
        if char_count < 1000:
            length_color, length_label = "#4ade80", "Short"
        elif char_count < 3000:
            length_color, length_label = "#f59e0b", "Medium"
        else:
            length_color, length_label = "#ef4444", "Long"
        read_time = estimate_reading_time(item['content'], speed_pct)
        st.markdown(f"""
        <div style="font-size:0.75rem; color:{length_color}; margin-bottom:0.5rem; opacity:0.8;">
            📝 {length_label} &nbsp;·&nbsp; {char_count:,} chars &nbsp;·&nbsp; 🕐 {read_time}
            {'&nbsp;·&nbsp; ⚡ Quick Preview recommended' if char_count > 3000 and not preview_mode else ''}
        </div>
        """, unsafe_allow_html=True)

        # Generate Audio button (full width)
        btn_label = "⚡ Quick Preview" if preview_mode else "▶ Generate Audio"
        play_key = f"play_{idx}"
        if st.button(btn_label, key=play_key, use_container_width=True):
            with st.spinner("🎧 Synthesising speech…"):
                audio_data = get_cached_audio(item['title'], item['content'], preview_mode, voice_option, speed_option)
            if audio_data:
                st.session_state.audio_cache[idx] = audio_data
            else:
                st.error("⚠️ Audio generation failed, please try again.")

        # Audio player
        if idx in st.session_state.audio_cache:
            st.markdown('<div style="margin-top:0.4rem;"></div>', unsafe_allow_html=True)
            st.audio(st.session_state.audio_cache[idx], format="audio/mp3")

        # Article text expander
        with st.expander("📄 Read full article"):
            st.markdown(f'<p style="color:#a8b0d8;line-height:1.8;font-size:0.9rem;">{item["content"]}</p>', unsafe_allow_html=True)

        st.markdown("<div style='height:0.25rem;'></div>", unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="
        text-align:center;
        padding: 5rem 2rem;
        background: linear-gradient(145deg, #0f1228, #111530);
        border: 1px dashed rgba(100,108,255,0.25);
        border-radius: 20px;
        margin-top: 1rem;
    ">
        <div style="font-size:4rem; margin-bottom:1rem;">📰</div>
        <div style="font-size:1.3rem; font-weight:700; color:#c8ceff; margin-bottom:0.5rem;">
            Ready to Read
        </div>
        <div style="font-size:0.9rem; color:#6a72a8; max-width:360px; margin:0 auto;">
            Select a page from the sidebar to load today's articles and convert them to audio.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("""
<div style="
    margin-top: 3rem;
    padding: 1.5rem 2rem;
    background: linear-gradient(135deg, #0e1128, #0a0c18);
    border: 1px solid rgba(100,108,255,0.12);
    border-radius: 16px;
    display:flex; justify-content:space-between; align-items:center;
    flex-wrap:wrap; gap:0.75rem;
">
    <div>
        <span style="font-size:0.85rem; font-weight:700; color:#6c63ff;">🎙️ প্রথম আলো · Audio Reader</span>
        <span style="font-size:0.75rem; color:#374060; margin-left:10px;">v2.0</span>
    </div>
    <div style="font-size:0.78rem; color:#4a5280; line-height:1.6; text-align:right;">
        ⚡ Use <strong style="color:#6c63ff;">Quick Preview</strong> for articles &gt; 3 000 chars &nbsp;·&nbsp;
        🔁 Audio is cached for repeated plays
    </div>
</div>
""", unsafe_allow_html=True)