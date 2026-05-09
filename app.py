import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import re
import demjson3
from datetime import datetime
from pathlib import Path

import audio_pipeline

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
# TTS GENERATION (delegates to audio_pipeline)
# ------------------------------------------------------------
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

# Single-article generation lives in audio_pipeline.synthesize_article;
# bulletin assembly lives in audio_pipeline.get_or_build_bulletin.

@st.cache_data(show_spinner=False, max_entries=64)
def get_cached_article_audio(
    title: str,
    content: str,
    voice: str,
    base_rate: str,
    normalize_numbers: bool,
    style_version: str = audio_pipeline.STYLE_VERSION,
) -> bytes:
    """In-memory cache for per-article audio. Cross-call re-renders are instant.

    The style_version param is part of the key — bumping audio_pipeline
    invalidates everything cached here without manual cache clears.
    """
    return audio_pipeline.synthesize_article(
        title=title,
        content=content,
        voice=voice,
        base_rate=base_rate,
        normalize_numbers=normalize_numbers,
    )

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
# bulletin_store: keyed by (edate, page_id, voice, base_rate, normalize_numbers)
#   → (mp3_bytes, chapters_ms)
if 'bulletin_store' not in st.session_state:
    st.session_state.bulletin_store = {}

# article_audio: per-card audio so clicked-to-play stories survive reruns.
# Keyed by article index within the current page.
if 'article_audio' not in st.session_state:
    st.session_state.article_audio = {}

# ffmpeg / pydub availability — checked once at startup, surfaced as a banner.
_FFMPEG_OK, _FFMPEG_MSG = audio_pipeline.ffmpeg_status()

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

    # Speed slider: -50% (slow) → +100% (fast), default 0%, 5% steps
    speed_pct = st.slider(
        "🚀 Reading Speed",
        min_value=-50,
        max_value=100,
        value=0,
        step=5,
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

    normalize_numbers = st.checkbox(
        "🔢 Read numbers in Bangla words",
        value=True,
        help="Convert digits like ২০২৫ to spoken Bangla words (দুই হাজার পঁচিশ). "
             "Also handles dates, percentages, currency, and phone numbers."
    )

    st.markdown("""
    <div style="
        background:rgba(61,58,248,0.07); border:1px solid rgba(61,58,248,0.18);
        border-radius:10px; padding:10px 14px; margin-top:14px;
        font-size:0.78rem; color:#6a72a8; line-height:1.6;
    ">
        📻 <strong style="color:#a5b4fc;">Bulletin mode:</strong> the whole page plays as one continuous newscast — anchor-style headlines, real silences between stories, cached on disk for instant replay.
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

    # ── Bulletin controls ──
    bulletin_key = (today_date_str, page_id, voice_option, speed_option, normalize_numbers)
    bulletin_entry = st.session_state.bulletin_store.get(bulletin_key)
    bulletin_audio = bulletin_entry[0] if bulletin_entry else None
    chapters_ms: List[int] = bulletin_entry[1] if bulletin_entry else []

    est_total_ms = audio_pipeline.estimate_bulletin_duration_ms(news_list, speed_pct)
    est_str = audio_pipeline.format_duration(est_total_ms)

    summary_col, btn_col = st.columns([2, 1])
    with summary_col:
        st.markdown(f"""
        <div style="
            display:inline-flex; align-items:center; gap:10px;
            background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.2);
            border-radius:20px; padding:8px 18px;
            font-size:0.85rem; color:#4ade80; font-weight:500;
        ">📋 {len(news_list)} articles &nbsp;·&nbsp; 🕐 ~{est_str} bulletin</div>
        """, unsafe_allow_html=True)

    with btn_col:
        if not _FFMPEG_OK:
            st.error(f"📻 Bulletin disabled — {_FFMPEG_MSG}")
        else:
            btn_label = "🔁 Replay Bulletin" if bulletin_audio else f"📻 Play Full Bulletin (~{est_str})"
            if st.button(btn_label, key="play_bulletin", use_container_width=True, type="primary"):
                with st.status("🎙️ Generating bulletin…", expanded=True) as status:
                    progress_bar = st.progress(0, text="Starting…")
                    msg_box = st.empty()

                    def on_progress(done, total):
                        progress_bar.progress(done / total, text=f"Synthesising segment {done} of {total}")

                    try:
                        msg_box.markdown(f"Building {len(news_list)} articles into one continuous broadcast…")
                        mp3, chapters = audio_pipeline.get_or_build_bulletin(
                            news_list,
                            edate=today_date_str,
                            page_id=page_id,
                            voice=voice_option,
                            base_rate=speed_option,
                            normalize_numbers=normalize_numbers,
                            progress_cb=on_progress,
                        )
                        st.session_state.bulletin_store[bulletin_key] = (mp3, chapters)
                        bulletin_audio, chapters_ms = mp3, chapters
                        status.update(label="✅ Bulletin ready", state="complete", expanded=False)
                    except Exception as e:
                        status.update(label=f"❌ Bulletin failed: {e}", state="error")
                        st.exception(e)

    # ── Bulletin player (sticky-ish at top once generated) ──
    if bulletin_audio:
        st.markdown('<div style="margin:1rem 0 1.5rem;"></div>', unsafe_allow_html=True)
        st.audio(bulletin_audio, format="audio/mp3")
        st.markdown(f"""
        <div style="font-size:0.75rem; color:#6a72a8; margin-top:0.4rem; text-align:right;">
            📦 {len(bulletin_audio)//1024:,} KB &nbsp;·&nbsp; cached on disk for instant replay
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background:rgba(61,58,248,0.06); border:1px dashed rgba(100,108,255,0.25);
            border-radius:14px; padding:1.2rem 1.5rem; margin:1rem 0 1.5rem;
            font-size:0.85rem; color:#8b85ff; text-align:center;
        ">
            🎙️ Click <strong>Play Full Bulletin</strong> above to generate today's broadcast.
        </div>
        """, unsafe_allow_html=True)

    # ── Article cards (chapter markers) ──
    st.markdown('<div style="font-size:0.78rem;color:#6a72a8;font-weight:600;letter-spacing:0.06em;margin:1.5rem 0 0.75rem;">STORY LIST</div>', unsafe_allow_html=True)

    for idx, item in enumerate(news_list):

        # Chapter timestamp (only if bulletin is loaded)
        if chapters_ms and idx < len(chapters_ms):
            ts = audio_pipeline.format_duration(chapters_ms[idx])
            ts_html = f"""
              <div style="
                position:absolute; right:1rem; top:1.1rem;
                background:rgba(61,58,248,0.18); border:1px solid rgba(61,58,248,0.35);
                border-radius:6px; padding:3px 10px;
                font-size:0.72rem; font-weight:600; color:#a5b4fc;
                font-variant-numeric: tabular-nums;
              ">⏱ {ts}</div>
            """
        else:
            ts_html = ""

        # Card wrapper
        st.markdown(f"""
        <div style="
            background: linear-gradient(145deg, #0f1228 0%, #111530 100%);
            border: 1px solid rgba(100,108,255,0.15);
            border-radius: 16px;
            padding: 1.1rem 1.5rem 1rem;
            margin-bottom: 0.9rem;
            position: relative;
            overflow: hidden;
        ">
          <!-- Subtle left accent bar -->
          <div style="
            position:absolute; left:0; top:14px; bottom:14px;
            width:3px; border-radius:0 3px 3px 0;
            background: linear-gradient(180deg,#3d3af8,#6c63ff);
          "></div>
          {ts_html}

          <div style="padding-left:10px; padding-right:{'4.5rem' if ts_html else '0'};">
            <div style="font-size:0.7rem; color:#6a72a8; font-weight:600; margin-bottom:3px;">
              #{idx+1}
            </div>
            <h3 style="
              font-size:1rem; font-weight:700;
              color:#e8eaff; line-height:1.45;
              font-family:'Inter','Noto Sans Bengali',sans-serif;
              margin:0;
            ">{item['title']}</h3>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Per-card listen controls — generate just this story.
        char_count = len(item['content'])
        read_time = estimate_reading_time(item['content'], speed_pct)
        article_key = (page_id, idx, voice_option, speed_option, normalize_numbers)
        article_audio_bytes = st.session_state.article_audio.get(article_key)

        listen_col, info_col = st.columns([1, 2])
        with listen_col:
            listen_label = "🔁 Replay this story" if article_audio_bytes else f"▶ Listen ({read_time})"
            if st.button(listen_label, key=f"listen_{idx}", use_container_width=True):
                with st.spinner(f"🎧 Synthesising story #{idx+1}…"):
                    progress_bar = st.progress(0)
                    try:
                        def on_prog(done, total):
                            progress_bar.progress(done / total)
                        article_audio_bytes = get_cached_article_audio(
                            title=item['title'],
                            content=item['content'],
                            voice=voice_option,
                            base_rate=speed_option,
                            normalize_numbers=normalize_numbers,
                        )
                        st.session_state.article_audio[article_key] = article_audio_bytes
                    finally:
                        progress_bar.empty()
        with info_col:
            st.markdown(f"""
            <div style="font-size:0.75rem; color:#6a72a8; padding-top:0.55rem;">
                {char_count:,} chars · {read_time} reading time
            </div>
            """, unsafe_allow_html=True)

        if article_audio_bytes:
            st.audio(article_audio_bytes, format="audio/mp3")

        # Article text expander
        with st.expander(f"📄 Read full text"):
            st.markdown(f'<p style="color:#a8b0d8;line-height:1.8;font-size:0.9rem;">{item["content"]}</p>', unsafe_allow_html=True)

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
        <span style="font-size:0.75rem; color:#374060; margin-left:10px;">v3.0 · Bulletin</span>
    </div>
    <div style="font-size:0.78rem; color:#4a5280; line-height:1.6; text-align:right;">
        📻 <strong style="color:#6c63ff;">Radio bulletin mode</strong> &nbsp;·&nbsp;
        💾 Bulletins disk-cached at <code style="color:#8b85ff;">~/.cache/prothomalo-audio-reader/</code>
    </div>
</div>
""", unsafe_allow_html=True)