"""
Audio pipeline for Prothom Alo Audio Reader.

Splits articles into prosody-aware segments (anchor-style headline + lede + body
chunks), generates each segment in parallel via edge_tts, and assembles them
into a continuous broadcast.

Phase 1: Segment generation only — chunks are concatenated as raw MP3 bytes.
Phase 2 (added later) layers pydub on top for proper silences, loudness
normalisation, and a single bulletin MP3 covering all articles.

edge_tts hard constraint: the library auto-XML-escapes user text, so SSML tags
like <break> are impossible. Every prosody change must therefore be its own
chunk with its own rate / pitch / volume params.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Awaitable, Callable, Iterable, List, Optional, Tuple, TypeVar

import edge_tts

import bn_normalize

# pydub is optional at module-import time — we surface a friendly error if
# ffmpeg is missing rather than crashing later during synthesis.
try:
    from pydub import AudioSegment
    from pydub.utils import which as _pydub_which
    _PYDUB_IMPORT_ERROR: Optional[str] = None
except ImportError as _e:  # pragma: no cover
    AudioSegment = None  # type: ignore
    _pydub_which = None  # type: ignore
    _PYDUB_IMPORT_ERROR = str(_e)


def ffmpeg_status() -> Tuple[bool, str]:
    """Return (ok, message) describing whether bulletin assembly is usable.

    Called from the Streamlit UI at startup to render a friendly error banner
    instead of letting the user discover the problem mid-synthesis.
    """
    if _PYDUB_IMPORT_ERROR is not None:
        return False, f"pydub is not installed: {_PYDUB_IMPORT_ERROR}"
    assert _pydub_which is not None
    if _pydub_which("ffmpeg") is None:
        return False, (
            "ffmpeg is not on PATH — install it: "
            "`sudo apt install ffmpeg` (Ubuntu/Debian) or `brew install ffmpeg` (macOS)."
        )
    return True, "ok"

# ────────────────────────────────────────────────────────────────────────────
# Versioning — bump when prosody / pause / normalisation changes,
# so the disk-cached bulletins (Phase 2) invalidate cleanly.
# ────────────────────────────────────────────────────────────────────────────
STYLE_VERSION = "v5"

# ────────────────────────────────────────────────────────────────────────────
# Per-voice prosody tuning. Tanishaa's base pitch is higher than Bashkar's,
# so she needs a deeper drop to land in the same authoritative register.
# ────────────────────────────────────────────────────────────────────────────
VOICE_PROFILES = {
    "bn-IN-BashkarNeural":  {"headline_pitch_hz": -15},
    "bn-IN-TanishaaNeural": {"headline_pitch_hz": -20},
}
_DEFAULT_HEADLINE_PITCH_HZ = -15

# Pause durations (ms). Tunable.
PAUSE_AFTER_HEADLINE_MS  = 700
PAUSE_AFTER_LEDE_MS      = 350
PAUSE_BETWEEN_CHUNKS_MS  = 350
PAUSE_BETWEEN_ARTICLES_MS = 1200

# Prosody deltas (percentage points relative to user's base rate).
HEADLINE_RATE_DELTA = -12
LEDE_RATE_DELTA     = -6

# edge_tts rate is clamped to a sane range.
_MIN_RATE_PCT = -50
_MAX_RATE_PCT = 100

# Concurrency cap for edge_tts connections.
_TTS_CONCURRENCY = 16

# Max characters per body chunk, sentence-aligned.
_BODY_CHUNK_MAX_CHARS = 2500

# ────────────────────────────────────────────────────────────────────────────
# Noise patterns from scraped Prothom Alo HTML that sound terrible aloud.
# Reused from the previous flat preprocess_for_tts pipeline.
# ────────────────────────────────────────────────────────────────────────────
_NOISE_PATTERNS = [
    r'\(?\s*ফাইল\s*ছবি\s*\)?',
    r'\(?\s*ছবি\s*[:：]\s*[^)।\n]*\)?',
    r'\(?\s*প্রতিবেদক\s*\)?',
    r'\(?\s*সংবাদদাতা\s*\)?',
    r'\(?\s*বিশেষ\s*প্রতিনিধি\s*\)?',
    r'\(?\s*স্টাফ\s*রিপোর্টার\s*\)?',
    # Jump-page markers — "(প্রথম পৃষ্ঠার পর)", "শেষ পৃষ্ঠার পর", "১১ পৃষ্ঠার পর",
    # "২ নম্বর পৃষ্ঠার পর". Qualifier tokens cannot contain sentence enders or
    # parens, so the match never crosses a sentence boundary into prior text.
    r'\(?\s*(?:[^\s।!?()]+\s+){1,3}পৃষ্ঠার\s*পর\s*\)?',
    # "(চলবে)" / "(চলছে)" — to-be-continued markers. Bare চলবে is ambiguous
    # (it's also a regular verb), so we only strip the parenthesised form.
    r'\(\s*চল(?:বে|ছে)\s*\)',
    r'https?://\S+',
    r'\*{2,}',
]


# ────────────────────────────────────────────────────────────────────────────
# Data model
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Segment:
    """One TTS request: text + prosody + advisory pause to follow.

    The pause_after_ms field is used by the Phase 2 pydub assembler to insert
    real silence between chunks. In Phase 1 it's metadata only.
    """
    text: str
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    pause_after_ms: int = 0
    article_index: Optional[int] = field(default=None, compare=False)


# ────────────────────────────────────────────────────────────────────────────
# Rate composition
# ────────────────────────────────────────────────────────────────────────────
def compose_rate(base_rate: str, delta_pct: int) -> str:
    """Apply a percentage-point delta to a base rate string.

    >>> compose_rate("+0%", -12)
    '-12%'
    >>> compose_rate("+20%", -12)
    '+8%'
    >>> compose_rate("-30%", -30)
    '-50%'
    """
    m = re.match(r'^([+-])(\d+)%$', base_rate.strip())
    if not m:
        # Fall back: treat unknown base as 0
        base = 0
    else:
        sign, mag = m.group(1), int(m.group(2))
        base = mag if sign == '+' else -mag

    new_pct = max(_MIN_RATE_PCT, min(_MAX_RATE_PCT, base + delta_pct))
    if new_pct >= 0:
        return f"+{new_pct}%"
    return f"{new_pct}%"


def _format_pitch_hz(hz: int) -> str:
    """Format an integer Hz value into edge_tts's required ±NHz string."""
    if hz >= 0:
        return f"+{hz}Hz"
    return f"{hz}Hz"


# ────────────────────────────────────────────────────────────────────────────
# Text cleaning — extracted from the legacy preprocess_for_tts so title and
# body can be cleaned independently and emitted as separate segments.
# ────────────────────────────────────────────────────────────────────────────
def _clean_body(content: str) -> str:
    """Strip noise, convert paragraph breaks into sentence pauses, normalise."""
    text = content.strip()
    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Paragraph breaks → Bangla sentence ending so edge_tts produces a beat.
    text = re.sub(r'\n\n+', '। ', text)
    text = re.sub(r'\n', ' ', text)
    # Collapse repeated punctuation FIRST (e.g. "।।" → "।") so the spacing pass
    # below doesn't insert a space between the duplicates and prevent collapse.
    text = re.sub(r'([।!?,;])\s*\1+', r'\1', text)
    # Spacing after sentence enders
    text = re.sub(r'([।!?])([^\s])', r'\1 \2', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _clean_title(title: str) -> str:
    """Light cleanup for headlines — preserves question marks etc."""
    t = title.strip()
    # Drop a trailing Bengali danda if present; the segment boundary already
    # provides the break, doubling it sounds awkward in anchor delivery.
    t = t.rstrip('।').strip()
    t = re.sub(r'\s+', ' ', t)
    return t


def _split_sentences(text: str) -> List[str]:
    return [s for s in re.split(r'(?<=[।!?])\s+', text) if s.strip()]


def _chunk_sentences(sentences: List[str], max_len: int = _BODY_CHUNK_MAX_CHARS) -> List[str]:
    """Group sentences into chunks under max_len chars; never split a sentence."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        slen = len(sentence) + 1
        if current_len + slen <= max_len:
            current.append(sentence)
            current_len += slen
        else:
            if current:
                chunks.append(" ".join(current))
            if slen > max_len:
                # Pathologically long single sentence: fall back to comma split.
                parts = re.split(r'(?<=[,;])\s+', sentence)
                sub: List[str] = []
                sub_len = 0
                for p in parts:
                    plen = len(p) + 1
                    if sub_len + plen <= max_len:
                        sub.append(p)
                        sub_len += plen
                    else:
                        if sub:
                            chunks.append(" ".join(sub))
                        sub = [p]
                        sub_len = plen
                current = sub
                current_len = sub_len
            else:
                current = [sentence]
                current_len = slen
    if current:
        chunks.append(" ".join(current))
    return chunks


# ────────────────────────────────────────────────────────────────────────────
# Segment builders
# ────────────────────────────────────────────────────────────────────────────
def build_article_segments(
    title: str,
    content: str,
    base_rate: str,
    voice: str,
    *,
    article_index: Optional[int] = None,
    normalize_numbers: bool = True,
) -> List[Segment]:
    """Build Segments for a single article: headline → lede → body chunks.

    The first body sentence ('lede') gets a small slowdown to help the listener
    catch the topic, then the rest of the body plays at the user's base rate.

    If normalize_numbers is True, dates/years/currencies/percentages/phone
    numbers in the title and body are spelled out as Bangla words before
    chunking, so edge_tts pronounces them naturally.
    """
    title_clean = _clean_title(title)
    body_clean = _clean_body(content)
    if normalize_numbers:
        title_clean = bn_normalize.normalize_all(title_clean)
        body_clean = bn_normalize.normalize_all(body_clean)

    headline_pitch_hz = VOICE_PROFILES.get(voice, {}).get(
        "headline_pitch_hz", _DEFAULT_HEADLINE_PITCH_HZ
    )

    segments: List[Segment] = []

    if title_clean:
        segments.append(Segment(
            text=title_clean,
            rate=compose_rate(base_rate, HEADLINE_RATE_DELTA),
            pitch=_format_pitch_hz(headline_pitch_hz),
            pause_after_ms=PAUSE_AFTER_HEADLINE_MS,
            article_index=article_index,
        ))

    if not body_clean:
        return segments

    sentences = _split_sentences(body_clean)
    if not sentences:
        # Body had content but no sentence enders — emit as one chunk.
        segments.append(Segment(
            text=body_clean,
            rate=base_rate,
            pause_after_ms=PAUSE_BETWEEN_CHUNKS_MS,
            article_index=article_index,
        ))
        return segments

    # Lede: first sentence on its own, slightly slower.
    lede = sentences[0]
    rest = sentences[1:]
    segments.append(Segment(
        text=lede,
        rate=compose_rate(base_rate, LEDE_RATE_DELTA),
        pause_after_ms=PAUSE_AFTER_LEDE_MS,
        article_index=article_index,
    ))

    # Remaining sentences re-grouped into ≤2500-char chunks.
    if rest:
        body_chunks = _chunk_sentences(rest)
        for chunk in body_chunks:
            segments.append(Segment(
                text=chunk,
                rate=base_rate,
                pause_after_ms=PAUSE_BETWEEN_CHUNKS_MS,
                article_index=article_index,
            ))

    return segments


# ────────────────────────────────────────────────────────────────────────────
# Async generation
# ────────────────────────────────────────────────────────────────────────────
async def generate_segment(seg: Segment, voice: str) -> bytes:
    """Synthesise one Segment to MP3 bytes via edge_tts."""
    communicate = edge_tts.Communicate(
        seg.text,
        voice,
        rate=seg.rate,
        pitch=seg.pitch,
        volume=seg.volume,
    )
    out = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            out += chunk["data"]
    return out


ProgressCb = Optional[Callable[[int, int], None]]


async def generate_segments_async(
    segments: List[Segment],
    voice: str,
    progress_cb: ProgressCb = None,
) -> List[bytes]:
    """Generate all segments in parallel, capped by a connection semaphore.

    Returns blobs in the same order as `segments`. Calls progress_cb(done, total)
    after each segment finishes.
    """
    total = len(segments)
    sem = asyncio.Semaphore(_TTS_CONCURRENCY)
    done = 0
    lock = asyncio.Lock()

    async def run_one(seg: Segment) -> bytes:
        nonlocal done
        async with sem:
            blob = await generate_segment(seg, voice)
        async with lock:
            done += 1
            if progress_cb is not None:
                try:
                    progress_cb(done, total)
                except Exception:
                    pass  # progress callback errors should never abort generation
        return blob

    return await asyncio.gather(*(run_one(s) for s in segments))


# ────────────────────────────────────────────────────────────────────────────
# Async runner — replaces the previous per-call asyncio.new_event_loop()
# pattern with one that's safe to call repeatedly from sync Streamlit code.
# ────────────────────────────────────────────────────────────────────────────
T = TypeVar("T")


def run_async(coro: Awaitable[T]) -> T:
    """Run a coroutine from sync code, reusing or creating an event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already inside an event loop (rare in Streamlit but possible
            # under newer versions). Create a fresh loop for this call.
            raise RuntimeError("loop running")
        if loop.is_closed():
            raise RuntimeError("loop closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ────────────────────────────────────────────────────────────────────────────
# High-level convenience for Phase 1: build + generate + concat (no pydub).
# ────────────────────────────────────────────────────────────────────────────
def synthesize_article(
    title: str,
    content: str,
    voice: str,
    base_rate: str,
    *,
    normalize_numbers: bool = True,
    progress_cb: ProgressCb = None,
) -> bytes:
    """Synthesise one article to a single MP3 blob.

    If pydub + ffmpeg are available, segments are stitched with proper silences
    and loudness normalisation. Otherwise falls back to raw MP3 byte
    concatenation so the app stays usable even without ffmpeg.
    """
    segments = build_article_segments(
        title, content, base_rate, voice,
        normalize_numbers=normalize_numbers,
    )
    if not segments:
        return b""
    blobs = run_async(generate_segments_async(segments, voice, progress_cb))
    ok, _ = ffmpeg_status()
    if ok:
        audio_bytes, _chapters = stitch_segments_to_mp3(blobs, segments)
        return audio_bytes
    return b"".join(blobs)


# ────────────────────────────────────────────────────────────────────────────
# Phase 2: pydub assembly — real silences, loudness normalisation,
# chapter-timestamp tracking, single MP3 export.
# ────────────────────────────────────────────────────────────────────────────
_TARGET_DBFS = -16.0       # broadcast-ish loudness target
_EXPORT_BITRATE = "96k"     # speech-grade MP3
_EXPORT_FORMAT = "mp3"


def stitch_segments_to_mp3(
    blobs: List[bytes],
    segments: List[Segment],
    *,
    target_dbfs: float = _TARGET_DBFS,
    export_bitrate: str = _EXPORT_BITRATE,
) -> Tuple[bytes, List[int]]:
    """Stitch per-segment MP3 blobs into one MP3 with proper silences.

    Returns:
      (mp3_bytes, chapter_start_ms_per_article)

    Where chapter_start_ms_per_article[i] is the millisecond offset at which
    article index i's first segment begins. Ordered by ascending article_index.

    No crossfade — crossfading speech smears phonemes. We rely on hard cuts +
    explicit AudioSegment.silent() inserts between segments.
    """
    if AudioSegment is None:
        raise RuntimeError(
            "pydub/ffmpeg unavailable — call ffmpeg_status() before stitching"
        )
    if len(blobs) != len(segments):
        raise ValueError(
            f"blob/segment length mismatch: {len(blobs)} vs {len(segments)}"
        )

    final = AudioSegment.silent(duration=0)
    chapter_starts: dict[int, int] = {}
    cursor_ms = 0
    last_idx = len(segments) - 1

    for i, (blob, seg) in enumerate(zip(blobs, segments)):
        if not blob:
            continue
        clip = AudioSegment.from_file(BytesIO(blob), format="mp3")

        # Loudness normalisation: pull each clip's dBFS toward the target.
        # Skip if the clip is silent (dBFS == -inf).
        if clip.dBFS != float("-inf"):
            clip = clip.apply_gain(target_dbfs - clip.dBFS)

        # Record chapter start for the first segment of each article.
        if seg.article_index is not None and seg.article_index not in chapter_starts:
            chapter_starts[seg.article_index] = cursor_ms

        final += clip
        cursor_ms += len(clip)

        # Inter-segment silence (skip after the very last segment).
        if i < last_idx and seg.pause_after_ms > 0:
            silence = AudioSegment.silent(duration=seg.pause_after_ms)
            final += silence
            cursor_ms += seg.pause_after_ms

    # Export to MP3 mono 96k (matches edge_tts source bitrate; smaller files).
    final = final.set_channels(1)
    out = BytesIO()
    final.export(out, format=_EXPORT_FORMAT, bitrate=export_bitrate)

    chapters_ordered = [chapter_starts[k] for k in sorted(chapter_starts)]
    return out.getvalue(), chapters_ordered


def build_bulletin_segments(
    news_list: List[dict],
    voice: str,
    base_rate: str,
    *,
    normalize_numbers: bool = True,
) -> List[Segment]:
    """Flatten all articles into one segment list with inter-article pauses.

    The last segment of each article (except the final article) gets its
    pause_after_ms bumped to PAUSE_BETWEEN_ARTICLES_MS so the listener feels a
    distinct gap between stories.
    """
    flat: List[Segment] = []
    n = len(news_list)
    for idx, item in enumerate(news_list):
        article_segs = build_article_segments(
            title=item.get("title", ""),
            content=item.get("content", ""),
            base_rate=base_rate,
            voice=voice,
            article_index=idx,
            normalize_numbers=normalize_numbers,
        )
        if not article_segs:
            continue
        # Bump trailing pause for non-final articles to the inter-article gap.
        if idx < n - 1:
            article_segs[-1] = Segment(
                text=article_segs[-1].text,
                rate=article_segs[-1].rate,
                pitch=article_segs[-1].pitch,
                volume=article_segs[-1].volume,
                pause_after_ms=PAUSE_BETWEEN_ARTICLES_MS,
                article_index=article_segs[-1].article_index,
            )
        flat.extend(article_segs)
    return flat


# ────────────────────────────────────────────────────────────────────────────
# Disk cache for assembled bulletins. Key includes everything that could
# change the audio: edition, page, voice, rate, article titles, style version.
# Stored as `{hash}.mp3` + sibling `{hash}.json` for chapter timestamps.
# ────────────────────────────────────────────────────────────────────────────
_CACHE_DIR = Path(os.path.expanduser("~/.cache/prothomalo-audio-reader/bulletins"))
_CACHE_BUDGET_BYTES = 500 * 1024 * 1024  # 500 MB LRU cap


def _bulletin_cache_key(
    edate: str,
    page_id: str,
    voice: str,
    base_rate: str,
    article_titles: Iterable[str],
    normalize_numbers: bool,
) -> str:
    payload = json.dumps(
        {
            "edate": edate,
            "page_id": page_id,
            "voice": voice,
            "base_rate": base_rate,
            "style_version": STYLE_VERSION,
            "normalize_numbers": normalize_numbers,
            "titles": list(article_titles),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _enforce_cache_budget(cache_dir: Path, budget_bytes: int) -> None:
    """Best-effort LRU eviction by mtime. Silently no-ops on errors."""
    try:
        files = [p for p in cache_dir.iterdir() if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        if total <= budget_bytes:
            return
        # Oldest first; remove until under budget.
        files.sort(key=lambda p: p.stat().st_mtime)
        for p in files:
            if total <= budget_bytes:
                break
            try:
                size = p.stat().st_size
                p.unlink()
                total -= size
            except OSError:
                continue
    except OSError as e:
        logging.warning("cache eviction skipped: %s", e)


def get_or_build_bulletin(
    news_list: List[dict],
    *,
    edate: str,
    page_id: str,
    voice: str,
    base_rate: str,
    normalize_numbers: bool = True,
    progress_cb: ProgressCb = None,
) -> Tuple[bytes, List[int]]:
    """Return (mp3_bytes, chapter_start_ms_list) for a full-page bulletin.

    Disk-cached at ~/.cache/prothomalo-audio-reader/bulletins/. Survives
    Streamlit restarts. LRU-evicted at 500MB total.

    Requires pydub + ffmpeg; raises RuntimeError if unavailable.
    """
    ok, msg = ffmpeg_status()
    if not ok:
        raise RuntimeError(f"Bulletin mode unavailable: {msg}")

    titles = [item.get("title", "") for item in news_list]
    key = _bulletin_cache_key(edate, page_id, voice, base_rate, titles, normalize_numbers)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mp3_path = _CACHE_DIR / f"{key}.mp3"
    meta_path = _CACHE_DIR / f"{key}.json"

    if mp3_path.exists() and meta_path.exists():
        try:
            chapters = json.loads(meta_path.read_text())["chapters"]
            # touch mtime so LRU treats this as recently used
            mp3_path.touch()
            meta_path.touch()
            return mp3_path.read_bytes(), chapters
        except (OSError, ValueError, KeyError) as e:
            logging.warning("bulletin cache read failed, regenerating: %s", e)

    segments = build_bulletin_segments(
        news_list, voice=voice, base_rate=base_rate,
        normalize_numbers=normalize_numbers,
    )
    if not segments:
        return b"", []

    blobs = run_async(generate_segments_async(segments, voice, progress_cb))
    mp3_bytes, chapters = stitch_segments_to_mp3(blobs, segments)

    try:
        mp3_path.write_bytes(mp3_bytes)
        meta_path.write_text(json.dumps({"chapters": chapters}))
        _enforce_cache_budget(_CACHE_DIR, _CACHE_BUDGET_BYTES)
    except OSError as e:
        logging.warning("bulletin cache write failed: %s", e)

    return mp3_bytes, chapters


def estimate_bulletin_duration_ms(news_list: List[dict], speed_pct: int) -> int:
    """Rough total duration of the bulletin including inter-article gaps.

    Used by the UI to show "~XX min" on the Play Bulletin button before the
    user commits to generation.
    """
    base_chars_per_sec = 11.0
    multiplier = max(0.1, 1 + (speed_pct / 100))
    chars_per_sec = base_chars_per_sec * multiplier

    total_chars = 0
    for item in news_list:
        total_chars += len(item.get("title", "")) + len(item.get("content", ""))

    speech_ms = int((total_chars / chars_per_sec) * 1000)
    n = len(news_list)
    # Per-article overhead: headline pause + lede pause + ~3 chunk pauses.
    per_article_overhead = (
        PAUSE_AFTER_HEADLINE_MS + PAUSE_AFTER_LEDE_MS + 3 * PAUSE_BETWEEN_CHUNKS_MS
    )
    overhead_ms = (
        n * per_article_overhead
        + max(0, n - 1) * PAUSE_BETWEEN_ARTICLES_MS
    )
    return speech_ms + overhead_ms


def format_duration(ms: int) -> str:
    """ms → 'mm:ss' or 'h:mm:ss' for very long bulletins."""
    total_s = max(0, ms) // 1000
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
