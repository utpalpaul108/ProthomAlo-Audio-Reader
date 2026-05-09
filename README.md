# ProthomAlo Audio Reader

A Streamlit app that turns the daily Prothom Alo Bangla e-paper into a continuous radio-style audio bulletin.

- **Anchor-style headlines** — each story's title is read with a slower rate and slightly lower pitch, like a real newscast.
- **Real silences** — paragraph breaks become 350 ms beats, story changes become 1.2 s gaps (powered by `pydub`).
- **Loudness normalised** — every chunk is pulled to the same dBFS so volume doesn't bounce article-to-article.
- **Disk-cached bulletins** — once you've generated a page's bulletin, replays are instant. Cache survives Streamlit restarts.
- **Bangla number normalization** *(Phase 3)* — `২০২৫` is spoken as "দুই হাজার পঁচিশ", dates and percentages spelled out.

## System dependencies

The bulletin assembler uses `pydub`, which needs `ffmpeg` available on `PATH`:

```bash
# Debian / Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (one option)
winget install ffmpeg
```

If `ffmpeg` is missing, the app surfaces a banner and disables bulletin mode.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (default `http://localhost:8501`).

## How to use

1. Pick today's edition page from the sidebar (e.g. *প্রথম পাতা*).
2. Choose a voice (`bn-IN-BashkarNeural` male / `bn-IN-TanishaaNeural` female).
3. Adjust reading speed if you want.
4. Click **📻 Play Full Bulletin** — the app synthesises every article on the page in parallel, stitches them with proper silences, and serves one continuous MP3.
5. Each article card on the page shows its **chapter timestamp** so you know where to scrub for any specific story.

## Project layout

```
app.py              Streamlit UI — page selector, sidebar, bulletin player, story-list cards
audio_pipeline.py   Segment-based TTS pipeline, parallel edge_tts generation,
                    pydub assembly, disk-cached bulletin builder
bn_normalize.py     Bangla number / date / currency / phone normalization (Phase 3)
style.css           Dark-theme overrides on top of Streamlit
```

## Cache location

Generated bulletins live at `~/.cache/prothomalo-audio-reader/bulletins/`, capped at 500 MB total (oldest evicted first). Delete the directory to force a fresh regeneration.
