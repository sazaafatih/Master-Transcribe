import os
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx
import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Video Transcriber API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MAX_CHUNK_BYTES = 20 * 1024 * 1024  # 20MB per chunk


class TranscribeRequest(BaseModel):
    url: str
    language: str = "id"


class TranslateRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return FileResponse("index.html")


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable not set")

    work_dir = Path(tempfile.mkdtemp())

    try:
        # Step 1: Download audio
        audio_path = download_audio(req.url, work_dir)

        # Step 2: Chunk if needed
        file_size = os.path.getsize(audio_path)
        if file_size > MAX_CHUNK_BYTES:
            chunks = split_audio(audio_path, work_dir)
        else:
            chunks = [audio_path]

        # Step 3: Transcribe each chunk
        transcript_parts = []
        for i, chunk in enumerate(chunks):
            text = await transcribe_chunk(chunk, req.language)
            transcript_parts.append(text)

        full_transcript = " ".join(transcript_parts).strip()
        return {
            "transcript": full_transcript,
            "chunks_processed": len(chunks),
            "characters": len(full_transcript),
        }

    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/translate")
async def translate(req: TranslateRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable not set")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    system_prompt = """Kamu adalah penerjemah profesional yang ahli menerjemahkan konten video ke Bahasa Indonesia.

Cara kerjamu:
- Terjemahkan dengan natural dan enak dibaca, seperti subtitle film berkualitas
- Jangan terlalu kaku atau formal — pakai bahasa yang mengalir dan mudah dipahami orang Indonesia
- Pertahankan semua istilah teknis, nama orang, nama produk, atau istilah khusus apa adanya (jangan diterjemahkan)
- Kalau ada humor atau idiom, adaptasikan ke konteks Indonesia yang natural — bukan terjemahan literal
- Jaga alur dan ritme kalimat tetap enak dibaca
- Output hanya teks terjemahan saja, tanpa penjelasan atau komentar tambahan"""

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": req.text},
                ],
                "temperature": 0.3,
                "max_tokens": 8192,
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Groq API error: {response.text}")

    data = response.json()
    translated = data["choices"][0]["message"]["content"].strip()
    return {"translated": translated}


def download_audio(url: str, work_dir: Path) -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(work_dir / "audio.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",  # Low quality = smaller file, fine for speech
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "cookiefile": None,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find output mp3
    for f in work_dir.iterdir():
        if f.suffix == ".mp3":
            return str(f)

    raise HTTPException(status_code=400, detail="Audio extraction failed — file not found after download")


def split_audio(audio_path: str, work_dir: Path) -> list:
    # Get total duration via ffprobe
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    duration = float(info["format"]["duration"])
    file_size = os.path.getsize(audio_path)

    num_chunks = math.ceil(file_size / MAX_CHUNK_BYTES)
    chunk_duration = duration / num_chunks

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_duration
        chunk_path = str(work_dir / f"chunk_{i:03d}.mp3")

        subprocess.run(
            [
                "ffmpeg", "-i", audio_path,
                "-ss", str(start),
                "-t", str(chunk_duration),
                "-q:a", "0", "-map", "a",
                chunk_path, "-y", "-loglevel", "quiet",
            ],
            check=True,
        )
        chunks.append(chunk_path)

    return chunks


async def transcribe_chunk(audio_path: str, language: str) -> str:
    async with httpx.AsyncClient(timeout=180) as client:
        with open(audio_path, "rb") as f:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (Path(audio_path).name, f, "audio/mpeg")},
                data={
                    "model": "whisper-large-v3",
                    "response_format": "text",
                    "language": language,
                },
            )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Groq API error: {response.text}")

    return response.text.strip()
