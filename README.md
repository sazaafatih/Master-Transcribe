# Video Transcriber API

Transcribe YouTube, Instagram Reels, and TikTok videos using Groq Whisper.

## Stack
- FastAPI + yt-dlp + FFmpeg
- Groq Whisper API (free)
- Deploy: Railway

## Deploy to Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select this repo
4. Add environment variable: `GROQ_API_KEY=your_key_here`
5. Deploy — Railway auto-detects the Dockerfile

## Get Groq API Key (Free)
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (no credit card needed)
3. Create API Key
4. Add to Railway environment variables

## API Endpoints

### POST /transcribe
```json
{
  "url": "https://www.youtube.com/watch?v=xxxxx",
  "language": "id"
}
```

Response:
```json
{
  "transcript": "...",
  "chunks_processed": 3,
  "characters": 2048
}
```

**Supported URLs**: YouTube, Instagram Reels, TikTok

**Language codes**: `id` (Indonesian), `en` (English), `auto` (auto-detect)
