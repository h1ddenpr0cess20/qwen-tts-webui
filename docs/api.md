# API Reference

Base URL: `http://localhost:8000`

All endpoints are JSON unless noted otherwise. Audio/video endpoints return binary blobs.

## POST /api/tts

Synthesize speech. Returns a WAV stream (`audio/wav`).

### Request body

```json
{
  "mode": "custom_voice",
  "text": "Hello world.",
  "chunk_text": true,
  "language": "English",
  "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "device": "cuda:0",
  "speaker": "Ryan",
  "instruct": "Warm, friendly, clear diction.",
  "ref_audio": "https://example.com/voice.wav",
  "ref_text": "Reference transcript.",
  "x_vector_only_mode": false,
  "voice_profile": "my_saved_voice"
}
```

### Notes by mode

- `custom_voice`
  - Uses `speaker` and optional `instruct`.
  - If `speaker` is missing, the server picks the first available speaker.
- `voice_design`
  - Requires `instruct`. Missing `instruct` returns HTTP 422.
- `voice_clone`
  - Requires `ref_audio` and `ref_text`, unless `x_vector_only_mode` is true.
  - If `voice_profile` is provided, the server ignores `ref_audio`/`ref_text` and uses the cached prompt.
  - When using `voice_profile`, the model ID must match the one the profile was created with.

### Validation and error behavior

- Text length:
  - If `chunk_text` is false and `text` exceeds 500 characters, the request fails with HTTP 400.
  - If `chunk_text` is true, the server splits text into 500-character chunks without breaking words.
- Sample rate mismatch across chunks causes HTTP 500.
- Unsupported mode returns HTTP 400.

## GET /api/meta

Returns default model IDs, device, supported speakers/languages, and the voice profile directory.

Example response:

```json
{
  "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "device": "cuda:0",
  "defaults": {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "voice_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
  },
  "speakers": ["..."],
  "languages": ["..."],
  "voices_dir": "C:\\path\\to\\voices"
}
```

## POST /api/convert

Convert a WAV data URL to MP3. Returns an MP3 stream (`audio/mpeg`).

### Request body

```json
{
  "target_format": "mp3",
  "data_url": "data:audio/wav;base64,...."
}
```

### Errors

- Returns HTTP 500 if `pydub` or `ffmpeg` is missing.
- Returns HTTP 415 if the data URL is invalid.

## POST /api/video

Renders an MP4 video with waveform/spectrum/pulse visuals and optional transcript overlay.

### Request body

```json
{
  "data_url": "data:audio/wav;base64,....",
  "transcript": "Optional transcript text",
  "style": "waveform",
  "layout": "vertical"
}
```

### Supported values

- `style`: `waveform`, `spectrum`, `pulse`
- `layout`: `vertical`, `square`, `landscape`

### Errors

- HTTP 415 for invalid data URL.
- HTTP 400 for unsupported layout.
- HTTP 500 if `ffmpeg` is missing or drawtext is unavailable.

## GET /api/voice_profiles

List saved voice profiles.

Example response:

```json
{
  "profiles": [
    {
      "name": "my_voice",
      "original_name": "My Voice",
      "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
      "saved_at": 1738030000.0
    }
  ]
}
```

## POST /api/voice_profiles

Create a new voice profile for reuse.

### Request body

```json
{
  "name": "my_voice",
  "ref_audio": "https://example.com/voice.wav",
  "ref_text": "Transcript of ref_audio",
  "x_vector_only_mode": false,
  "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "device": "cuda:0"
}
```

### Behavior

- `name` is sanitized for filesystem safety. Invalid names return HTTP 400.
- If `x_vector_only_mode` is false, `ref_text` is required.
- The created profile is stored as a `.pt` file plus a `.meta.json` file.

## GET /api/voice_profiles/{name}/export

Downloads the raw `.pt` profile file for backup or transfer.

## DELETE /api/voice_profiles/{name}

Deletes the profile `.pt` and metadata file.

## GET /api/health

Simple health check.

Example response:

```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "device": "cuda:0"
}
```
