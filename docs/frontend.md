# Frontend

The UI is a static HTML/JS app served by FastAPI from `frontend/`. It uses no build step and interacts with the API directly.

## Key files

- `frontend/index.html` - Markup and structure.
- `frontend/js/main.js` - Main UI logic and event handlers.
- `frontend/js/api.js` - API client wrapper.
- `frontend/js/audio.js` - Recording and WAV conversion helpers.
- `frontend/js/ui.js` - UI state updates and model selection logic.
- `frontend/css/*` - Styling.

## UI behavior overview

1. On page load, the UI calls `/api/meta` to fetch:
   - default model IDs
   - supported speakers and languages
   - device info
2. Voice profiles are loaded from `/api/voice_profiles`.
3. The model drop-down is populated per mode, plus server defaults.
4. Generate triggers `/api/tts` and streams a WAV response into the player.
5. MP3 conversion calls `/api/convert` with a WAV data URL.
6. Video export calls `/api/video` with a WAV data URL and optional transcript.

## Recording and uploads

- File uploads are decoded in the browser and converted to WAV data URLs.
- Browser recording uses `MediaRecorder`, stores audio as `webm`, and converts to WAV.
- The UI passes a data URL to `/api/tts` for `ref_audio`, which the backend decodes.

## Voice profile tools

When `voice_profile` is selected:

- The UI disables the transcript and x-vector options (the profile already includes a prompt).
- The UI pins the model selector to the profile's model when available.

## Offline mode

`frontend/index.html` pulls a Google Font. Remove the `<link>` tags if you need strictly offline assets.
