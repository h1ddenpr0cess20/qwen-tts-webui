# Troubleshooting

## Import errors: qwen-tts or torch

Symptoms:
- The server fails to start with an error mentioning `qwen-tts` or `torch`.

Fix:
- Install dependencies with `pip install -r requirements.txt`.
- Ensure your Python environment has a GPU-capable PyTorch build if you expect CUDA.

## Model downloads fail or are slow

Symptoms:
- First request takes a long time.
- Errors during model download.

Fix:
- Pre-download models and set `QWEN_TTS_MODEL` to a local path.
- Confirm your environment has network access for model downloads.

## MP3 conversion fails

Symptoms:
- `/api/convert` returns 500.
- UI shows "MP3 conversion failed."

Fix:
- Install `ffmpeg` and ensure it is on your PATH.
- Confirm `pydub` is installed (it is included in `requirements.txt`).

## Video export fails

Symptoms:
- `/api/video` returns 500.
- UI shows "Video export failed."

Fix:
- Install `ffmpeg` with `drawtext` enabled.
- Set `QWEN_TTS_VIDEO_FONT` to a font path if subtitles fail for CJK text.

## Voice Design request rejected

Symptoms:
- 422 error with a message about `instruct`.

Fix:
- Provide a persona description in the `instruct` field when using `voice_design`.

## Voice Clone requires transcript

Symptoms:
- 422 error asking for `ref_text`.

Fix:
- Provide `ref_text`, or enable `x_vector_only_mode` if you want to skip the transcript.

## Text too long

Symptoms:
- 400 error about text exceeding 500 characters.

Fix:
- Enable `chunk_text` to split long text automatically.

## Profile model mismatch

Symptoms:
- 400 error indicating profile was saved with a different model.

Fix:
- Use the model ID that created the profile, or re-save the profile with the desired model.

## Sample rate mismatch

Symptoms:
- 500 error: "Sample rate mismatch between chunks."

Fix:
- Re-run with smaller text or a different model. This is rare and often transient.
