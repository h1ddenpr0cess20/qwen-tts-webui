const byId = (id) => document.getElementById(id);

export const els = {
  form: byId("tts-form"),
  status: byId("status"),
  profileStatus: byId("profile-status"),
  player: byId("player"),
  downloadLink: byId("download-link"),
  languageSelect: byId("language"),
  speakerSelect: byId("speaker"),
  submitBtn: byId("submit-btn"),
  modelSelect: byId("model_id"),
  voiceProfileSelect: byId("voice_profile"),
  audioFormatSelect: byId("audio_format"),
  videoStyleSelect: byId("video_style"),
  videoLayoutSelect: byId("video_layout"),
  videoTranscriptCheckbox: byId("video_transcript"),
  exportVideoBtn: byId("export-video-btn"),
  videoDownloadLink: byId("video-download-link"),
  videoStatus: byId("video-status"),
  deleteProfileBtn: byId("delete-profile-btn"),
  exportProfileBtn: byId("export-profile-btn"),
  importProfileBtn: byId("import-profile-btn"),
  importProfileFile: byId("import-profile-file"),
  refText: byId("ref_text"),
  xVectorCheckbox: byId("x_vector_only_mode"),
  refAudioInput: byId("ref_audio"),
  clearRefBtn: byId("clear-ref-btn"),
  recordPill: byId("record-pill"),
  recordLabel: byId("record-label"),
  recordStatus: byId("record-status"),
  metaModel: byId("meta-model"),
  metaDevice: byId("meta-device"),
  fileInput: byId("ref_file"),
  saveProfileBtn: byId("save-profile-btn"),
  saveNameInput: byId("save_name"),
  textInput: byId("text"),
};

export const modeRadios = Array.from(document.querySelectorAll('input[name="mode"]'));
export const modeFields = Array.from(document.querySelectorAll(".mode-field"));

export function setStatus(message, isError = false) {
  if (!els.status) return;
  els.status.textContent = message || "";
  els.status.classList.toggle("status--error", !!isError);
}

export function setVideoStatus(message, isError = false) {
  if (!els.videoStatus) return;
  els.videoStatus.textContent = message || "";
  els.videoStatus.classList.toggle("status--error", !!isError);
}

export function setProfileStatus(message, isError = false) {
  if (!els.profileStatus) return;
  els.profileStatus.textContent = message || "";
  els.profileStatus.classList.toggle("status--error", !!isError);
}
