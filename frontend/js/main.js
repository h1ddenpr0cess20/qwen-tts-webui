import {
  fetchMeta,
  fetchVoiceProfiles,
  synthesizeTts,
  convertToMp3,
  renderVideo,
  createVoiceProfile,
  deleteVoiceProfile,
  exportVoiceProfile,
  importVoiceProfile,
} from "./api.js";
import { blobToWavDataUrl, createRecorder, dataUrlFromBlob } from "./audio.js";
import { els, modeRadios, setProfileStatus, setStatus, setVideoStatus } from "./dom.js";
import { clearVideoUrl, resetAudioState, state } from "./state.js";
import {
  applyMeta,
  ensureModelOption,
  effectiveModel,
  populateModelSelect,
  populateVoiceProfiles,
  updateModeFields,
  updateModelMeta,
  updateRefTextState,
} from "./ui.js";

if (els.refAudioInput) {
  els.refAudioInput.dataset.usesStored = "false";
}

function clearRefAudioInputs() {
  state.storedRefAudioDataUrl = null;
  if (els.refAudioInput) {
    els.refAudioInput.value = "";
    els.refAudioInput.dataset.usesStored = "false";
    els.refAudioInput.title = "";
  }
  if (els.fileInput) {
    els.fileInput.value = "";
  }
  if (els.recordStatus) {
    els.recordStatus.textContent = "";
    els.recordStatus.title = "";
  }
}

async function loadVoiceProfiles() {
  try {
    const data = await fetchVoiceProfiles();
    state.voiceProfiles = data.profiles || [];
    populateVoiceProfiles();
    setProfileStatus(state.voiceProfiles.length ? "" : "No saved voices yet.");
  } catch (error) {
    console.error(error);
    setProfileStatus("Could not load saved voices.");
    state.voiceProfiles = [];
    populateVoiceProfiles();
  }
}

async function loadMeta() {
  try {
    const data = await fetchMeta();
    applyMeta(data);
    await loadVoiceProfiles();
  } catch (error) {
    console.error(error);
    setStatus("Could not load metadata. The API might be offline.", true);
  }
}

async function synthesize(e) {
  e.preventDefault();
  const formData = new FormData(els.form);
  const refAudioFieldValue = formData.get("ref_audio")?.trim();
  const refAudio =
    (els.refAudioInput?.dataset.usesStored === "true" && state.storedRefAudioDataUrl) ||
    refAudioFieldValue ||
    null;
  const refTextValue = formData.get("ref_text")?.trim();
  const payload = {
    mode: formData.get("mode") || "custom_voice",
    text: formData.get("text")?.trim(),
    chunk_text: formData.get("chunk_text") === "on",
    language: formData.get("language") || null,
    speaker: formData.get("speaker") || null,
    instruct: formData.get("instruct") || null,
    model_id: formData.get("model_id") || null,
    voice_profile: formData.get("voice_profile") || null,
    ref_audio: refAudio,
    ref_text: refTextValue || null,
    x_vector_only_mode: formData.get("x_vector_only_mode") === "on",
  };

  if (!payload.text) {
    setStatus("Please enter text.", true);
    return;
  }

  if (payload.mode === "voice_design" && !payload.instruct) {
    setStatus("Voice Design needs a style description.", true);
    return;
  }

  if (payload.mode === "voice_clone" && !payload.voice_profile) {
    if (!payload.ref_audio) {
      setStatus("Voice Clone needs a reference audio (record/upload/paste).", true);
      return;
    }
    if (!payload.x_vector_only_mode && !payload.ref_text) {
      setStatus("Provide the transcript for the reference audio (or enable x-vector only).", true);
      return;
    }
  }

  if (els.submitBtn) {
    els.submitBtn.disabled = true;
    els.submitBtn.textContent = "Generating...";
  }
  setStatus("Running synthesis... this can take a bit for large models.");
  if (els.downloadLink) {
    els.downloadLink.hidden = true;
  }
  resetAudioState();
  if (els.videoDownloadLink) {
    els.videoDownloadLink.hidden = true;
  }
  clearVideoUrl();
  setVideoStatus("");

  try {
    const blob = await synthesizeTts(payload);
    const url = URL.createObjectURL(blob);
    state.lastWavBlob = blob;
    state.lastUrls = { wav: url, mp3: null };
    if (els.player) {
      els.player.src = url;
      els.player.play().catch(() => {});
    }
    if (els.downloadLink) {
      els.downloadLink.href = url;
      els.downloadLink.download = "qwen3_tts.wav";
      els.downloadLink.hidden = false;
    }
    setStatus("Ready. Enjoy your audio!");
    const usedModel = effectiveModel(payload.mode, payload.model_id, payload.voice_profile);
    state.lastGenerationMeta = {
      mode: payload.mode,
      model_id: usedModel,
      voice_profile: payload.voice_profile,
    };
    updateModelMeta(payload.mode, usedModel, payload.voice_profile);
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Failed to synthesize.", true);
  } finally {
    if (els.submitBtn) {
      els.submitBtn.disabled = false;
      els.submitBtn.textContent = "Generate";
    }
  }
}

async function getDownloadUrlForFormat(fmt) {
  if (!state.lastWavBlob) {
    setStatus("Generate audio first.", true);
    return null;
  }
  if (fmt === "wav") {
    return state.lastUrls.wav;
  }
  if (fmt === "mp3") {
    if (state.lastUrls.mp3) return state.lastUrls.mp3;
    try {
      const dataUrl = await dataUrlFromBlob(state.lastWavBlob);
      const blob = await convertToMp3(dataUrl);
      const url = URL.createObjectURL(blob);
      state.lastUrls.mp3 = url;
      return url;
    } catch (error) {
      console.error(error);
      setStatus(error.message || "MP3 conversion failed.", true);
      return null;
    }
  }
  return state.lastUrls.wav;
}

async function exportVideo() {
  if (!state.lastWavBlob) {
    setVideoStatus("Generate audio first.", true);
    return;
  }
  if (!els.exportVideoBtn) return;
  els.exportVideoBtn.disabled = true;
  const originalLabel = els.exportVideoBtn.textContent;
  els.exportVideoBtn.textContent = "Rendering...";
  setVideoStatus("Rendering video...");
  if (els.videoDownloadLink) {
    els.videoDownloadLink.hidden = true;
  }
  try {
    const transcript =
      els.videoTranscriptCheckbox?.checked ? (els.textInput?.value || "").trim() : "";
    const dataUrl = await dataUrlFromBlob(state.lastWavBlob);
    const payload = {
      data_url: dataUrl,
      transcript: transcript || null,
      style: els.videoStyleSelect?.value || "waveform",
      layout: els.videoLayoutSelect?.value || "vertical",
    };
    const blob = await renderVideo(payload);
    clearVideoUrl();
    const url = URL.createObjectURL(blob);
    state.lastVideoUrl = url;
    if (els.videoDownloadLink) {
      els.videoDownloadLink.href = url;
      const layout = els.videoLayoutSelect?.value || "vertical";
      els.videoDownloadLink.download = `qwen3_tts_${layout}.mp4`;
      els.videoDownloadLink.hidden = false;
    }
    setVideoStatus("Video ready.");
  } catch (error) {
    console.error(error);
    setVideoStatus(error.message || "Video export failed.", true);
  } finally {
    els.exportVideoBtn.disabled = false;
    els.exportVideoBtn.textContent = originalLabel;
  }
}

const recorder = createRecorder({
  onStart: () => {
    if (els.recordLabel) els.recordLabel.textContent = "⏹️ Stop recording";
    if (els.recordStatus) els.recordStatus.textContent = "Recording…";
  },
  onStop: () => {
    if (els.recordLabel) els.recordLabel.textContent = "🎙️ Start recording";
  },
  onDataUrl: (wavUrl) => {
    state.storedRefAudioDataUrl = wavUrl;
    if (els.refAudioInput) {
      els.refAudioInput.value = "Recorded clip";
      els.refAudioInput.dataset.usesStored = "true";
      els.refAudioInput.title = "Recorded clip";
    }
    if (els.recordStatus) {
      els.recordStatus.textContent = "Recorded clip attached.";
    }
  },
  onError: ({ message, error }) => {
    if (els.recordStatus) els.recordStatus.textContent = message;
    if (error) console.error(error);
  },
});

modeRadios.forEach((radio) => radio.addEventListener("change", updateModeFields));
els.modelSelect?.addEventListener("change", () => updateModelMeta());
els.voiceProfileSelect?.addEventListener("change", () => {
  const selectedProfile = els.voiceProfileSelect.value;
  const profile = state.voiceProfiles.find((p) => p.name === selectedProfile);
  populateModelSelect();
  if (profile?.model_id) {
    ensureModelOption(profile.model_id);
    els.modelSelect.value = profile.model_id;
  }
  updateModelMeta();
  updateRefTextState();
});
els.form?.addEventListener("submit", synthesize);
els.refAudioInput?.addEventListener("input", () => {
  els.refAudioInput.dataset.usesStored = "false";
  state.storedRefAudioDataUrl = null;
  if (els.recordStatus) {
    els.recordStatus.textContent = "";
    els.recordStatus.title = "";
  }
});
els.fileInput?.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const wavUrl = await blobToWavDataUrl(file);
    state.storedRefAudioDataUrl = wavUrl;
    if (els.refAudioInput) {
      els.refAudioInput.value = file.name;
      els.refAudioInput.dataset.usesStored = "true";
      els.refAudioInput.title = file.name;
    }
    if (els.recordStatus) {
      els.recordStatus.textContent = file.name;
      els.recordStatus.title = file.name;
    }
  } catch (error) {
    if (els.recordStatus) els.recordStatus.textContent = "Failed to process audio file.";
    console.error(error);
  }
});

els.recordPill?.addEventListener("click", () => {
  if (recorder.isRecording()) {
    recorder.stop();
  } else {
    recorder.start();
  }
});
els.clearRefBtn?.addEventListener("click", () => {
  clearRefAudioInputs();
});

els.downloadLink?.addEventListener("click", async (e) => {
  e.preventDefault();
  const fmt = els.audioFormatSelect?.value || "wav";
  const url = await getDownloadUrlForFormat(fmt);
  if (!url) return;
  const a = document.createElement("a");
  a.href = url;
  a.download = `qwen3_tts.${fmt}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
});

els.audioFormatSelect?.addEventListener("change", async (e) => {
  const fmt = e.target.value || "wav";
  if (!state.lastWavBlob) return;
  const url = await getDownloadUrlForFormat(fmt);
  if (url && els.downloadLink) {
    els.downloadLink.href = url;
    els.downloadLink.download = `qwen3_tts.${fmt}`;
    els.downloadLink.hidden = false;
    setStatus(`Ready: ${fmt.toUpperCase()} available.`);
    if (state.lastGenerationMeta) {
      updateModelMeta(
        state.lastGenerationMeta.mode,
        state.lastGenerationMeta.model_id,
        state.lastGenerationMeta.voice_profile
      );
    } else {
      updateModelMeta();
    }
  }
});
els.exportVideoBtn?.addEventListener("click", exportVideo);

els.saveProfileBtn?.addEventListener("click", async () => {
  const name = els.saveNameInput?.value.trim() || "";
  const refAudio =
    (els.refAudioInput?.dataset.usesStored === "true" && state.storedRefAudioDataUrl) ||
    els.refAudioInput?.value.trim();
  const refText = els.refText?.value.trim();
  const xvecOnly = !!els.xVectorCheckbox?.checked;
  const modelId = els.modelSelect?.value || null;
  setProfileStatus("");

  if (!name) {
    setProfileStatus("Enter a profile name.");
    return;
  }
  if (!refAudio) {
    setProfileStatus("Attach reference audio first.");
    return;
  }
  if (!xvecOnly && !refText) {
    setProfileStatus("Add transcript or enable x-vector only.");
    return;
  }

  if (els.saveProfileBtn) {
    els.saveProfileBtn.disabled = true;
    els.saveProfileBtn.textContent = "Saving...";
  }
  try {
    const data = await createVoiceProfile({
      name,
      ref_audio: refAudio,
      ref_text: refText,
      x_vector_only_mode: xvecOnly,
      model_id: modelId,
    });
    setProfileStatus(`Saved voice profile: ${data.original_name || data.name}.`);
    if (els.saveNameInput) els.saveNameInput.value = "";
    await loadVoiceProfiles();
    if (els.voiceProfileSelect) {
      els.voiceProfileSelect.value = data.name;
    }
    updateRefTextState();
  } catch (error) {
    console.error(error);
    setProfileStatus(error.message || "Failed to save profile.");
  } finally {
    if (els.saveProfileBtn) {
      els.saveProfileBtn.disabled = false;
      els.saveProfileBtn.textContent = "Save profile";
    }
  }
});

els.exportProfileBtn?.addEventListener("click", async () => {
  const name = els.voiceProfileSelect?.value;
  setProfileStatus("");
  if (!name) {
    setProfileStatus("Select a profile to export.");
    return;
  }
  els.exportProfileBtn.disabled = true;
  const originalLabel = els.exportProfileBtn.textContent;
  els.exportProfileBtn.textContent = "Exporting...";
  try {
    const blob = await exportVoiceProfile(name);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.pt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setProfileStatus(`Exported voice profile: ${name}.`);
  } catch (error) {
    console.error(error);
    setProfileStatus(error.message || "Failed to export profile.");
  } finally {
    els.exportProfileBtn.disabled = false;
    els.exportProfileBtn.textContent = originalLabel;
  }
});

els.importProfileBtn?.addEventListener("click", () => {
  els.importProfileFile?.click();
});

els.importProfileFile?.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  setProfileStatus("");
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pt")) {
    setProfileStatus("Choose a .pt profile file to import.", true);
    e.target.value = "";
    return;
  }
  const originalLabel = els.importProfileBtn?.textContent;
  if (els.importProfileBtn) {
    els.importProfileBtn.disabled = true;
    els.importProfileBtn.textContent = "Importing...";
  }
  try {
    const data = await importVoiceProfile(file);
    setProfileStatus(`Imported voice profile: ${data.original_name || data.name}.`);
    await loadVoiceProfiles();
    if (els.voiceProfileSelect) {
      els.voiceProfileSelect.value = data.name;
    }
    updateRefTextState();
  } catch (error) {
    console.error(error);
    setProfileStatus(error.message || "Failed to import profile.", true);
  } finally {
    if (els.importProfileBtn) {
      els.importProfileBtn.disabled = false;
      els.importProfileBtn.textContent = originalLabel || "Import .pt";
    }
    e.target.value = "";
  }
});

els.deleteProfileBtn?.addEventListener("click", async () => {
  const name = els.voiceProfileSelect?.value;
  setProfileStatus("");
  if (!name) {
    setProfileStatus("Select a profile to delete.");
    return;
  }
  if (!confirm(`Delete voice profile "${name}"?`)) {
    return;
  }
  els.deleteProfileBtn.disabled = true;
  els.deleteProfileBtn.textContent = "Deleting...";
  try {
    await deleteVoiceProfile(name);
    setProfileStatus(`Deleted voice profile: ${name}.`);
    await loadVoiceProfiles();
    if (els.voiceProfileSelect) {
      els.voiceProfileSelect.value = "";
    }
    updateRefTextState();
  } catch (error) {
    console.error(error);
    setProfileStatus(error.message || "Failed to delete profile.");
  } finally {
    els.deleteProfileBtn.disabled = false;
    els.deleteProfileBtn.textContent = "Delete profile";
  }
});

updateModeFields();
updateRefTextState();
loadMeta();
