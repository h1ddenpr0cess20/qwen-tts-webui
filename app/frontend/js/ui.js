import { els, modeFields, modeRadios } from "./dom.js";
import { state } from "./state.js";

const MODEL_OPTIONS = {
  custom_voice: [
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
  ],
  voice_design: ["Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
  voice_clone: ["Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"],
};

const findProfileByName = (name) => state.voiceProfiles.find((p) => p.name === name);

export function currentMode() {
  const checked = modeRadios.find((r) => r.checked);
  return checked ? checked.value : "custom_voice";
}

export function ensureModelOption(value) {
  if (!value || !els.modelSelect) return;
  const exists = Array.from(els.modelSelect.options).some((opt) => opt.value === value);
  if (!exists) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    els.modelSelect.appendChild(opt);
  }
}

export function effectiveModel(modeOverride = null, modelIdOverride = null, profileNameOverride = null) {
  const mode = modeOverride || currentMode();
  const profileName = profileNameOverride ?? (els.voiceProfileSelect?.value || "");
  const selectedModel = modelIdOverride ?? (els.modelSelect?.value || "");
  if (modelIdOverride) return modelIdOverride;
  if (mode === "voice_clone" && profileName) {
    const profile = findProfileByName(profileName);
    if (profile?.model_id) return profile.model_id;
    const fallback = state.defaultModels.voice_clone || state.defaultModels.custom_voice || "";
    return fallback ? `${fallback} (profile: ${profileName})` : `${profileName} (profile)`;
  }
  if (selectedModel) return selectedModel;
  if (mode && state.defaultModels[mode]) return state.defaultModels[mode];
  const recommended = MODEL_OPTIONS[mode]?.[0];
  if (recommended) return recommended;
  return state.defaultModels.custom_voice || "-";
}

export function updateModelMeta(modeOverride = null, modelIdOverride = null, profileNameOverride = null) {
  if (!els.metaModel) return;
  els.metaModel.textContent = effectiveModel(modeOverride, modelIdOverride, profileNameOverride);
}

export function updateRefTextState() {
  const hasProfile = !!els.voiceProfileSelect?.value;
  if (els.refText) {
    els.refText.disabled = hasProfile;
  }
  if (els.xVectorCheckbox) {
    els.xVectorCheckbox.disabled = hasProfile;
    if (hasProfile) {
      els.xVectorCheckbox.checked = false;
    }
  }
}

export function populateModelSelect() {
  if (!els.modelSelect) return;
  const mode = currentMode();
  const recommended = MODEL_OPTIONS[mode] || [];
  const seen = new Set();
  els.modelSelect.innerHTML = "";
  const addOption = (value) => {
    if (!value || seen.has(value)) return;
    seen.add(value);
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    els.modelSelect.appendChild(opt);
  };
  recommended.forEach(addOption);
  addOption(state.defaultModels[mode]);
  addOption(state.defaultModels.custom_voice);
  if (!seen.size) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Use server default";
    els.modelSelect.appendChild(opt);
    els.modelSelect.value = "";
  } else {
    const preferDefault =
      (state.defaultModels[mode] && seen.has(state.defaultModels[mode])) ||
      (!recommended.length &&
        state.defaultModels.custom_voice &&
        seen.has(state.defaultModels.custom_voice));
    const preferred = preferDefault
      ? state.defaultModels[mode] || state.defaultModels.custom_voice
      : recommended.find((v) => seen.has(v)) || Array.from(seen)[0];
    els.modelSelect.value = preferred || "";
  }
  updateModelMeta();
}

export function updateModeFields() {
  const mode = currentMode();
  modeFields.forEach((el) => {
    const modes = (el.dataset.modes || "").split(",").map((s) => s.trim());
    el.classList.toggle("hidden", !modes.includes(mode));
  });
  populateModelSelect();
}

export function applyMeta(data) {
  state.defaultModels = data.defaults || {};
  if (els.metaModel) {
    els.metaModel.textContent = state.defaultModels.custom_voice || data.model_id || "-";
  }
  if (els.metaDevice) {
    els.metaDevice.textContent = data.device || "-";
    els.metaDevice.setAttribute("title", data.voices_dir || "");
  }
  populateModelSelect();

  if (els.speakerSelect) {
    els.speakerSelect.innerHTML = '<option value="">Pick a speaker</option>';
    (data.speakers || []).forEach((speaker) => {
      const opt = document.createElement("option");
      opt.value = speaker;
      opt.textContent = speaker;
      els.speakerSelect.appendChild(opt);
    });
  }

  if (els.languageSelect) {
    els.languageSelect.innerHTML = '<option value="">Auto detect</option>';
    (data.languages || []).forEach((lang) => {
      const opt = document.createElement("option");
      opt.value = lang;
      opt.textContent = lang;
      els.languageSelect.appendChild(opt);
    });
  }
}

export function populateVoiceProfiles() {
  if (!els.voiceProfileSelect) return;
  const previous = els.voiceProfileSelect.value;
  els.voiceProfileSelect.innerHTML = '<option value="">None</option>';
  state.voiceProfiles.forEach((profile) => {
    const opt = document.createElement("option");
    opt.value = profile.name;
    const label = profile.original_name || profile.name;
    opt.textContent = label + (profile.model_id ? ` (${profile.model_id})` : "");
    els.voiceProfileSelect.appendChild(opt);
  });
  if (previous && state.voiceProfiles.some((p) => p.name === previous)) {
    els.voiceProfileSelect.value = previous;
  }
  updateModelMeta();
  updateRefTextState();
  if (els.deleteProfileBtn) {
    els.deleteProfileBtn.disabled = !state.voiceProfiles.length;
  }
  if (els.exportProfileBtn) {
    els.exportProfileBtn.disabled = !state.voiceProfiles.length;
  }
}
