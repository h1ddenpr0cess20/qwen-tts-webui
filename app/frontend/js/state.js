export const state = {
  defaultModels: {},
  voiceProfiles: [],
  lastWavBlob: null,
  lastUrls: { wav: null, mp3: null },
  lastVideoUrl: null,
  lastGenerationMeta: null,
  storedRefAudioDataUrl: null,
};

export function resetAudioState() {
  state.lastWavBlob = null;
  state.lastUrls = { wav: null, mp3: null };
}

export function clearVideoUrl() {
  if (state.lastVideoUrl) {
    URL.revokeObjectURL(state.lastVideoUrl);
    state.lastVideoUrl = null;
  }
}
