const JSON_HEADERS = { "Content-Type": "application/json" };

async function ensureOk(res) {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res;
}

async function requestJson(url, options = {}) {
  const res = await ensureOk(await fetch(url, options));
  return res.json();
}

async function requestBlob(url, options = {}) {
  const res = await ensureOk(await fetch(url, options));
  return res.blob();
}

export function fetchMeta() {
  return requestJson("/api/meta");
}

export function fetchVoiceProfiles() {
  return requestJson("/api/voice_profiles");
}

export function synthesizeTts(payload) {
  return requestBlob("/api/tts", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export function convertToMp3(dataUrl) {
  return requestBlob("/api/convert", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ target_format: "mp3", data_url: dataUrl }),
  });
}

export function renderVideo(payload) {
  return requestBlob("/api/video", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export function createVoiceProfile(payload) {
  return requestJson("/api/voice_profiles", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export function deleteVoiceProfile(name) {
  return requestJson(`/api/voice_profiles/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export function exportVoiceProfile(name) {
  return requestBlob(`/api/voice_profiles/${encodeURIComponent(name)}/export`);
}
