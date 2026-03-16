const audioCtx = typeof AudioContext !== "undefined" ? new AudioContext() : null;

export async function blobToWavDataUrl(blob) {
  if (!audioCtx) throw new Error("AudioContext not supported.");
  const arrayBuf = await blob.arrayBuffer();
  const audioBuf = await audioCtx.decodeAudioData(arrayBuf);
  const numCh = audioBuf.numberOfChannels;
  const sampleRate = audioBuf.sampleRate;
  const length = audioBuf.length;
  const interleaved = new Float32Array(length * numCh);
  for (let ch = 0; ch < numCh; ch++) {
    const channelData = audioBuf.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      interleaved[i * numCh + ch] = channelData[i];
    }
  }
  const wavBuffer = new ArrayBuffer(44 + interleaved.length * 2);
  const view = new DataView(wavBuffer);
  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + interleaved.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numCh, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numCh * 2, true);
  view.setUint16(32, numCh * 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, interleaved.length * 2, true);
  let offset = 44;
  for (let i = 0; i < interleaved.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, interleaved[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  const wavBlob = new Blob([view], { type: "audio/wav" });
  return await dataUrlFromBlob(wavBlob);
}

export async function dataUrlFromBlob(blob) {
  return await new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsDataURL(blob);
  });
}

export function createRecorder({ onStart, onStop, onDataUrl, onError } = {}) {
  let mediaRecorder = null;
  let recordingChunks = [];
  let stream = null;

  const safeCall = (handler, payload) => {
    if (typeof handler === "function") handler(payload);
  };

  async function start() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordingChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordingChunks.push(e.data);
      };
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordingChunks, { type: "audio/webm" });
        blobToWavDataUrl(blob)
          .then((wavUrl) => safeCall(onDataUrl, wavUrl))
          .catch((error) =>
            safeCall(onError, { message: "Could not decode recording.", error })
          );
        stream?.getTracks().forEach((t) => t.stop());
        safeCall(onStop);
      };
      mediaRecorder.start();
      safeCall(onStart);
    } catch (error) {
      safeCall(onError, { message: "Mic access failed.", error });
    }
  }

  function stop() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
  }

  function isRecording() {
    return mediaRecorder && mediaRecorder.state === "recording";
  }

  return { start, stop, isRecording };
}
