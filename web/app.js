// Browser client: Silero VAD (vad-web) -> WebSocket -> gapless playback worklet, with barge-in.
"use strict";

const CFG = {
  wsUrl: `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
  playbackRate: 24000,          // AudioContext rate (Kokoro native); other rates are resampled
  minUtteranceMs: 300,          // ignore blips shorter than this
  vad: {                        // tune here if it cuts you off / waits too long
    positiveSpeechThreshold: 0.6,
    negativeSpeechThreshold: 0.4,
    redemptionMs: 600,          // silence needed to end an utterance (lower = snappier)
    minSpeechMs: 250,
    preSpeechPadMs: 300,
  },
  vadBase: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
  ortBase: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
};

const $ = (id) => document.getElementById(id);
const chat = $("chat"), statusEl = $("status"), statusText = $("statusText"), info = $("info");

let ws = null, ctx = null, player = null, micvad = null;
let turnActive = false, playing = false, awaitingAck = false;
let lastAudioTurn = null, tUtteranceEnd = 0, perceivedPending = false;
let pendingInterrupt = null, resolveInterrupt = null;
let header = null;                       // JSON header waiting for its binary frame
let sentenceText = new Map();            // index -> text for the current turn
let botBubble = null;

// ---------- UI helpers ----------
function setState(kind, text) { statusEl.className = `status ${kind}`; statusText.textContent = text; }
function addMsg(role, text) {
  const hint = chat.querySelector(".hint"); if (hint) hint.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`; el.textContent = text;
  chat.appendChild(el); chat.scrollTop = chat.scrollHeight;
  return el;
}
function append(el, text) { el.textContent += (el.textContent ? " " : "") + text; chat.scrollTop = chat.scrollHeight; }

const STAGES = [
  ["stt_ms", "STT"], ["llm_first_token_ms", "LLM first token"], ["first_sentence_ms", "First sentence"],
  ["first_audio_ms", "First audio (TTS)"], ["ttfa_ms", "Time to first audio (server)"],
  ["perceived_ms", "Perceived (VAD end → sound)"], ["total_ms", "Turn total"],
];
const history = Object.fromEntries(STAGES.map(([k]) => [k, []]));
function renderTimings(t) {
  for (const [k] of STAGES) if (t[k] != null) history[k].push(t[k]);
  const tb = document.querySelector("#timings tbody"); tb.innerHTML = "";
  for (const [k, label] of STAGES) {
    const arr = history[k]; if (!arr.length) continue;
    const avg = arr.reduce((a, b) => a + b, 0) / arr.length;
    const tr = document.createElement("tr");
    if (k === "ttfa_ms" || k === "perceived_ms") tr.className = "key";
    tr.innerHTML = `<td>${label}</td><td>${t[k] != null ? Math.round(t[k]) : "–"}</td><td>${Math.round(avg)}</td>`;
    tb.appendChild(tr);
  }
}

// ---------- audio helpers ----------
function int16ToFloat32(buf) {
  const i16 = new Int16Array(buf), f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
  return f32;
}
function float32ToInt16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) { const s = Math.max(-1, Math.min(1, f32[i])); out[i] = s < 0 ? s * 32768 : s * 32767; }
  return out;
}
function resample(f32, from, to) {
  if (from === to) return f32;
  const n = Math.round(f32.length * to / from), out = new Float32Array(n), ratio = (f32.length - 1) / Math.max(n - 1, 1);
  for (let i = 0; i < n; i++) { const p = i * ratio, i0 = Math.floor(p), i1 = Math.min(i0 + 1, f32.length - 1); out[i] = f32[i0] + (f32[i1] - f32[i0]) * (p - i0); }
  return out;
}

// ---------- barge-in ----------
function interrupt() {
  if (!(turnActive || playing) || awaitingAck) return pendingInterrupt;
  awaitingAck = true;
  pendingInterrupt = new Promise((res) => (resolveInterrupt = res));
  player.port.postMessage({ type: "flush" });     // stops sound instantly, replies with progress
  return pendingInterrupt;
}

function onPlayerMessage(e) {
  const m = e.data;
  if (m.type === "started") {
    playing = true; setState("speaking", "Speaking…");
    const txt = sentenceText.get(m.id);
    if (txt) { if (!botBubble) botBubble = addMsg("bot", ""); append(botBubble, txt); }
    if (m.id === 0 && perceivedPending) {
      perceivedPending = false;
      renderTimings({ perceived_ms: performance.now() - tUtteranceEnd });
    }
  } else if (m.type === "idle") {
    playing = false;
    if (!turnActive) setState("listening", "Listening…");
  } else if (m.type === "flushed") {
    ws.send(JSON.stringify({ type: "interrupt", turn: lastAudioTurn, sentence: m.id, fraction: m.fraction }));
    if (botBubble) botBubble.classList.add("cut");
    botBubble = null; playing = false; turnActive = false;
    setState("listening", "Listening…");
    if (resolveInterrupt) { resolveInterrupt(); resolveInterrupt = null; }
    pendingInterrupt = null;
  }
}

// ---------- websocket ----------
function onWsMessage(ev) {
  if (typeof ev.data !== "string") {            // binary audio frame for `header`
    if (!header || awaitingAck) { header = null; return; }
    let f32 = int16ToFloat32(ev.data);
    f32 = resample(f32, header.sample_rate, ctx.sampleRate);
    player.port.postMessage({ type: "push", buf: f32, id: header.index, turn: header.turn }, [f32.buffer]);
    header = null; return;
  }
  const m = JSON.parse(ev.data);
  switch (m.type) {
    case "ready":
      info.textContent = `Profile: ${m.profile} · STT ${m.stt} · LLM ${m.llm} · TTS ${m.tts}`; break;
    case "transcript":
      if (m.text) { addMsg("user", m.text).insertAdjacentHTML("beforeend", `<span class="meta">${m.language}</span>`); botBubble = null; sentenceText.clear(); }
      break;
    case "audio":
      if (!awaitingAck) { header = m; lastAudioTurn = m.turn; sentenceText.set(m.index, m.text); } else header = null;
      break;
    case "done":
      turnActive = false;
      if (m.timings) renderTimings(m.timings);
      if (!playing) setState("listening", "Listening…");
      break;
    case "interrupted": awaitingAck = false; turnActive = false; break;
    case "reset_ok": chat.innerHTML = '<p class="hint">New conversation started.</p>'; break;
    case "error": setState("error", "Error"); info.textContent = `Server error: ${m.message}`; turnActive = false; break;
  }
}

function connectWs() {
  return new Promise((resolve, reject) => {
    ws = new WebSocket(CFG.wsUrl); ws.binaryType = "arraybuffer";
    ws.onopen = resolve; ws.onerror = () => reject(new Error("WebSocket connection failed"));
    ws.onmessage = onWsMessage;
    ws.onclose = () => { if (micvad) stop(); };
  });
}

// ---------- start / stop ----------
async function start() {
  $("btnStart").disabled = true; setState("thinking", "Loading VAD…");
  try {
    ctx = new AudioContext({ sampleRate: CFG.playbackRate });
    await ctx.resume();
    await ctx.audioWorklet.addModule("worklet.js");
    player = new AudioWorkletNode(ctx, "player");
    player.connect(ctx.destination);
    player.port.onmessage = onPlayerMessage;

    await connectWs();

    micvad = await vad.MicVAD.new({
      ...CFG.vad,
      baseAssetPath: CFG.vadBase,
      onnxWASMBasePath: CFG.ortBase,
      onSpeechStart: () => { if (!turnActive && !playing) setState("speaking-user", "Hearing you…"); },
      onSpeechRealStart: () => { interrupt(); setState("speaking-user", "Hearing you…"); },   // barge-in
      onVADMisfire: () => { if (!turnActive && !playing) setState("listening", "Listening…"); },
      onSpeechEnd: async (audio) => {
        await interrupt();                                   // no-op unless something is active
        if (audio.length < CFG.minUtteranceMs * 16) { setState("listening", "Listening…"); return; }
        tUtteranceEnd = performance.now(); perceivedPending = true;
        turnActive = true; lastAudioTurn = null; setState("thinking", "Thinking…");
        ws.send(float32ToInt16(audio).buffer);               // 16 kHz mono int16
      },
    });
    micvad.start();
    setState("listening", "Listening…");
    $("btnStop").disabled = false; $("btnReset").disabled = false;
  } catch (err) {
    console.error(err); setState("error", "Error"); info.textContent = String(err.message || err);
    $("btnStart").disabled = false;
  }
}

function stop() {
  try { micvad && micvad.destroy(); } catch (_) {}
  micvad = null;
  try { ws && ws.close(); } catch (_) {}
  try { ctx && ctx.close(); } catch (_) {}
  turnActive = playing = awaitingAck = false;
  setState("idle", "Stopped");
  $("btnStart").disabled = false; $("btnStop").disabled = true; $("btnReset").disabled = true;
}

$("btnStart").onclick = start;
$("btnStop").onclick = stop;
$("btnReset").onclick = () => ws && ws.send(JSON.stringify({ type: "reset" }));
