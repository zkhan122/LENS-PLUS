import "./styles.css";
import { startCameraStream, stopStream } from "./media/cameraSource";
import { FileSource } from "./media/fileSource";
import type { InferenceMessage } from "./types/messages";
import { isInferenceMessage } from "./types/messages";
import { UiLogger } from "./ui/logger";
import { OverlayRenderer } from "./ui/overlay";
import { WebRtcClient } from "./webrtc/client";
import { getSignalingBaseUrl } from "./webrtc/signaling";

type SourceMode = "camera" | "file";

function mustQuery<T extends Element>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) {
    throw new Error(`Missing element: ${selector}`);
  }
  return node;
}

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("#app not found");
}

app.innerHTML = `
  <main class="layout">
    <header>
      <h1>LENS+ Stream Client</h1>
      <p>Basic phone streaming UI for WebRTC + server inference scaffolding.</p>
    </header>

    <section class="panel controls">
      <label>
        Source
        <select id="source-mode">
          <option value="camera">Phone Camera</option>
          <option value="file">Video File (dev)</option>
        </select>
      </label>

      <label id="file-input-wrap" class="hidden">
        Video file
        <input id="file-input" type="file" accept="video/*" />
      </label>

      <div class="button-row">
        <button id="start-source">Start Source</button>
        <button id="stop-source" class="ghost">Stop Source</button>
      </div>

      <div class="button-row">
        <button id="connect">Connect</button>
        <button id="disconnect" class="ghost">Disconnect</button>
      </div>

      <label class="toggle">
        <input id="tts-toggle" type="checkbox" />
        Speak guidance text
      </label>

      <label class="toggle">
        <input id="debug-toggle" type="checkbox" />
        Debug mode
      </label>

      <p>Status: <strong id="status">idle</strong></p>
    </section>

    <section class="panel video-panel">
      <div class="video-wrap">
        <video id="preview" autoplay muted playsinline></video>
        <canvas id="overlay"></canvas>
      </div>
    </section>

    <section class="panel log-panel">
      <h2>Events</h2>
      <ul id="event-log"></ul>
    </section>

    <section class="panel debug-panel hidden" id="debug-panel">
      <h2>Stream Proof</h2>
      <div class="button-row">
        <button id="debug-refresh">Refresh Sessions</button>
        <button id="debug-use-active" class="ghost">Use Active Session</button>
      </div>
      <label>
        Session
        <select id="debug-session-select"></select>
      </label>
      <p id="debug-meta">No session selected.</p>
      <img id="debug-snapshot" alt="Backend snapshot" />
    </section>
  </main>
`;

const sourceModeEl = mustQuery<HTMLSelectElement>("#source-mode");
const fileInputWrapEl = mustQuery<HTMLLabelElement>("#file-input-wrap");
const fileInputEl = mustQuery<HTMLInputElement>("#file-input");
const startSourceEl = mustQuery<HTMLButtonElement>("#start-source");
const stopSourceEl = mustQuery<HTMLButtonElement>("#stop-source");
const connectEl = mustQuery<HTMLButtonElement>("#connect");
const disconnectEl = mustQuery<HTMLButtonElement>("#disconnect");
const ttsToggleEl = mustQuery<HTMLInputElement>("#tts-toggle");
const debugToggleEl = mustQuery<HTMLInputElement>("#debug-toggle");
const statusEl = mustQuery<HTMLElement>("#status");
const previewEl = mustQuery<HTMLVideoElement>("#preview");
const overlayEl = mustQuery<HTMLCanvasElement>("#overlay");
const eventLogEl = mustQuery<HTMLUListElement>("#event-log");
const debugPanelEl = mustQuery<HTMLElement>("#debug-panel");
const debugRefreshEl = mustQuery<HTMLButtonElement>("#debug-refresh");
const debugUseActiveEl = mustQuery<HTMLButtonElement>("#debug-use-active");
const debugSessionSelectEl = mustQuery<HTMLSelectElement>("#debug-session-select");
const debugMetaEl = mustQuery<HTMLElement>("#debug-meta");
const debugSnapshotEl = mustQuery<HTMLImageElement>("#debug-snapshot");

let sourceMode: SourceMode = "camera";
let activeStream: MediaStream | null = null;
let activeSessionId: string | null = null;
const fileSource = new FileSource();
const logger = new UiLogger(eventLogEl);
const overlay = new OverlayRenderer(overlayEl);
const apiBaseUrl = getSignalingBaseUrl();

const webrtc = new WebRtcClient(
  (state) => {
    statusEl.textContent = state;
  },
  (payload) => {
    handleInferencePayload(payload);
  },
  (message) => {
    logger.log(message);
  }
);

previewEl.addEventListener("loadedmetadata", () => {
  overlay.syncWithVideo(previewEl);
});

window.addEventListener("resize", () => {
  overlay.syncWithVideo(previewEl);
});

sourceModeEl.addEventListener("change", () => {
  sourceMode = sourceModeEl.value as SourceMode;
  fileInputWrapEl.classList.toggle("hidden", sourceMode !== "file");
  logger.log(`Source mode set to ${sourceMode}`);
});

startSourceEl.addEventListener("click", async () => {
  try {
    await startSource();
  } catch (error) {
    logger.log(`Source start failed: ${String(error)}`);
  }
});

stopSourceEl.addEventListener("click", async () => {
  await stopAll();
  logger.log("Source stopped");
});

connectEl.addEventListener("click", async () => {
  if (!activeStream) {
    logger.log("Start a source before connecting");
    return;
  }

  try {
    statusEl.textContent = "connecting";
    await webrtc.connect(activeStream);
    activeSessionId = webrtc.getSessionId();
    if (activeSessionId) {
      await refreshDebugSessions(activeSessionId);
    }
  } catch (error) {
    statusEl.textContent = "failed";
    logger.log(`Connect failed: ${String(error)}`);
  }
});

disconnectEl.addEventListener("click", async () => {
  await webrtc.disconnect();
  activeSessionId = null;
  logger.log("Disconnected");
});

debugToggleEl.addEventListener("change", () => {
  const enabled = debugToggleEl.checked;
  debugPanelEl.classList.toggle("hidden", !enabled);
  if (enabled) {
    void refreshDebugSessions(activeSessionId);
  }
});

debugRefreshEl.addEventListener("click", async () => {
  await refreshDebugSessions(activeSessionId);
});

debugUseActiveEl.addEventListener("click", () => {
  if (!activeSessionId) {
    logger.log("No active session yet");
    return;
  }
  debugSessionSelectEl.value = activeSessionId;
  void updateSnapshotPreview(activeSessionId);
});

debugSessionSelectEl.addEventListener("change", () => {
  const selected = debugSessionSelectEl.value;
  if (!selected) {
    return;
  }
  void updateSnapshotPreview(selected);
});

window.setInterval(() => {
  if (!debugToggleEl.checked) {
    return;
  }
  void refreshDebugSessions(activeSessionId);
}, 2000);

async function startSource(): Promise<void> {
  await stopAll();

  if (sourceMode === "camera") {
    activeStream = await startCameraStream();
    logger.log("Camera stream started");
  } else {
    const selectedFile = fileInputEl.files?.[0];
    if (!selectedFile) {
      throw new Error("Pick a video file first");
    }
    activeStream = await fileSource.start(selectedFile);
    logger.log(`File stream started: ${selectedFile.name}`);
  }

  previewEl.srcObject = activeStream;
  await previewEl.play();
  overlay.syncWithVideo(previewEl);
  statusEl.textContent = "source_ready";
}

async function stopAll(): Promise<void> {
  await webrtc.disconnect();
  activeSessionId = null;
  stopStream(activeStream);
  fileSource.cleanup();
  activeStream = null;
  previewEl.srcObject = null;
  overlay.clear();
  statusEl.textContent = "idle";
}

type DebugSession = {
  session_id: string;
  total_frames: number;
  has_snapshot: boolean;
  latest_jpeg_at: string | null;
  connection_state: string;
  snapshot_errors?: number;
  last_snapshot_error?: string | null;
};

async function refreshDebugSessions(preferredSessionId: string | null): Promise<void> {
  try {
    const response = await fetch(`${apiBaseUrl}/debug/sessions`);
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }

    const payload = (await response.json()) as { sessions?: DebugSession[] };
    const sessions = payload.sessions ?? [];
    renderSessionSelect(sessions, preferredSessionId);

    const selected = debugSessionSelectEl.value;
    if (selected) {
      await updateSnapshotPreview(selected, sessions);
    } else {
      debugMetaEl.textContent = "No sessions available.";
      debugSnapshotEl.removeAttribute("src");
    }
  } catch (error) {
    logger.log(`Debug fetch failed: ${String(error)}`);
  }
}

function renderSessionSelect(
  sessions: DebugSession[],
  preferredSessionId: string | null
): void {
  const previous = debugSessionSelectEl.value;
  const preferred = preferredSessionId ?? previous;
  debugSessionSelectEl.innerHTML = "";

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = sessions.length ? "Select session" : "No sessions";
  debugSessionSelectEl.appendChild(placeholder);

  for (const session of sessions) {
    const option = document.createElement("option");
    option.value = session.session_id;
    option.textContent = `${session.session_id.slice(0, 8)}... (${session.total_frames} frames)`;
    debugSessionSelectEl.appendChild(option);
  }

  const match = sessions.find((session) => session.session_id === preferred);
  debugSessionSelectEl.value = match ? match.session_id : "";
}

async function updateSnapshotPreview(
  sessionId: string,
  sessions?: DebugSession[]
): Promise<void> {
  const session = sessions?.find((item) => item.session_id === sessionId);
  if (session) {
    debugMetaEl.textContent = [
      `State: ${session.connection_state}`,
      `frames: ${session.total_frames}`,
      `snapshot: ${session.has_snapshot ? "ready" : "not ready"}`,
      `errors: ${session.snapshot_errors ?? 0}`
    ].join(" | ");

    if (!session.has_snapshot) {
      debugSnapshotEl.removeAttribute("src");
      if (session.last_snapshot_error) {
        logger.log(`Snapshot warning: ${session.last_snapshot_error}`);
      }
      return;
    }
  }

  debugSnapshotEl.src = `${apiBaseUrl}/debug/sessions/${sessionId}/latest.jpg?t=${Date.now()}`;
}

function handleInferencePayload(rawPayload: string): void {
  logger.log(`Result: ${rawPayload}`);

  let parsed: unknown;
  try {
    parsed = JSON.parse(rawPayload);
  } catch {
    return;
  }

  if (!isInferenceMessage(parsed)) {
    return;
  }

  const message = parsed as InferenceMessage;
  if (message.objects) {
    overlay.drawBoxes(message.objects);
  } else {
    overlay.clear();
  }

  if (ttsToggleEl.checked && message.guidance_text) {
    const utterance = new SpeechSynthesisUtterance(message.guidance_text);
    utterance.rate = 1;
    speechSynthesis.cancel();
    speechSynthesis.speak(utterance);
  }
}
