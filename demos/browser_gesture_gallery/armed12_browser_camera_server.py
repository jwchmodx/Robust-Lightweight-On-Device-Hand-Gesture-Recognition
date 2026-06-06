from __future__ import annotations

import argparse
import json
import ssl
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import torch

from armed12_gallery_demo import (
    DEMO_ROOT,
    DEFAULT_MODEL,
    DEFAULT_STATIC_LABELS,
    DEFAULT_STATIC_MEAN,
    DEFAULT_STATIC_MODEL,
    DEFAULT_STATIC_STD,
    DISLIKE_CLASS,
    DOWN_CLASS,
    FIST_CLASS,
    LEFT_CLASS,
    LIKE_CLASS,
    PALM_CLASS,
    RIGHT_CLASS,
    UP_CLASS,
    Prediction,
    StaticPrediction,
    load_model,
    load_static_model,
    predict,
    predict_static,
    prepare_recorded_sequence,
)


HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Armed Gesture Gallery</title>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
  <style>
    :root { --text:#f7f7f4; --muted:rgba(247,247,244,.62); --line:rgba(255,255,255,.16); --green:#68f28e; --amber:#ffd166; }
    * { box-sizing:border-box; }
    html,body { margin:0; width:100%; height:100%; overflow:hidden; color:var(--text); font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#070809; }
    .stage { position:relative; width:100vw; height:100vh; overflow:hidden; background:#050505; }
    .gallery { position:absolute; inset:0; perspective:1200px; background:#050505; }
    .photo { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; background:#070707; transform-origin:center; }
    .photo.enter-right { animation: enterRight .55s cubic-bezier(.18,.72,.22,1); }
    .photo.enter-left { animation: enterLeft .55s cubic-bezier(.18,.72,.22,1); }
    .photo.enter-up { animation: enterUp .55s cubic-bezier(.18,.72,.22,1); }
    .photo.enter-down { animation: enterDown .55s cubic-bezier(.18,.72,.22,1); }
    @keyframes enterRight { from { transform:translateX(42%) rotateY(-38deg) scale(.84); opacity:.35; } to { transform:none; opacity:1; } }
    @keyframes enterLeft { from { transform:translateX(-42%) rotateY(38deg) scale(.84); opacity:.35; } to { transform:none; opacity:1; } }
    @keyframes enterUp { from { transform:translateY(-42%) rotateX(-38deg) scale(.84); opacity:.35; } to { transform:none; opacity:1; } }
    @keyframes enterDown { from { transform:translateY(42%) rotateX(38deg) scale(.84); opacity:.35; } to { transform:none; opacity:1; } }
    video { display:none; }
    canvas.preview { position:absolute; right:18px; bottom:18px; width:min(260px,24vw); aspect-ratio:4/3; border:1px solid var(--line); border-radius:14px; background:#111; box-shadow:0 18px 44px rgba(0,0,0,.35); transform:scaleX(-1); }
    .topbar { position:absolute; top:18px; left:18px; right:18px; display:flex; justify-content:space-between; gap:14px; pointer-events:none; }
    .guide,.debug { border:1px solid var(--line); background:rgba(12,14,18,.66); backdrop-filter:blur(18px) saturate(130%); box-shadow:0 18px 40px rgba(0,0,0,.28); }
    .guide { min-width:min(560px,calc(100vw - 210px)); padding:16px 18px; border-radius:16px; }
    .eyebrow { margin-bottom:6px; color:var(--muted); font-size:12px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
    .instruction { font-size:clamp(24px,3.2vw,48px); line-height:1.05; font-weight:850; }
    .sub { margin-top:8px; color:var(--muted); font-size:14px; }
    .mode { min-width:150px; padding:12px 14px; border:1px solid rgba(255,255,255,.18); border-radius:999px; background:rgba(255,255,255,.08); text-align:center; font-size:13px; font-weight:900; }
    .mode.on { color:#061008; background:linear-gradient(135deg,var(--green),#d4ff7e); box-shadow:0 0 34px rgba(104,242,142,.34); }
    .progress { position:absolute; left:18px; right:18px; bottom:66px; height:5px; border-radius:999px; background:rgba(255,255,255,.13); overflow:hidden; }
    .progress>div { height:100%; width:0%; border-radius:inherit; background:linear-gradient(90deg,var(--green),var(--amber)); transition:width 120ms linear; }
    .debug { position:absolute; left:18px; right:300px; bottom:16px; display:flex; justify-content:space-between; gap:12px; padding:10px 12px; border-radius:12px; color:var(--muted); font-size:12px; }
    .effects { position:absolute; inset:0; overflow:hidden; pointer-events:none; }
    .emoji { position:absolute; left:50%; font-family:"Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif; font-size:clamp(52px,8vw,110px); filter:drop-shadow(0 18px 24px rgba(0,0,0,.34)); animation:floatEmoji 1000ms cubic-bezier(.18,.72,.22,1) forwards; }
    @keyframes floatEmoji { from { opacity:0; transform:translate(-50%,20px) scale(.72) rotate(-8deg); } 16% { opacity:1; } to { opacity:0; transform:translate(calc(-50% + var(--x)),var(--travel)) scale(1.25) rotate(var(--r)); } }
    .hello { position:absolute; left:50%; top:50%; padding:18px 26px; border:1px solid rgba(255,255,255,.28); border-radius:18px; background:rgba(8,10,14,.72); backdrop-filter:blur(18px) saturate(140%); font-size:clamp(36px,7vw,92px); font-weight:850; box-shadow:0 22px 70px rgba(0,0,0,.38); animation:helloPop 1200ms cubic-bezier(.18,.72,.22,1) forwards; }
    @keyframes helloPop { from { opacity:0; transform:translate(-50%,-42%) scale(.82); } 14%,72% { opacity:1; transform:translate(-50%,-50%) scale(1); } to { opacity:0; transform:translate(-50%,-58%) scale(1.06); } }
  </style>
</head>
<body>
  <main class="stage">
    <div class="gallery"><img id="photo" class="photo" alt="gallery" /></div>
    <div class="topbar">
      <div class="guide"><div class="eyebrow">Gesture Gallery</div><div id="instruction" class="instruction">카메라 준비 중</div><div id="status" class="sub">starting</div></div>
      <div id="mode" class="mode">STARTING</div>
    </div>
    <div class="progress"><div id="progress"></div></div>
    <div class="debug"><span id="static">static - 0.00</span><span id="swipe">swipe - 0.00</span></div>
    <canvas id="preview" class="preview"></canvas>
    <video id="video" playsinline muted></video>
    <div id="effects" class="effects"></div>
  </main>
  <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('preview');
    const ctx = canvas.getContext('2d');
    const photo = document.getElementById('photo');
    const instructionEl = document.getElementById('instruction');
    const statusEl = document.getElementById('status');
    const modeEl = document.getElementById('mode');
    const staticEl = document.getElementById('static');
    const swipeEl = document.getElementById('swipe');
    const progressEl = document.getElementById('progress');
    const effectsEl = document.getElementById('effects');
    let gallery = [];
    let lastIndex = -1;
    let lastLikeSeq = 0, lastDislikeSeq = 0, lastHelloSeq = 0;
    let inferenceInFlight = false;
    let lastInferenceAt = 0;
    const inferenceIntervalMs = 100;

    function guideFor(mode) {
      if (mode === 'IDLE') return '주먹을 쥐어 시작';
      if (mode === 'ARMED' || mode === 'RECORDING') return '지금 움직이세요';
      if (mode === 'ERROR') return '카메라 오류';
      return '준비 중';
    }
    function burstReaction(emoji, direction) {
      const items = [['-34vw','-12px','-14deg'],['-18vw','-54px','10deg'],['0vw','-18px','-6deg'],['17vw','-66px','13deg'],['32vw','-22px','-10deg']];
      for (const [x,y,r] of items) {
        const el = document.createElement('div');
        el.className = 'emoji';
        el.textContent = emoji;
        el.style.top = direction === 'down' ? '26%' : '68%';
        el.style.setProperty('--x', x);
        el.style.setProperty('--r', r);
        el.style.setProperty('--travel', direction === 'down' ? `calc(190px + ${y})` : `calc(-260px + ${y})`);
        effectsEl.appendChild(el);
        setTimeout(() => el.remove(), 1100);
      }
    }
    function showHello() {
      const el = document.createElement('div');
      el.className = 'hello';
      el.textContent = '반갑습니다!';
      effectsEl.appendChild(el);
      setTimeout(() => el.remove(), 1250);
    }
    function updateUi(data) {
      const mode = data.mode || 'STARTING';
      instructionEl.textContent = guideFor(mode);
      statusEl.textContent = data.status || '';
      modeEl.textContent = mode;
      modeEl.classList.toggle('on', mode === 'ARMED' || mode === 'RECORDING');
      staticEl.textContent = `static ${data.static || '-'} ${(data.staticConfidence || 0).toFixed(2)}`;
      swipeEl.textContent = `swipe ${data.swipe || '-'} ${(data.swipeConfidence || 0).toFixed(2)}`;
      progressEl.style.width = `${Math.max(0, Math.min(1, data.progress || 0)) * 100}%`;
      if (gallery.length && data.imageIndex !== lastIndex) {
        const old = lastIndex;
        lastIndex = data.imageIndex;
        photo.src = gallery[data.imageIndex % gallery.length].url;
        photo.className = 'photo';
        void photo.offsetWidth;
        const dir = data.transitionDirection || (old >= 0 && data.imageIndex > old ? 'right' : 'left');
        photo.classList.add(`enter-${dir}`);
      }
      if ((data.likeSeq || 0) > lastLikeSeq) { lastLikeSeq = data.likeSeq || 0; burstReaction('👍', 'up'); }
      if ((data.dislikeSeq || 0) > lastDislikeSeq) { lastDislikeSeq = data.dislikeSeq || 0; burstReaction('👎', 'down'); }
      if ((data.helloSeq || 0) > lastHelloSeq) { lastHelloSeq = data.helloSeq || 0; showHello(); }
    }
    async function loadGallery() {
      const res = await fetch('/api/gallery', { cache: 'no-store' });
      gallery = await res.json();
      if (gallery.length) { photo.src = gallery[0].url; lastIndex = 0; }
    }
    async function sendLandmarks(landmarks) {
      const vector63 = landmarks.flatMap(p => [p.x, p.y, p.z || 0]);
      const res = await fetch('/api/landmarks', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ landmarks: vector63 }) });
      updateUi(await res.json());
    }
    async function sendNoHand() {
      const res = await fetch('/api/no-hand', {method:'POST'});
      updateUi(await res.json());
    }
    async function sendInference(results) {
      const now = performance.now();
      if (inferenceInFlight || now - lastInferenceAt < inferenceIntervalMs) return;
      inferenceInFlight = true;
      lastInferenceAt = now;
      try {
        if (results.multiHandLandmarks && results.multiHandLandmarks.length) await sendLandmarks(results.multiHandLandmarks[0]);
        else await sendNoHand();
      } catch (err) {
        statusEl.textContent = `연결 재시도 중: ${err && err.message || err}`;
      } finally {
        inferenceInFlight = false;
      }
    }
    function drawPreview(results) {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
      if (results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
          drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {color:'#68f28e', lineWidth:3});
          drawLandmarks(ctx, landmarks, {color:'#fff', lineWidth:1, radius:3});
        }
      }
      ctx.restore();
    }
	    async function main() {
	      await loadGallery();
	      if (!window.isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
	        instructionEl.textContent = 'HTTPS 또는 localhost가 필요합니다';
	        statusEl.textContent = '브라우저가 공인 IP의 HTTP 카메라 접근을 막을 수 있습니다';
	      }
	      const hands = new Hands({locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
      hands.setOptions({maxNumHands:1, modelComplexity:1, minDetectionConfidence:0.45, minTrackingConfidence:0.45});
      hands.onResults(async results => {
        drawPreview(results);
        await sendInference(results);
      });
      const camera = new Camera(video, {onFrame: async () => { await hands.send({image: video}); }, width: 640, height: 480});
      await camera.start();
    }
    main().catch(err => {
      instructionEl.textContent = '카메라 권한이 필요합니다';
      statusEl.textContent = String(err && err.message || err);
    });
  </script>
</body>
</html>
"""


class BrowserGestureEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.checkpoint, self.class_names = load_model(args.model, self.device)
        self.static_model, self.static_id_to_label, self.static_mean, self.static_std = load_static_model(
            args.static_model, args.static_labels, args.static_mean, args.static_std, self.device
        )
        self.target_len = int(self.checkpoint["sequence_length"])
        self.gallery = self._load_gallery(args.images)
        self.state = "IDLE"
        self.image_index = 0
        self.fist_count = 0
        self.last_armed_time = 0.0
        self.last_sample_time = 0.0
        self.recording_started = 0.0
        self.last_action_time = 0.0
        self.record_buffer: list[np.ndarray] = []
        self.prediction = Prediction()
        self.static_prediction = StaticPrediction()
        self.progress = 0.0
        self.status = "Show one hand"
        self.like_seq = 0
        self.dislike_seq = 0
        self.hello_seq = 0
        self.last_like_time = -999.0
        self.last_dislike_time = -999.0
        self.last_hello_time = -999.0
        self.transition_direction = "right"

    def _load_gallery(self, image_dir: Path | None) -> list[dict]:
        files = []
        if image_dir and image_dir.exists():
            for path in sorted(image_dir.iterdir()):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    files.append({"name": path.name, "url": f"/gallery/{path.name}"})
        return files

    def no_hand(self) -> dict:
        now = time.monotonic()
        if self.state == "RECORDING" and now - self.recording_started >= self.args.recording_timeout:
            self.state = "IDLE"
            self.record_buffer = []
            self.progress = 0.0
            self.status = "Make a fist again"
        elif self.state == "ARMED":
            self.status = "지금 움직이세요"
        else:
            self.status = "Show one hand"
        return self.snapshot()

    def process(self, vector63: list[float]) -> dict:
        now = time.monotonic()
        v63 = np.array(vector63, dtype=np.float32)
        v42 = v63.reshape(21, 3)[:, :2].reshape(42).astype(np.float32)
        static_label, static_conf = predict_static(
            self.static_model, v42, self.static_mean, self.static_std, self.device, self.static_id_to_label
        )
        is_fist = static_label == FIST_CLASS and static_conf >= self.args.fist_confidence_threshold
        self.fist_count = min(self.args.fist_frames, self.fist_count + 1) if is_fist else 0
        self.static_prediction = StaticPrediction(static_label, static_conf, self.fist_count >= self.args.fist_frames)

        if static_label == LIKE_CLASS and static_conf >= self.args.like_confidence_threshold and now - self.last_like_time >= self.args.like_cooldown_seconds:
            self.like_seq += 1
            self.last_like_time = now
        if static_label == DISLIKE_CLASS and static_conf >= self.args.dislike_confidence_threshold and now - self.last_dislike_time >= self.args.dislike_cooldown_seconds:
            self.dislike_seq += 1
            self.last_dislike_time = now
        if static_label == PALM_CLASS and static_conf >= self.args.hello_confidence_threshold and now - self.last_hello_time >= self.args.hello_cooldown_seconds:
            self.hello_seq += 1
            self.last_hello_time = now

        if self.state == "IDLE":
            self.progress = 0.0
            if self.static_prediction.armed and now - self.last_action_time >= self.args.cooldown_seconds:
                self.state = "ARMED"
                self.last_armed_time = now
                self.record_buffer = []
                self.status = "지금 움직이세요"
            else:
                self.status = "Make a fist to start"
        elif self.state == "ARMED":
            if now - self.last_armed_time >= self.args.active_timeout:
                self.state = "IDLE"
                self.status = "Make a fist again"
            elif self.static_prediction.armed:
                self.status = "지금 움직이세요"
            else:
                self.state = "RECORDING"
                self.record_buffer = []
                self.last_sample_time = 0.0
                self.recording_started = now
                self.status = "지금 움직이세요"
        elif self.state == "RECORDING":
            if now - self.last_sample_time >= 1.0 / self.args.sample_fps and not is_fist:
                self.record_buffer.append(v63.copy())
                self.last_sample_time = now
            self.progress = min(1.0, len(self.record_buffer) / max(1, int(round(self.args.sample_fps * self.args.record_seconds))))
            if self.progress >= 1.0:
                self._finish_recording(now)

        return self.snapshot()

    def _finish_recording(self, now: float) -> None:
        if not self.record_buffer:
            self.state = "IDLE"
            self.status = "Make a fist again"
            return
        model_input, reason = prepare_recorded_sequence(
            np.stack(self.record_buffer).astype(np.float32),
            target_len=self.target_len,
            min_detected_frames=self.args.min_detected_frames,
            min_longest_run=self.args.min_longest_run,
            min_active_density=self.args.min_active_density,
        )
        if model_input is None:
            self.prediction = Prediction("-", 0.0, False)
            self.status = reason
        else:
            label, conf = predict(self.model, model_input, self.device, self.class_names)
            accepted = label in {LEFT_CLASS, RIGHT_CLASS, UP_CLASS, DOWN_CLASS} and conf >= self.args.confidence_threshold
            self.prediction = Prediction(label, conf, accepted)
            if accepted and self.gallery:
                if label == LEFT_CLASS:
                    self.image_index = (self.image_index + 1) % len(self.gallery)
                    self.transition_direction = "right"
                    self.status = "Moved right"
                elif label == RIGHT_CLASS:
                    self.image_index = (self.image_index - 1) % len(self.gallery)
                    self.transition_direction = "left"
                    self.status = "Moved left"
                elif label == UP_CLASS:
                    self.image_index = (self.image_index - self.args.columns) % len(self.gallery)
                    self.transition_direction = "up"
                    self.status = "Moved up"
                elif label == DOWN_CLASS:
                    self.image_index = (self.image_index + self.args.columns) % len(self.gallery)
                    self.transition_direction = "down"
                    self.status = "Moved down"
                self.last_action_time = now
            else:
                self.status = "No move"
        self.state = "IDLE"
        self.record_buffer = []
        self.progress = 0.0
        self.status = f"{self.status} - make a fist for next swipe"

    def snapshot(self) -> dict:
        return {
            "status": self.status,
            "mode": self.state,
            "imageIndex": self.image_index,
            "imageCount": len(self.gallery),
            "static": self.static_prediction.label,
            "staticConfidence": self.static_prediction.confidence,
            "swipe": self.prediction.label,
            "swipeConfidence": self.prediction.confidence,
            "progress": self.progress,
            "likeSeq": self.like_seq,
            "dislikeSeq": self.dislike_seq,
            "helloSeq": self.hello_seq,
            "transitionDirection": self.transition_direction,
        }


def make_handler(engine: BrowserGestureEngine, root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def send_json(self, data: object) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/gallery":
                self.send_json(engine.gallery)
            elif path == "/state.json":
                self.send_json(engine.snapshot())
            elif path.startswith("/gallery/"):
                name = Path(unquote(path.removeprefix("/gallery/"))).name
                file_path = root / name
                if not file_path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = file_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/no-hand":
                self.send_json(engine.no_hand())
                return
            if path != "/api/landmarks":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.send_json(engine.process(payload.get("landmarks", [])))

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser-camera gesture gallery server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--static-model", type=Path, default=DEFAULT_STATIC_MODEL)
    parser.add_argument("--static-labels", type=Path, default=DEFAULT_STATIC_LABELS)
    parser.add_argument("--static-mean", type=Path, default=DEFAULT_STATIC_MEAN)
    parser.add_argument("--static-std", type=Path, default=DEFAULT_STATIC_STD)
    parser.add_argument("--images", type=Path, default=DEMO_ROOT / "gallery")
    parser.add_argument("--sample-fps", type=float, default=12.0)
    parser.add_argument("--record-seconds", type=float, default=0.65)
    parser.add_argument("--active-timeout", type=float, default=6.0)
    parser.add_argument("--recording-timeout", type=float, default=1.4)
    parser.add_argument("--cooldown-seconds", type=float, default=0.75)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--fist-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--fist-frames", type=int, default=3)
    parser.add_argument("--like-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--like-cooldown-seconds", type=float, default=1.2)
    parser.add_argument("--dislike-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--dislike-cooldown-seconds", type=float, default=1.2)
    parser.add_argument("--hello-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--hello-cooldown-seconds", type=float, default=1.5)
    parser.add_argument("--min-detected-frames", type=int, default=3)
    parser.add_argument("--min-longest-run", type=int, default=2)
    parser.add_argument("--min-active-density", type=float, default=0.30)
    parser.add_argument("--ssl-cert", type=Path)
    parser.add_argument("--ssl-key", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.images.resolve()
    engine = BrowserGestureEngine(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, root))
    scheme = "http"
    if args.ssl_cert and args.ssl_key:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.ssl_cert, args.ssl_key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print(f"Serving browser-camera gallery on {scheme}://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
