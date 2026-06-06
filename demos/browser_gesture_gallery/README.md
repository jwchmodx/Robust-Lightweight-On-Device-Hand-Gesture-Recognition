# Browser Gesture Gallery Demo

This folder contains a standalone browser-camera gallery demo for the Armed12 dynamic gesture model plus the static HAGRID MLP gesture model.

## Included files

- `armed12_browser_camera_server.py`: HTTPS/HTTP browser-camera demo server.
- `armed12_gallery_demo.py`: shared model loading and gesture inference code.
- `models/gru_armed4_24f_best.pt`: dynamic GRU checkpoint.
- `static/best_mlp.pt`, `label_to_id.json`, `mean.npy`, `std.npy`: static gesture checkpoint and normalization files.
- `gallery/`: four sample images used by the gallery.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally

Camera access works reliably on `localhost`:

```bash
python armed12_browser_camera_server.py --host 127.0.0.1 --port 3000
```

Open:

```text
http://127.0.0.1:3000/
```

## Run with HTTPS

Browsers usually require HTTPS for camera access on non-localhost hosts. Create a local self-signed certificate:

```bash
openssl req -x509 -newkey rsa:2048 -sha256 -days 30 -nodes \
  -keyout key.pem -out cert.pem \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```

Then run:

```bash
python armed12_browser_camera_server.py --host 0.0.0.0 --port 8443 --ssl-cert cert.pem --ssl-key key.pem
```

Open `https://localhost:8443/` and allow the browser security exception if prompted.

## Gestures

- `fist`: arm the next movement recording.
- release/move after fist: classify the short dynamic gesture clip.
- left/right/up/down two-finger swipe: navigate gallery with a 3D transition.
- `like`: show floating thumbs-up emojis.
- `dislike`: show floating thumbs-down emojis.
- `palm`: show `반갑습니다!`.
