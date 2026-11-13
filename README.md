# <p align="center">Visio AI VMS</p>

<p align="center">
  <strong>AI-assisted video monitoring for camera fleets, live events, and practical security workflows.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11 or newer">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenCV-Video%20Analytics-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
</p>

---

Visio AI VMS is a self-hosted video management system prototype. It brings camera feeds, configurable detections, and event review together in a browser-based dashboard.

## ✨ Features

- **Camera monitoring** — connect webcams, video files, custom stream URLs, and supported-brand RTSP cameras.
- **AI-assisted detections** — detect people, vehicles, and animals with YOLOX ONNX models.
- **Configurable analytics** — enable motion, camera tamper, restricted-zone intrusion, line crossing, loitering, crowd, fall, unattended-object, fire, and face/watchlist events.
- **Live dashboard** — view camera streams, detection overlays, event history, snapshots, and counts.
- **Local persistence** — camera settings, events, and face entries are stored in a local SQLite database.
- **Optional face models** — YuNet and SFace models enable face detection and recognition features.

> **Prototype notice:** Change the built-in demo login before exposing the service to a network. Use only cameras and face data you are authorized to monitor, and follow applicable privacy laws.

## 🚀 Getting started

### Requirements

- Python 3.11 or newer
- Windows, macOS, or Linux
- Optional: a supported GPU runtime for accelerated ONNX inference

### Install

```powershell
git clone https://github.com/intelliboyy/VISIO_AI_.git
cd VISIO_AI_
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

### Download AI models

Model files are intentionally not stored in Git. Download them locally when you want AI detection:

```bash
python download_models.py
```

The script downloads YOLOX and OpenCV Zoo face models into `models/`. Without the YOLOX weights, the application can use its basic person-detection fallback; face features require both face model files.

### Run

```bash
python server.py
```

Open the address printed in the terminal (default: `http://localhost:8090`). A new local database starts with the demo account `admin` / `admin123`; change these credentials before using the application beyond a local prototype.

## ⚙️ Configuration

| Environment variable | Purpose |
| --- | --- |
| `VISIO_DATA` | Directory for the SQLite database and event snapshots. Defaults to `./data`. |
| `VISIO_MODEL` | YOLOX model name override, such as `yolox_tiny` or `yolox_s`. |

Camera connection details and analytics options are managed from the dashboard. Model weights and runtime data stay on the host machine and are ignored by Git.

## 🧰 Project layout

```text
.
├── analytics.py       # Tracking and rule-based analytics
├── detector.py        # YOLOX ONNX inference
├── download_models.py # Downloads model weights
├── server.py          # FastAPI app and local persistence
├── ui.html            # Browser dashboard
├── data/              # Local database and snapshots (not tracked)
└── models/            # Downloaded ONNX weights (not tracked)
```

## 📄 License

See [LICENSE](./LICENSE) for license terms. Third-party model weights retain their respective upstream licenses.