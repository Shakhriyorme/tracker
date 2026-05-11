# 🎓 AI-Powered Attendance Tracker

> **Local-first, real-time attendance system** built with computer vision. Enroll once via webcam, then walk past the camera to get automatically marked present, late, or absent. No cloud APIs, no subscription fees — runs entirely offline.

## ✨ Key Features
- 📷 **Real-Time Tracking** – MJPEG live stream with YOLOv8 person detection + SORT tracking
- 👤 **Smart Enrollment** – Multi-pose face sampling (front/left/right) with automatic duplicate detection & pose estimation
- 🧠 **4-Tier Recognition Fallback** – `insightface` → `facenet-pytorch` → `dlib` → OpenCV LBP (auto-selects best available at startup)
- 📊 **Live Dashboard** – Real-time counters, hourly arrival heatmap (Chart.js), roster table, CSV export, 30-day attendance calendar
- ⚙️ **Flexible Sessions** – Student mode (present/late/absent threshold) & Worker mode (clock-in/clock-out + hours)
- 🗄️ **Dual Database Support** – SQLite (default, zero-config) or PostgreSQL + `pgvector` (Neon) via single `DATABASE_URL` env var
- 📱 **IP Camera & Phone Support** – Auto-rotation, front/back camera toggle via HTTP API, seamless USB/IP-cam switching

## 🖥️ Frontend Interface
The web UI is built with server-rendered Jinja2 templates + vanilla JS + Chart.js. All pages share a consistent navigation and session status bar.

| Page | Route | Description | Key Components |
|------|-------|-------------|----------------|
| **Home** | `/` | Session management & people overview | Create session form, active session status, member selector, session table with activate/export/edit/delete |
| **Enroll** | `/enroll` | Register new faces | Live camera preview, name/ID/role form, pose angle selector (front/left/right), capture button, enrolled persons list |
| **Live** | `/live` | Real-time recognition feed | MJPEG video stream, FPS/ID/Unknown HUD overlay, snapshot download, "Just arrived" tracker |
| **Dashboard** | `/dashboard` | Analytics & reporting | Present/late/absent counters, hourly arrivals bar chart, attendance roster table, unknown sightings log, CSV export |
| **Person** | `/person/<id>` | Individual history | 30-day calendar heatmap, session-by-session history, attendance percentage, profile details |

## 🏗️ Architecture & Data Flow
### 🔹 Enrollment Flow
1. User opens `/enroll`, positions face in camera preview
2. Clicks "Capture" → system grabs frame → runs face detection + 512-d embedding extraction
3. Embedding is L2-normalized & stored in `face_samples` table with pose label
4. `persons` table is updated/created, cooldowns reset for immediate tracking

### 🔹 Live Recognition Flow
1. **Camera** → Background thread reads frames (USB or IP-cam) with auto-reconnect & rotation
2. **YOLOv8** → Detects `person` class bboxes (`imgsz=480` for CPU efficiency)
3. **SORT** → Kalman filter + IoU/Hungarian assignment assigns stable `track_id` across frames
4. **Recognition Guard** → Only attempts face match every 5 frames & max 12 times per track (saves ~70% CPU)
5. **Face Pipeline** → Crop → Embed → Cosine similarity vs DB → If `dist ≤ threshold`, lock identity & log attendance
6. **MJPEG Stream** → Annotated video broadcast to `/video_feed` with HUD overlay

## 🛠️ Tech Stack
| Layer | Technology |
|-------|------------|
| **Web** | Flask, Jinja2, Vanilla JS, Chart.js (CDN) |
| **Database** | SQLite (default) / PostgreSQL + `pgvector` (Neon) |
| **Detection** | YOLOv8n (`ultralytics`) |
| **Tracking** | SORT (Kalman + Hungarian + IoU), vendored |
| **Face Recognition** | InsightFace → FaceNet → DLib → OpenCV LBP |
| **Concurrency** | `threading`, `threading.Lock`, background processing loop |
| **Deployment** | Local/Edge, no Docker/auth/API keys required |

## 🚀 Quick Start
```bash
# 1. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-face.txt  # Optional: enables high-accuracy backends

# 3. (Optional) Seed demo data for dashboard preview
python seed.py

# 4. Run the application
python app.py
# → Open http://127.0.0.1:5000 in your browser