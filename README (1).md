# Face Tracker — Intelligent Visitor Counter

An AI-driven unique visitor counter that processes video streams to detect, track, and recognize faces in real-time. Built for the Katomaran Hackathon.

---

## Demo Video

[ADD LOOM LINK HERE]

---

## Setup Instructions

### 1. Open Google Colab and mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install dependencies

```bash
pip install ultralytics insightface opencv-python-headless onnxruntime streamlit pyngrok gdown
```

### 3. Place your videos in Google Drive and update `config.json` with the folder path

### 4. Run the notebook cells 1 through 14 in order

Cell 9 is the main processing loop.

### 5. For RTSP live stream

Set `rtsp_url` in `config.json` and rerun Cell 9:
```json
"rtsp_url": "rtsp://username:password@camera_ip:554/stream"
```

### 6. Launch dashboard

Run Cell 14 with your ngrok authtoken from https://dashboard.ngrok.com/get-started/your-authtoken

---

## Sample config.json

```json
{
  "detection_skip_frames": 5,
  "similarity_threshold": 0.55,
  "stability_frames": 1,
  "video_folder": "/content/drive/MyDrive/Video Datasets",
  "db_path": "/content/face_tracker/visitors.db",
  "log_dir": "/content/face_tracker/logs",
  "exit_timeout_frames": 30,
  "rtsp_url": ""
}
```

| Parameter | Description |
|---|---|
| `detection_skip_frames` | Run YOLO every N frames |
| `similarity_threshold` | Cosine similarity cutoff for face re-identification |
| `stability_frames` | Minimum frames a track must survive before being counted |
| `exit_timeout_frames` | Frames a face can be missing before declaring an exit |
| `rtsp_url` | RTSP stream URL - leave empty for video file mode |

---

## Assumptions

- All videos are from the same fixed camera so embeddings are shared across videos as one continuous session
- Face crops smaller than 20x20 pixels are skipped
- The system runs on CPU. GPU would reduce processing time significantly
- Exit is declared after 30 consecutive missed detection cycles
- CCTV footage produces small face crops (~60-90px) where embedding similarity is inherently lower than frontal high-resolution faces. Centroid tracking is used as primary re-ID within videos. Embedding matching handles re-ID across track breaks on a best-effort basis

---

## Architecture

```mermaid
flowchart TD
    A[Video Files / RTSP Stream] --> B[YOLOv8 Face Detector]
    B --> C[InsightFace ArcFace Embeddings]
    C --> D{Embedding Matcher}
    D -->|New Face| E[Register + Count]
    D -->|Known Face| F[Re-entry — Do Not Count]
    E --> G[Centroid + IoU Tracker]
    F --> G
    G -->|Track Active| H[Log Tracking]
    G -->|Track Timeout| I[Log Exit]
    E --> J[SQLite Database]
    I --> J
    J --> K[Streamlit Dashboard]
    E --> L[File System Logs]
    I --> L
```

---

## AI Planning Document

### Planning the app

The problem was broken into four independent modules:

1. **Detection** - YOLO detects face bounding boxes every N frames
2. **Recognition** - InsightFace generates 512-dim ArcFace embeddings per face
3. **Tracking** - Centroid + IoU tracker links detections across frames and declares exits
4. **Logging** - All events written to SQLite DB, events.log, and local image store

### Features

- YOLOv8 face detection with configurable skip frames
- InsightFace ArcFace 512-dim embedding generation
- Centroid + IoU tracking as primary re-ID within video
- Embedding cosine similarity for re-ID across track breaks
- Auto-registration of new faces with UUID
- Entry and exit logging with timestamped cropped face images
- Structured folder storage: logs/entries/YYYY-MM-DD/ and logs/exits/YYYY-MM-DD/
- SQLite database with visitors, unique_visitors, and embeddings tables
- events.log with ENTRY, EXIT, TRACKING, EMBEDDING_GENERATED, RE_IDENTIFIED events
- RTSP live stream support via config
- Streamlit dashboard with face gallery, timeline, event log, and CSV export

### Compute estimate

| Component | CPU Load | GPU Load |
|---|---|---|
| YOLOv8n detection | 15-25% | N/A |
| InsightFace embedding | 30-40% | ~5% with CUDA |
| Centroid + IoU tracker | less than 1% | N/A |
| SQLite writes | less than 1% | N/A |
| Total | 40-60% average | ~5-10% with GPU |

Processing time on CPU is approximately 2-3 minutes per video. On GPU with CUDAExecutionProvider, real-time 30fps processing becomes feasible.

---

## Sample Output

```
Total events logged  : 538
Unique visitors      : 172
Entry events         : 269
Exit events          : 269
Videos covered       : 23
Entry images saved   : 269
Exit images saved    : 269
Log file lines       : 677
```

---

## Known Limitations

The CCTV camera is mounted at a distance, resulting in small face crops (~60-90px). InsightFace's ArcFace model is optimised for frontal faces at 112x112px or larger. On this dataset, cosine similarity for the same person across frames averaged only 0.11, making pure threshold-based re-identification unreliable. The centroid tracker handles re-ID within videos reliably, and embedding matching is used as best-effort re-ID across track breaks.

For production deployment, a higher resolution camera or a model specifically trained on low-resolution CCTV footage such as OSNet would significantly improve re-identification accuracy.

---

This project is a part of a hackathon run by https://katomaran.com
