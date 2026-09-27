# Drawing Feature Paths Using YOLO

> **Take-home test assignment for a job interview** (computer vision). The final solution is at the top level; the development history is in [`drafts/`](drafts).

Real-time multi-object tracking with **YOLOv8 + BoT-SORT** that draws the movement path of each object once it enters a region of interest, and clears the path when the object reaches an exit zone.

Typical use cases: monitoring how objects move through a specific area (a conveyor section, a doorway, a road lane) and visualising their trajectories live.

## How it works

1. Each frame from the camera is passed to `YOLO('yolov8x.pt').track(...)` with the **BoT-SORT** tracker, so every detected object gets a persistent ID across frames.
2. Every ID is assigned its own random colour; the bounding box and ID are drawn on the frame.
3. **Trigger zone** (green): while an object's bounding box intersects it, the object's centre point is appended to its trajectory.
4. **Delete zone** (red): when an object's bounding box intersects it, its trajectory is removed.
5. Trajectories are rendered as polylines on top of the video stream.

Zones are axis-aligned rectangles `(x1, y1, x2, y2)` defined at the top of the script:

```python
trigger_zones = [(100, 300, 300, 800)]
delete_zones  = [(500, 50, 600, 700)]
```

## Quick start

```bash
pip install -r requirements.txt
python trajectory_tracking.py
```

- The script uses the default webcam (`cv2.VideoCapture(0)`). To process a video file instead, pass its path to `cv2.VideoCapture(...)`.
- The `yolov8x.pt` weights are downloaded automatically by Ultralytics on the first run. For a faster, lighter model, use `yolov8n.pt`.
- Press **`q`** to quit.

## Files

| File | Description |
|---|---|
| `trajectory_tracking.py` | Final version of the tracker (recommended entry point) |
| `trajectory_tracking_v1.py` | Earlier variant of the same pipeline |

## Development process

This was a take-home task for a job interview. [`drafts/`](drafts) keeps the intermediate versions:

| Stage | Files | Idea |
|---|---|---|
| Tracking logic | `1.py`, `1.ipynb`, `praktik1.py` … `praktik8.py`, `practik3.py` … `practik9.py`, `nonpractik10.py` | Trigger and delete zones (including zones drawn with the mouse), ByteTrack vs BoT-SORT, trajectory drawing |
| Desktop interface | `interfase-practik1.py` … `interfases-practik6.py`, `ITOG1nterfases-practik7.py`, `ITOG2nterfases-practik7.py` | PyQt5 application for choosing the video source and zones and watching the tracking |
| GPU | `cuda.py`, `CUDAmoment.py`, `GPUmoment.py` | Checking CUDA availability and a GPU version of the interface |
| Earlier final | `Итогтраектория.py` | Previous version of `trajectory_tracking.py` |

## Tech stack

Python · Ultralytics YOLOv8 · BoT-SORT · OpenCV

## License

[MIT](LICENSE)
