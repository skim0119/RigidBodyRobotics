# Inverse Biliard

This folder contains three main scripts.

## 1) `extractrejectory.py`

What it is for:
- Extracts 2D ball trajectories from a video.
- Lets you pick table corners and ball ROIs, then tracks white/red/yellow over time.
- Converts image coordinates to table metric coordinates (meters) via homography.

How to use:
```bash
python extractrejectory.py \
  --video <video_name>.mp4
```

What it generates:
- `<video_name>.csv` with per-frame ball centers (`*_x_m`, `*_y_m`, plus pixel columns).
- `<video_name>.png` trajectory plot.
- `<video_name>_tracked.mp4` overlay video (if enabled).
- `<video_name>.extract_cache.json` cached corner/ROI selections.
