# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

Opens a tkinter control panel. Select the model, set the confidence threshold, click "Включить цензуру" — a transparent click-through overlay appears over the entire screen blocking alcohol with black rectangles.

## Running Tests

```bash
python -m pytest Tests/
python -m pytest Tests/test_tracker.py
python -m pytest Tests/test_tracker.py::TestTrackSmoother::test_backprojection
```

`Tests/test_detector.py` requires `Models/best.onnx` to be present — it loads the real model. `Tests/test_tracker.py` tests `TrackSmoother` in isolation using `Detection` dataclasses directly.

## Model Training

```bash
pip install -r requirements-train.txt
python train.py
```

Dataset and Roboflow credentials are hardcoded in `train.py` (adonantonin/alcohol-iaeeq, v4, YOLO format). After training, the script prompts whether to save as `Models/best.onnx` (main) or `Models/archive/model_YYYYMMDD.onnx`. Only `Models/best.onnx` is committed to git; `Models/archive/` is gitignored.

AMD GPU requires PyTorch with ROCm instead of the default CUDA build:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

## Architecture

Real-time pipeline: screen capture → resize to 640×640 → YOLO detection → TrackSmoother (IoU + EMA) → transparent overlay with black rectangles. Runs at ~20 FPS on CPU.

**Threading model**: Main thread runs tkinter (`gui.py`). On Start: Thread A (`_capture_loop`) grabs frames into a bounded queue (maxsize=3, drops oldest on overflow), Thread B (`_pipeline_loop`) pulls frames, runs detection+tracking, calls `overlay.update_boxes()`, Thread C (`_stats_loop`, daemon) updates the performance status label every 2 s. The overlay's Win32 message loop runs in Thread D (daemon). On Stop: `stop_event.set()` + `overlay.stop()`.

**Core pipeline** (`Core/`):
- `capture.py` — `ScreenCapturer` auto-selects backend: MSS on Windows/X11 (raises `NotImplementedError` on Wayland). Takes `fps_limit` to throttle capture rate. `get_monitor_rect()` returns `{left, top, width, height}`.
- `backends/mss_backend.py` — capture via `mss`; `list_monitors()` is a static method used to populate the monitor spinner in the GUI.
- `detector.py` — `BottleDetector` wraps Ultralytics YOLO. `.pt` files filter COCO class 39 (bottle); `.onnx` files use no class filter (custom model, class 0). `track()` is an alias for `detect()` — it uses `model.predict()` and returns `track_id=None`; **ByteTrack is not active**. For ONNX models, `_patch_ort_speed()` is called before loading, which monkey-patches `ort.InferenceSession` to auto-select the best available execution provider (TensorRT → CUDA → CoreML → DirectML → CPU) and enables `ORT_ENABLE_ALL` graph optimization. `reset_tracker()` is a no-op unless a predictor with live trackers is attached.
- `tracker.py` — `TrackSmoother` receives `Detection` objects (all with `track_id=None`) in 640×640 space, projects to screen coords, matches detections to existing tracks via IoU + center-distance, applies EMA smoothing (`ema_alpha`, default 0.6), and holds boxes for `decay_frames` (default 10 in class, 6 when instantiated from GUI) after a track disappears. Supports `set_params()` for live tuning and `get_statistics()` / `get_track_stats()` for debugging. Negative track IDs are synthetic (assigned when `Detection.track_id is None`). Optional `enable_velocity_prediction` (experimental, off by default).
- `utils.py` — `FpsCounter`: thread-safe sliding-window counter; `.tick()` per frame, `.get()` returns current FPS.

**Overlay** (`Overlay/`):
- `base.py` — `OverlayBase` ABC defining `start()`, `stop()`, `update_boxes()`, `is_running()`.
- `Overlay/windows/overlay_win.py` — `WindowsOverlay` (does **not** subclass `OverlayBase` but matches its interface). Uses `WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE` Win32 window. Transparent color key is magenta `RGB(255,0,255)` — black is reserved for the censor rectangles. Renders via double-buffered GDI (`CreateCompatibleDC` + `BitBlt`). The `_WDA_EXCLUDEFROMCAPTURE` flag prevents the overlay from being captured in screenshots.
- `Overlay/linux/overlay_lin.py` — Linux GTK3 implementation (exists but not wired into `main.py` yet).

**GUI** (`gui.py`): `CensorApp` — model path picker, confidence slider (0.10–0.90, step 0.05), monitor index spinner (1–8), target FPS spinner (10–60, passed as `fps_limit` to `ScreenCapturer`), Start/Stop button, FPS label updated via `root.after(1000, ...)`. All worker-thread → UI updates must go through `root.after(0, callback)`. Confidence changes while running are propagated live via `_on_conf_changed` → `detector.set_conf()` + `smoother.set_params()`.

**Entry point** (`main.py`): Creates a Windows named mutex (`Local\\AlcoholCensorSingleInstance`) to prevent duplicate instances before launching tkinter.

**Frame pipeline detail**: `_pipeline_loop` resizes grabbed frames with `cv2.resize(frame, (input_size, input_size))` (default 640, or 416 if resolution_mode set) before passing to `BottleDetector.track()`. Detections come back in input-size coords with `track_id=None`; `TrackSmoother.update()` scales them to screen coords using the monitor rect from `ScreenCapturer.get_monitor_rect()`.

## Linux Setup (Hyprland)

```bash
sudo pacman -S xdg-desktop-portal-hyprland pipewire gstreamer \
               gst-plugin-pipewire gst-plugins-base \
               python-dbus python-gobject gtk-layer-shell
```

On first run, Hyprland will show a screen-share permission dialog — select the monitor to capture. The Wayland capture backend is not yet implemented; set `XDG_SESSION_TYPE=x11` to force MSS.
