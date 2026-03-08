# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
python main.py
```

Откроется окно управления (tkinter). Выбрать модель, настроить порог, нажать
"Включить цензуру" — появится прозрачный оверлей поверх всего экрана.
Оверлей click-through: мышь и клавиатура работают в обычном режиме.

## Running Tests

```bash
python -m pytest Tests/
python -m pytest Tests/test_detector.py
python -m pytest Tests/test_detector.py::test_function_name
```

## Model Training

```bash
# 1. Download dataset from Roboflow (see TZ.md — Участник 4)
# 2. Train, export to ONNX and copy to Models/ — all in one step
python train.py
```

## Architecture

Real-time screen capture → YOLO26 detection → ByteTrack tracking → transparent overlay with black rectangles. Runs at ~20 FPS on CPU.

**Core pipeline** (`Core/`):
- `capture.py` — `ScreenCapturer` wraps platform backends; auto-selects based on env vars
- `backends/mss_backend.py` — Windows / X11 capture via `mss`
- `backends/wayland_backend.py` — Hyprland capture via PipeWire + xdg-desktop-portal
- `detector.py` — `BottleDetector` runs YOLO26 + ByteTrack; returns `Detection` dataclasses in 640×640 space
- `tracker.py` — `TrackSmoother` projects detections to screen coords, applies EMA smoothing and decay

**Overlay** (`Overlay/`): `base.py` defines `OverlayBase` ABC; platform implementations run their render loop in a background thread. `update_boxes()` is always thread-safe.
- Linux: GTK3 + gtk-layer-shell (wlr-layer-shell protocol), Cairo rendering
- Windows: Win32 layered window (`WS_EX_LAYERED`), DIBSection + `UpdateLayeredWindow`

**GUI** (`gui.py`): tkinter control panel — Start/Stop button, model path picker, confidence slider, monitor selector, FPS display. Runs on the main thread. All widget updates from background threads must go through `root.after(0, callback)`.

**Threading model**: Main thread runs tkinter. On Start: Thread A captures frames into a bounded queue (maxsize=2), Thread B pulls frames, runs detection+tracking, calls `overlay.update_boxes()`. Thread C is the OS render loop (GTK/Win32). On Stop: `stop_event.set()` + `overlay.stop()` + join threads.

## Linux Setup (Hyprland)

```bash
sudo pacman -S xdg-desktop-portal-hyprland pipewire gstreamer \
               gst-plugin-pipewire gst-plugins-base \
               python-dbus python-gobject gtk-layer-shell
```

On first run, Hyprland will show a screen-share permission dialog — select the monitor to capture.
