import os
import sys
import time
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2

from Core.capture import ScreenCapturer
from Core.detector import BottleDetector
from Core.tracker import TrackSmoother
from Core.utils import FpsCounter
from Overlay.windows.overlay_win import WindowsOverlay


class CensorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Alcohol Censor")
        self.root.resizable(False, False)

        self.project_dir = Path(__file__).resolve().parent
        self.model_path = tk.StringVar(
            value=str(self.project_dir / "Models" / "best.onnx")
        )

        self.conf_value = tk.DoubleVar(value=0.35)
        self.monitor_index = tk.IntVar(value=1)

        self._running = False
        self._stop_event = None

        self._capturer = None
        self._detector = None
        self._smoother = None
        self._overlay = None
        self._fps_counter = None

        self._frame_queue = None
        self._capture_thread = None
        self._pipeline_thread = None

        self._build_ui()
        self._update_fps_label()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        pad = 8

        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        title = tk.Label(frame, text="Alcohol Censor", font=("Arial", 14, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        tk.Label(frame, text="Модель:").grid(row=1, column=0, sticky="w")
        self.model_entry = tk.Entry(frame, textvariable=self.model_path, width=45)
        self.model_entry.grid(row=1, column=1, padx=pad)
        tk.Button(frame, text="...", width=4, command=self._choose_model).grid(row=1, column=2)

        tk.Label(frame, text="Порог:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.conf_scale = tk.Scale(
            frame,
            from_=0.10,
            to=0.90,
            resolution=0.05,
            orient="horizontal",
            variable=self.conf_value,
            length=260,
        )
        self.conf_scale.grid(row=2, column=1, sticky="w", pady=(10, 0))

        self.conf_label = tk.Label(frame, textvariable=self.conf_value)
        self.conf_label.grid(row=2, column=2, sticky="w", pady=(10, 0))

        tk.Label(frame, text="Монитор:").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.monitor_spin = tk.Spinbox(
            frame,
            from_=1,
            to=8,
            width=5,
            textvariable=self.monitor_index,
        )
        self.monitor_spin.grid(row=3, column=1, sticky="w", pady=(10, 0))

        self.toggle_button = tk.Button(
            frame,
            text="Включить цензуру",
            width=24,
            height=2,
            command=self._toggle,
        )
        self.toggle_button.grid(row=4, column=0, columnspan=3, pady=(18, 8))

        self.status_label = tk.Label(frame, text="Статус: выключено")
        self.status_label.grid(row=5, column=0, columnspan=3, sticky="w")

        self.fps_label = tk.Label(frame, text="FPS: --")
        self.fps_label.grid(row=6, column=0, columnspan=3, sticky="w")

        self.info_label = tk.Label(
            frame,
            text="Перед запуском убедись, что Models/best.onnx существует.",
            fg="gray",
        )
        self.info_label.grid(row=7, column=0, columnspan=3, sticky="w", pady=(10, 0))

    def _choose_model(self):
        file_path = filedialog.askopenfilename(
            title="Выбери модель",
            initialdir=str(self.project_dir / "Models"),
            filetypes=[
                ("Model files", "*.onnx *.pt"),
                ("ONNX files", "*.onnx"),
                ("YOLO PyTorch files", "*.pt"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            self.model_path.set(file_path)

    def _toggle(self):
        if self._running:
            self._stop_censorship()
        else:
            self._start_censorship()

    def _start_censorship(self):
        model = Path(self.model_path.get())

        if not model.exists():
            messagebox.showerror("Ошибка", f"Файл модели не найден:\n{model}")
            return

        try:
            self._set_controls_state("disabled")

            self._stop_event = threading.Event()
            self._frame_queue = queue.Queue(maxsize=2)

            self._fps_counter = FpsCounter()

            self._capturer = ScreenCapturer(
                monitor_index=int(self.monitor_index.get()),
                fps_limit=20,
            )
            self._capturer.start()

            monitor_rect = self._capturer.get_monitor_rect()
            screen_w = monitor_rect["width"]
            screen_h = monitor_rect["height"]
            monitor_left = monitor_rect["left"]
            monitor_top = monitor_rect["top"]

            self._detector = BottleDetector(
                model_path=str(model),
                conf=float(self.conf_value.get()),
                device="auto",
            )

            self._smoother = TrackSmoother(
                ema_alpha=0.6,
                decay_frames=10,
                min_confidence=float(self.conf_value.get()),
            )

            self._overlay = WindowsOverlay(screen_w, screen_h)
            self._overlay.start()

            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
            )

            self._pipeline_thread = threading.Thread(
                target=self._pipeline_loop,
                args=(screen_w, screen_h, monitor_left, monitor_top),
                daemon=True,
            )

            self._running = True
            self._capture_thread.start()
            self._pipeline_thread.start()

            self.toggle_button.config(text="Выключить цензуру")
            self.status_label.config(text="Статус: работает")

        except Exception as e:
            self._running = False
            self._set_controls_state("normal")
            self._safe_cleanup()
            messagebox.showerror("Ошибка запуска", str(e))

    def _stop_censorship(self):
        self.status_label.config(text="Статус: выключается...")
        self.root.update_idletasks()

        self._running = False

        if self._stop_event:
            self._stop_event.set()

        try:
            if self._overlay:
                self._overlay.update_boxes([])
                self._overlay.stop()
        except Exception:
            pass

        try:
            if self._capturer:
                self._capturer.stop()
        except Exception:
            pass

        try:
            if self._smoother:
                self._smoother.reset()
        except Exception:
            pass

        try:
            if self._detector:
                self._detector.reset_tracker()
        except Exception:
            pass

        self._capturer = None
        self._detector = None
        self._smoother = None
        self._overlay = None
        self._fps_counter = None
        self._frame_queue = None
        self._stop_event = None

        self.toggle_button.config(text="Включить цензуру")
        self.status_label.config(text="Статус: выключено")
        self.fps_label.config(text="FPS: --")
        self._set_controls_state("normal")

    def _capture_loop(self):
        while self._stop_event and not self._stop_event.is_set():
            try:
                frame = self._capturer.grab()

                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                self._frame_queue.put(frame)

            except Exception as e:
                self.root.after(0, self._show_worker_error, f"Ошибка захвата экрана: {e}")
                break

    def _pipeline_loop(self, screen_w, screen_h, monitor_left, monitor_top):
        while self._stop_event and not self._stop_event.is_set():
            try:
                try:
                    frame = self._frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                small_frame = cv2.resize(frame, (640, 640))

                detections = self._detector.track(small_frame)

                boxes = self._smoother.update(
                    detections=detections,
                    screen_w=screen_w,
                    screen_h=screen_h,
                    monitor_left=monitor_left,
                    monitor_top=monitor_top,
                    input_w=640,
                    input_h=640,
                )

                if self._overlay:
                    self._overlay.update_boxes(boxes)

                if self._fps_counter:
                    self._fps_counter.tick()

            except Exception as e:
                self.root.after(0, self._show_worker_error, f"Ошибка обработки кадра: {e}")
                break

    def _update_fps_label(self):
        if self._running and self._fps_counter:
            fps = self._fps_counter.get()
            self.fps_label.config(text=f"FPS: {fps:.1f}")

        self.root.after(1000, self._update_fps_label)

    def _show_worker_error(self, text):
        if self._running:
            messagebox.showerror("Ошибка", text)
            self._stop_censorship()

    def _safe_cleanup(self):
        try:
            if self._overlay:
                self._overlay.update_boxes([])
                self._overlay.stop()
        except Exception:
            pass

        try:
            if self._capturer:
                self._capturer.stop()
        except Exception:
            pass

        try:
            if self._smoother:
                self._smoother.reset()
        except Exception:
            pass
        
        try:
            if self._detector:
                self._detector.reset_tracker()
        except Exception:
            pass

        self._capturer = None
        self._detector = None
        self._smoother = None
        self._overlay = None
        self._fps_counter = None
        self._frame_queue = None
        self._capture_thread = None
        self._pipeline_thread = None
        self._stop_event = None

    def _set_controls_state(self, state):
        self.model_entry.config(state=state)
        self.conf_scale.config(state=state)
        self.monitor_spin.config(state=state)

    def _on_close(self):
        if self._running:
            self._stop_censorship()
        self.root.destroy()