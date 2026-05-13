import os
import sys
import queue
import threading
import time
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

        self.conf_value = tk.DoubleVar(value=0.20)  # Низкий порог для лучшей стабильности
        self.monitor_index = tk.IntVar(value=1)

        self._running = False
        self._starting = False
        self._stop_event = None

        self._capturer = None
        self._detector = None
        self._smoother = None
        self._overlay = None
        self._fps_counter = None

        self._frame_queue = None
        self._capture_thread = None
        self._pipeline_thread = None

        # Буфер для усреднения детекций (устранение мерцания)
        self._detection_buffer = []
        self._buffer_size = 3  # Усредняем 3 кадра

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

        tk.Label(frame, text="Порог уверенности:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.conf_scale = tk.Scale(
            frame,
            from_=0.10,
            to=0.50,
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
            text="Совет: порог 0.15-0.25. Оверлей держит боксы 0.5 сек после пропажи",
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
        if self._starting:
            return

        if self._running:
            self._stop_censorship()
        else:
            self._start_censorship()

    def _start_censorship(self):
        if self._running or self._starting:
            return

        model = Path(self.model_path.get())

        if not model.exists():
            messagebox.showerror("Ошибка", f"Файл модели не найден:\n{model}")
            return

        try:
            self._starting = True
            self._set_controls_state("disabled")
            self.toggle_button.config(state="disabled", text="Запуск...")
            self.status_label.config(text="Статус: запускается...")
            self.root.update_idletasks()

            self._stop_event = threading.Event()
            self._frame_queue = queue.Queue(maxsize=2)

            # Очищаем буфер детекций
            self._detection_buffer = []

            self._fps_counter = FpsCounter()

            self._capturer = ScreenCapturer(
                monitor_index=int(self.monitor_index.get()),
                fps_limit=30,
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

            # Упрощенный трекер - только преобразование координат
            self._smoother = TrackSmoother(
                ema_alpha=0.5,
                min_confidence=float(self.conf_value.get()),
            )

            # Оверлей с АГРЕССИВНЫМ удержанием (0.5 секунды)
            self._overlay = WindowsOverlay(
                screen_w,
                screen_h,
                update_rate=24,
                hold_seconds=0.5,  # Держим бокс 0.5 секунды после пропажи
                fade_seconds=0.2
            )
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

            self.toggle_button.config(state="normal", text="Выключить цензуру")
            self.status_label.config(text="Статус: работает")

        except Exception as e:
            self._running = False
            self._set_controls_state("normal")
            self._safe_cleanup()
            messagebox.showerror("Ошибка запуска", str(e))
        finally:
            self._starting = False
            if not self._running:
                self.toggle_button.config(state="normal", text="Включить цензуру")

    def _stop_censorship(self):
        if self._starting:
            return

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
        self._detection_buffer = []

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

    def _merge_detections(self, detections_list):
        """
        Объединить детекции из нескольких кадров (медианный фильтр для устранения мерцания)
        """
        if not detections_list:
            return []

        # Собираем все боксы со всех кадров
        all_boxes = []
        for detections in detections_list:
            for det in detections:
                all_boxes.append(det)

        if not all_boxes:
            return []

        # Группируем похожие боксы по позиции
        merged = []
        used = [False] * len(all_boxes)

        for i, box1 in enumerate(all_boxes):
            if used[i]:
                continue

            similar = [box1]
            center1 = ((box1.x1 + box1.x2) / 2, (box1.y1 + box1.y2) / 2)

            for j, box2 in enumerate(all_boxes[i + 1:], i + 1):
                if used[j]:
                    continue

                center2 = ((box2.x1 + box2.x2) / 2, (box2.y1 + box2.y2) / 2)
                distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

                if distance < 50:  # Близкие боксы
                    similar.append(box2)
                    used[j] = True

            if similar:
                # Усредняем координаты
                avg_x1 = sum(b.x1 for b in similar) / len(similar)
                avg_y1 = sum(b.y1 for b in similar) / len(similar)
                avg_x2 = sum(b.x2 for b in similar) / len(similar)
                avg_y2 = sum(b.y2 for b in similar) / len(similar)
                avg_conf = sum(b.conf for b in similar) / len(similar)

                from Core.detector import Detection
                merged.append(Detection(
                    x1=int(avg_x1), y1=int(avg_y1),
                    x2=int(avg_x2), y2=int(avg_y2),
                    conf=avg_conf, track_id=None
                ))

            used[i] = True

        return merged

    def _pipeline_loop(self, screen_w, screen_h, monitor_left, monitor_top):
        """Основной pipeline обработки с буферизацией для устранения мерцания"""
        frame_counter = 0
        last_print_time = time.time()
        debug = False  # Включите True для отладки в консоль

        while self._stop_event and not self._stop_event.is_set():
            try:
                try:
                    frame = self._frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                frame_counter += 1

                # Получаем детекции
                detections = self._detector.track(frame)

                # Отладка
                if debug:
                    current_time = time.time()
                    if current_time - last_print_time > 2.0:
                        print(f"[DEBUG] Кадр {frame_counter}: найдено {len(detections)} объектов")
                        for d in detections:
                            print(f"  - Бутылка: conf={d.conf:.2f}")
                        last_print_time = current_time

                # БУФЕРИЗАЦИЯ: накапливаем детекции за несколько кадров
                self._detection_buffer.append(detections)
                if len(self._detection_buffer) > self._buffer_size:
                    self._detection_buffer.pop(0)

                # УСРЕДНЕНИЕ: объединяем детекции из буфера
                if len(self._detection_buffer) >= 2:
                    detections = self._merge_detections(self._detection_buffer)

                # Обновляем порог уверенности в трекере
                current_conf = float(self.conf_value.get())
                if hasattr(self._smoother, 'min_confidence') and self._smoother.min_confidence != current_conf:
                    self._smoother.min_confidence = current_conf

                # Преобразуем в экранные координаты
                boxes = self._smoother.update(
                    detections=detections,
                    screen_w=screen_w,
                    screen_h=screen_h,
                    monitor_left=monitor_left,
                    monitor_top=monitor_top,
                    input_w=frame.shape[1],
                    input_h=frame.shape[0],
                )

                # Отправляем в оверлей (он сам решит, что рисовать, с учетом удержания)
                if self._overlay:
                    self._overlay.update_boxes(boxes)

                if self._fps_counter:
                    self._fps_counter.tick()

            except Exception as e:
                print(f"[ERROR] {e}")
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
        self._detection_buffer = []

    def _set_controls_state(self, state):
        self.model_entry.config(state=state)
        self.conf_scale.config(state=state)
        self.monitor_spin.config(state=state)

    def _on_close(self):
        if self._running:
            self._stop_censorship()
        self.root.destroy()