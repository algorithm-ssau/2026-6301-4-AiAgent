import os
import sys
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np

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

        # Применяем современный стиль
        self._apply_styling()

        self.project_dir = Path(__file__).resolve().parent
        self.model_path = tk.StringVar(
            value=str(self.project_dir / "Models" / "best.onnx")
        )

        self.conf_value = tk.DoubleVar(value=0.35)
        self.conf_value.trace_add("write", self._on_conf_changed)
        self.monitor_index = tk.IntVar(value=1)

        # Настройки производительности
        self.target_fps = tk.IntVar(value=20)
        self.resolution_mode = tk.StringVar(value="640x640")

        # Состояние приложения
        self._running = False
        self._starting = False
        self._stop_event = None
        self._error_count = 0
        self._max_errors = 10

        # Компоненты
        self._capturer = None
        self._detector = None
        self._smoother = None
        self._overlay = None
        self._fps_counter = None

        # Потоки и очереди
        self._frame_queue = None
        self._capture_thread = None
        self._pipeline_thread = None
        self._stats_thread = None

        self._build_ui()
        self._update_fps_label()
        self._check_model_exists()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_styling(self):
        """Применяет стили для улучшения внешнего вида"""
        style = {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'button_bg': '#3c3c3c',
            'button_active': '#4c4c4c',
            'entry_bg': '#3c3c3c',
            'select_color': '#5c5c5c'
        }

        try:
            self.root.configure(bg=style['bg'])

            # Пытаемся применить тему, если доступна
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    def _build_ui(self):
        pad = 8
        main_frame = tk.Frame(self.root, padx=12, pady=12)
        main_frame.pack(fill="both", expand=True)

        # Заголовок
        title = tk.Label(
            main_frame,
            text="Alcohol Censor",
            font=("Arial", 14, "bold")
        )
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        # Модель
        tk.Label(main_frame, text="Модель:").grid(row=1, column=0, sticky="w")
        self.model_entry = tk.Entry(main_frame, textvariable=self.model_path, width=45)
        self.model_entry.grid(row=1, column=1, padx=pad)
        tk.Button(
            main_frame,
            text="...",
            width=4,
            command=self._choose_model
        ).grid(row=1, column=2)

        # Порог уверенности
        tk.Label(main_frame, text="Порог:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.conf_scale = tk.Scale(
            main_frame,
            from_=0.10,
            to=0.90,
            resolution=0.05,
            orient="horizontal",
            variable=self.conf_value,
            length=260
        )
        self.conf_scale.grid(row=2, column=1, sticky="w", pady=(10, 0))
        self.conf_label = tk.Label(main_frame, textvariable=self.conf_value)
        self.conf_label.grid(row=2, column=2, sticky="w", pady=(10, 0))

        # Монитор и настройки
        tk.Label(main_frame, text="Монитор:").grid(row=3, column=0, sticky="w", pady=(10, 0))
        monitor_frame = tk.Frame(main_frame)
        monitor_frame.grid(row=3, column=1, sticky="w", pady=(10, 0))

        self.monitor_spin = tk.Spinbox(
            monitor_frame,
            from_=1,
            to=8,
            width=5,
            textvariable=self.monitor_index
        )
        self.monitor_spin.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(monitor_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_spin = tk.Spinbox(
            monitor_frame,
            from_=10,
            to=60,
            width=5,
            textvariable=self.target_fps
        )
        self.fps_spin.pack(side=tk.LEFT, padx=(5, 0))

        # Кнопка управления
        self.toggle_button = tk.Button(
            main_frame,
            text="Включить цензуру",
            width=24,
            height=2,
            command=self._toggle,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold")
        )
        self.toggle_button.grid(row=4, column=0, columnspan=3, pady=(18, 8))

        # Статус
        status_frame = tk.Frame(main_frame)
        status_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 5))

        self.status_label = tk.Label(status_frame, text="Статус: выключено")
        self.status_label.pack(side=tk.LEFT)

        self.fps_label = tk.Label(status_frame, text="FPS: --")
        self.fps_label.pack(side=tk.RIGHT)

        # Прогресс-бар для ресурсов (опционально)
        self.resource_label = tk.Label(
            main_frame,
            text="⚡ Готов к работе",
            fg="gray",
            font=("Arial", 8)
        )
        self.resource_label.grid(row=6, column=0, columnspan=3, sticky="w")

        # Информация
        self.info_label = tk.Label(
            main_frame,
            text="💡 Совет: Используйте модель ONNX для лучшей производительности",
            fg="#888888",
            font=("Arial", 8)
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
            self._check_model_exists()

    def _check_model_exists(self):
        """Проверяет существование файла модели и обновляет UI"""
        model = Path(self.model_path.get())
        if model.exists():
            self.info_label.config(
                text="✅ Модель найдена. Готов к работе.",
                fg="#4CAF50"
            )
        else:
            self.info_label.config(
                text="⚠️ Модель не найдена. Пожалуйста, выберите корректный файл модели.",
                fg="#FF9800"
            )

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
            self._check_model_exists()
            return

        try:
            self._starting = True
            self._set_controls_state("disabled")
            self.toggle_button.config(
                state="disabled",
                text="Запуск...",
                bg="#FF9800"
            )
            self.status_label.config(text="Статус: запускается...")
            self.root.update_idletasks()

            # Сброс счетчиков ошибок
            self._error_count = 0

            self._stop_event = threading.Event()
            self._frame_queue = queue.Queue(maxsize=3)  # Увеличен размер очереди

            self._fps_counter = FpsCounter()

            # Инициализация захвата экрана
            self._capturer = ScreenCapturer(
                monitor_index=int(self.monitor_index.get()),
                fps_limit=self.target_fps.get(),
            )
            self._capturer.start()

            monitor_rect = self._capturer.get_monitor_rect()
            screen_w = monitor_rect["width"]
            screen_h = monitor_rect["height"]
            monitor_left = monitor_rect["left"]
            monitor_top = monitor_rect["top"]

            # Определяем размер входного изображения
            if self.resolution_mode.get() == "640x640":
                input_size = 640
            else:
                input_size = 416

            # Инициализация детектора
            self._detector = BottleDetector(
                model_path=str(model),
                conf=float(self.conf_value.get()),
                device="auto",
            )

            # Инициализация трекера
            self._smoother = TrackSmoother(
                ema_alpha=0.6,
                decay_frames=6,
                min_confidence=float(self.conf_value.get()),
            )

            # Инициализация оверлея
            self._overlay = WindowsOverlay(screen_w, screen_h)
            self._overlay.start()

            # Запуск потоков
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="CaptureThread"
            )

            self._pipeline_thread = threading.Thread(
                target=self._pipeline_loop,
                args=(screen_w, screen_h, monitor_left, monitor_top, input_size),
                daemon=True,
                name="PipelineThread"
            )

            self._stats_thread = threading.Thread(
                target=self._stats_loop,
                daemon=True,
                name="StatsThread"
            )

            self._running = True
            self._capture_thread.start()
            self._pipeline_thread.start()
            self._stats_thread.start()

            self.toggle_button.config(
                state="normal",
                text="Выключить цензуру",
                bg="#F44336"
            )
            self.status_label.config(text="Статус: работает")

        except Exception as e:
            self._running = False
            self._set_controls_state("normal")
            self._safe_cleanup()
            messagebox.showerror("Ошибка запуска", f"Не удалось запустить цензуру:\n{str(e)}")
            self.toggle_button.config(state="normal", text="Включить цензуру", bg="#4CAF50")
        finally:
            self._starting = False

    def _stop_censorship(self):
        if self._starting:
            return

        self.status_label.config(text="Статус: выключается...")
        self.root.update_idletasks()

        self._running = False

        if self._stop_event:
            self._stop_event.set()

        # Останавливаем оверлей с задержкой для очистки
        if self._overlay:
            try:
                self._overlay.update_boxes([])
                time.sleep(0.1)
                self._overlay.stop()
            except Exception:
                pass

        # Останавливаем захват
        if self._capturer:
            try:
                self._capturer.stop()
            except Exception:
                pass

        # Сбрасываем трекеры
        if self._smoother:
            try:
                self._smoother.reset()
            except Exception:
                pass

        if self._detector:
            try:
                self._detector.reset_tracker()
            except Exception:
                pass

        # Очищаем очередь
        if self._frame_queue:
            try:
                while not self._frame_queue.empty():
                    self._frame_queue.get_nowait()
            except Exception:
                pass

        # Сброс компонентов
        self._capturer = None
        self._detector = None
        self._smoother = None
        self._overlay = None
        self._fps_counter = None
        self._frame_queue = None
        self._stop_event = None

        # Ожидание завершения потоков (с таймаутом)
        for thread in [self._capture_thread, self._pipeline_thread, self._stats_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=0.5)

        self._capture_thread = None
        self._pipeline_thread = None
        self._stats_thread = None

        self.toggle_button.config(text="Включить цензуру", bg="#4CAF50")
        self.status_label.config(text="Статус: выключено")
        self.fps_label.config(text="FPS: --")
        self.resource_label.config(text="⚡ Цензура остановлена", fg="gray")
        self._set_controls_state("normal")

    def _capture_loop(self):
        """Поток захвата экрана"""
        frame_count = 0
        last_log_time = time.time()

        while self._running and self._stop_event and not self._stop_event.is_set():
            try:
                frame = self._capturer.grab()

                if frame is None:
                    continue

                # Управление очередью
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                self._frame_queue.put(frame)
                frame_count += 1

                # Логирование производительности каждые 5 секунд
                if time.time() - last_log_time > 5:
                    queue_size = self._frame_queue.qsize()
                    if queue_size > 1:
                        print(f"Размер очереди кадров: {queue_size}")
                    last_log_time = time.time()

            except Exception as e:
                self._error_count += 1
                if self._error_count > self._max_errors:
                    self.root.after(0, self._show_worker_error, f"Критическая ошибка захвата: {e}")
                    break
                elif self._error_count % 5 == 0:
                    self.root.after(0, self._show_worker_error, f"Ошибка захвата экрана: {e}")

    def _pipeline_loop(self, screen_w, screen_h, monitor_left, monitor_top, input_size):
        """Поток обработки детекции"""
        process_times = []

        while self._running and self._stop_event and not self._stop_event.is_set():
            try:
                process_start = time.time()

                try:
                    frame = self._frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                # Ресайз с сохранением пропорций
                small_frame = cv2.resize(frame, (input_size, input_size),
                                         interpolation=cv2.INTER_LINEAR)

                # Детекция
                detections = self._detector.track(small_frame)

                # Сглаживание
                boxes = self._smoother.update(
                    detections=detections,
                    screen_w=screen_w,
                    screen_h=screen_h,
                    monitor_left=monitor_left,
                    monitor_top=monitor_top,
                    input_w=input_size,
                    input_h=input_size,
                )

                # Обновление оверлея
                if self._overlay and self._running:
                    self._overlay.update_boxes(boxes)

                # Обновление FPS
                if self._fps_counter:
                    self._fps_counter.tick()

                # Отслеживание времени обработки
                process_time = time.time() - process_start
                process_times.append(process_time)
                if len(process_times) > 30:
                    process_times.pop(0)

                # Сброс счетчика ошибок при успешной обработке
                self._error_count = max(0, self._error_count - 1)

            except Exception as e:
                self._error_count += 1
                if self._error_count > self._max_errors:
                    self.root.after(0, self._show_worker_error, f"Критическая ошибка обработки: {e}")
                    break
                elif self._error_count % 5 == 0:
                    self.root.after(0, self._show_worker_error, f"Ошибка обработки кадра: {e}")

    def _stats_loop(self):
        """Поток для сбора статистики производительности"""
        while self._running and self._stop_event and not self._stop_event.is_set():
            try:
                if self._fps_counter:
                    fps = self._fps_counter.get()

                    # Обновление UI с информацией о производительности
                    if fps < 15:
                        status = "⚠️ Низкая производительность"
                        color = "#FF9800"
                    elif fps > 25:
                        status = "⚡ Отличная производительность"
                        color = "#4CAF50"
                    else:
                        status = "✅ Нормальная работа"
                        color = "#2196F3"

                    self.root.after(0, lambda: self.resource_label.config(
                        text=status,
                        fg=color
                    ))

                time.sleep(2)  # Обновление каждые 2 секунды

            except Exception:
                pass

    def _update_fps_label(self):
        """Обновление отображения FPS в UI"""
        if self._running and self._fps_counter:
            fps = self._fps_counter.get()
            fps_text = f"FPS: {fps:.1f}"

            # Цветовая индикация FPS
            if fps < 15:
                fps_text = f"⚠️ {fps_text}"
            elif fps > 25:
                fps_text = f"⚡ {fps_text}"

            self.fps_label.config(text=fps_text)

        self.root.after(1000, self._update_fps_label)

    def _show_worker_error(self, text):
        """Показ ошибки из рабочих потоков"""
        if self._running:
            self._error_count += 1
            if self._error_count > self._max_errors:
                messagebox.showerror("Критическая ошибка",
                                     f"Произошло слишком много ошибок.\n{text}\n\nЦензура будет остановлена.")
                self._stop_censorship()
            else:
                self.status_label.config(text=f"Статус: ошибка ({self._error_count}/{self._max_errors})")
                self.root.after(3000, lambda: self.status_label.config(text="Статус: работает"))

    def _safe_cleanup(self):
        """Безопасная очистка ресурсов"""
        # Остановка оверлея
        if self._overlay:
            try:
                self._overlay.update_boxes([])
                self._overlay.stop()
            except Exception:
                pass

        # Остановка захвата
        if self._capturer:
            try:
                self._capturer.stop()
            except Exception:
                pass

        # Сброс трекеров
        if self._smoother:
            try:
                self._smoother.reset()
            except Exception:
                pass

        if self._detector:
            try:
                self._detector.reset_tracker()
            except Exception:
                pass

        # Очистка компонентов
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
        """Установка состояния элементов управления"""
        states = {"normal": "normal", "disabled": "disabled"}
        ui_state = states.get(state, "disabled")

        self.model_entry.config(state=ui_state)
        self.conf_scale.config(state=ui_state)
        self.monitor_spin.config(state=ui_state)
        self.fps_spin.config(state=ui_state)

    def _on_conf_changed(self, *_):
        """Обработка изменения порога уверенности"""
        conf = float(self.conf_value.get())
        if self._detector is not None:
            self._detector.set_conf(conf)
        if self._smoother is not None:
            self._smoother.set_params(min_confidence=conf)

    def _on_close(self):
        """Обработка закрытия окна"""
        if self._running:
            self._stop_censorship()
        self.root.destroy()


def main():
    """Точка входа в приложение"""
    try:
        root = tk.Tk()

        # Центрирование окна
        window_width = 500
        window_height = 450
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        app = CensorApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Критическая ошибка", f"Не удалось запустить приложение:\n{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()