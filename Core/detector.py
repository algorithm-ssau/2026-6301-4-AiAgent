from dataclasses import dataclass
from typing import List, Optional
import cv2
import time
import numpy as np


def _select_device() -> str:
    """Автоопределение устройства: CUDA (NVIDIA) / ROCm (AMD) / CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    track_id: int | None


class BottleDetector:
    def __init__(self, model_path: str, conf: float = 0.25, device: str = "auto",
                 input_size: int = 640):  # Фиксированный размер для ONNX модели
        if device == "auto":
            device = _select_device()

        from ultralytics import YOLO

        self._model = YOLO(model_path, task="detect")
        self._conf = conf
        self._device = device
        # ВАЖНО: ONNX модель ожидает 640, игнорируем input_size
        self._input_size = 640  # Фиксированный размер для модели

        # Для кастомной модели используем класс 0
        self._classes = None if str(model_path).endswith(".onnx") else [39]

        # Оптимизации для скорости
        if device == "cuda":
            self._model.to(device)

        # Кэш для последних результатов
        self._last_detections = []
        self._last_time = 0
        self._cache_timeout = 0.033  # ~30 FPS

        # Параметры для NMS
        self._iou = 0.45

    def track(self, frame_bgr, use_imgsz: Optional[int] = None) -> List[Detection]:
        """Оптимизированный трекинг с фиксированным размером 640x640 для ONNX"""
        current_time = time.time()

        # Кэширование для стабильных сцен
        if current_time - self._last_time < self._cache_timeout:
            return self._last_detections

        h, w = frame_bgr.shape[:2]

        # Всегда ресайзим до 640x640 для ONNX модели
        target_size = 640

        # Ресайз с сохранением пропорций и добавлением паддинга
        frame_resized = self._resize_with_padding(frame_bgr, target_size, target_size)

        # Запускаем инференс
        results = self._model(
            frame_resized,
            conf=self._conf,
            iou=self._iou,
            classes=self._classes,
            verbose=False,
            device=self._device,
            imgsz=target_size,
            half=True if self._device == "cuda" else False,
        )

        # Пост-обработка с преобразованием координат обратно в исходное разрешение
        detections = self._process_results(results, h, w, target_size)

        self._last_detections = detections
        self._last_time = current_time

        return detections

    def _resize_with_padding(self, img, target_w, target_h):
        """Ресайз изображения с сохранением пропорций и добавлением паддинга"""
        h, w = img.shape[:2]

        # Вычисляем масштаб
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Ресайз
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Создаем черный фон
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Центрируем изображение
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return padded

    def _process_results(self, results, original_h, original_w, target_size):
        """Обработка результатов с преобразованием координат"""
        detections = []
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            for i, (xyxy, conf) in enumerate(zip(boxes.xyxy, boxes.conf)):
                if conf < self._conf:
                    continue

                # Координаты в размере 640x640 (с учетом паддинга)
                x1_model = float(xyxy[0])
                y1_model = float(xyxy[1])
                x2_model = float(xyxy[2])
                y2_model = float(xyxy[3])

                # Преобразуем обратно в координаты исходного изображения
                x1, y1, x2, y2 = self._map_coordinates_back(
                    x1_model, y1_model, x2_model, y2_model,
                    original_h, original_w, target_size
                )

                # Проверка валидности
                if x1 >= x2 or y1 >= y2:
                    continue

                # Фильтр слишком маленьких объектов
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                # Получаем track_id если есть
                track_ids = boxes.id
                track_id = None
                if track_ids is not None and i < len(track_ids):
                    track_id = int(track_ids[i])

                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    conf=float(conf),
                    track_id=track_id,
                ))

        return detections

    def _map_coordinates_back(self, x1, y1, x2, y2, original_h, original_w, target_size):
        """Преобразование координат из 640x640 обратно в исходное разрешение"""
        # Параметры паддинга
        scale = min(target_size / original_w, target_size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2

        # Убираем паддинг
        x1 = x1 - x_offset
        y1 = y1 - y_offset
        x2 = x2 - x_offset
        y2 = y2 - y_offset

        # Масштабируем обратно
        scale_back_x = original_w / new_w
        scale_back_y = original_h / new_h

        x1 = int(max(0, min(original_w, x1 * scale_back_x)))
        y1 = int(max(0, min(original_h, y1 * scale_back_y)))
        x2 = int(max(0, min(original_w, x2 * scale_back_x)))
        y2 = int(max(0, min(original_h, y2 * scale_back_y)))

        return x1, y1, x2, y2

    def detect(self, frame_bgr) -> List[Detection]:
        """Быстрая детекция без трекинга"""
        return self.track(frame_bgr)

    def reset_tracker(self):
        """Сброс трекера"""
        p = getattr(self._model, "predictor", None)
        if p is not None and getattr(p, "trackers", None):
            try:
                p.trackers[0].reset()
            except Exception:
                pass

    def set_params(self, conf: Optional[float] = None, iou: Optional[float] = None):
        """Динамическое изменение параметров"""
        if conf is not None:
            self._conf = conf
        if iou is not None:
            self._iou = iou