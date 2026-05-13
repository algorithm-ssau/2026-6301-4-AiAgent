from dataclasses import dataclass
from typing import List, Optional
import cv2
import time
import numpy as np


def _select_device() -> str:
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
    def __init__(self, model_path: str, conf: float = 0.20, device: str = "auto"):
        if device == "auto":
            device = _select_device()

        from ultralytics import YOLO

        self._model = YOLO(model_path, task="detect")
        self._conf = conf
        self._device = device

        # Определяем классы
        if str(model_path).endswith(".pt"):
            self._classes = [39]  # COCO bottle class
        else:
            self._classes = None  # Кастомная модель

        self._iou = 0.45
        self._last_time = 0
        self._last_detections = []

    def track(self, frame_bgr) -> List[Detection]:
        current_time = time.time()

        # Небольшое кэширование для стабильности
        if current_time - self._last_time < 0.033:  # ~30 FPS
            return self._last_detections

        h, w = frame_bgr.shape[:2]

        # Ресайз до 640 (модель ожидает такой размер)
        target_size = 640
        frame_resized = cv2.resize(frame_bgr, (target_size, target_size))

        results = self._model(
            frame_resized,
            conf=self._conf,
            iou=self._iou,
            classes=self._classes,
            verbose=False,
            device=self._device,
            imgsz=target_size,
        )

        detections = []
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            scale_x = w / target_size
            scale_y = h / target_size

            for xyxy, conf in zip(boxes.xyxy, boxes.conf):
                if conf < self._conf:
                    continue

                x1 = int(xyxy[0] * scale_x)
                y1 = int(xyxy[1] * scale_y)
                x2 = int(xyxy[2] * scale_x)
                y2 = int(xyxy[3] * scale_y)

                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    conf=float(conf), track_id=None
                ))

        self._last_detections = detections
        self._last_time = current_time

        return detections

    def detect(self, frame_bgr) -> List[Detection]:
        return self.track(frame_bgr)

    def reset_tracker(self):
        pass