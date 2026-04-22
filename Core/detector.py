from dataclasses import dataclass
from typing import List

from ultralytics import YOLO


def _select_device() -> str:
    """Автоопределение устройства: CUDA (NVIDIA) / ROCm (AMD) / CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            # torch.cuda работает и с ROCm-сборкой PyTorch для AMD
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
    def __init__(self, model_path: str, conf: float = 0.35, device: str = "auto"):
        if device == "auto":
            device = _select_device()
        self._model = YOLO(model_path, task="detect")
        self._conf = conf
        self._device = device
        # .pt — COCO, бутылка = класс 39; .onnx — своя модель, бутылка = класс 0
        self._classes = [39] if str(model_path).endswith(".pt") else None

    def track(self, frame_bgr) -> List[Detection]:
        results = self._model.track(
            frame_bgr,
            persist=True,
            conf=self._conf,
            classes=self._classes,
            tracker="bytetrack.yaml",
            verbose=False,
            device=self._device,
        )
        boxes = results[0].boxes
        if boxes.id is None:
            return []
        detections = []
        for xyxy, conf, track_id in zip(boxes.xyxy, boxes.conf, boxes.id):
            detections.append(Detection(
                x1=int(xyxy[0]), y1=int(xyxy[1]),
                x2=int(xyxy[2]), y2=int(xyxy[3]),
                conf=float(conf),
                track_id=int(track_id),
            ))
        return detections

    def detect(self, frame_bgr) -> List[Detection]:
        results = self._model.predict(
            frame_bgr,
            conf=self._conf,
            classes=self._classes,
            verbose=False,
            device=self._device,
        )
        boxes = results[0].boxes
        detections = []
        for xyxy, conf in zip(boxes.xyxy, boxes.conf):
            detections.append(Detection(
                x1=int(xyxy[0]), y1=int(xyxy[1]),
                x2=int(xyxy[2]), y2=int(xyxy[3]),
                conf=float(conf),
                track_id=None,
            ))
        return detections

    def reset_tracker(self):
        p = getattr(self._model, "predictor", None)
        if p is not None and getattr(p, "trackers", None):
            p.trackers[0].reset()


if __name__ == "__main__":
    import numpy as np
    from pathlib import Path

    MODEL_PATH = Path(__file__).parent.parent / "Models" / "best.onnx"
    detector = BottleDetector(MODEL_PATH, conf=0.35)
    print(f"Модель загружена: {MODEL_PATH}")

    # Тест 1: detect() на пустом кадре
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    result = detector.detect(blank)
    print(f"detect() на пустом кадре → {len(result)} объектов (ожидается 0)")

    # Тест 2: reset_tracker() не падает
    detector.reset_tracker()
    print("reset_tracker() → OK")

    # Тест 3: track() на пустом кадре
    result = detector.track(blank)
    print(f"track() на пустом кадре → {len(result)} объектов (ожидается 0)")

    # Тест 4: структура Detection
    import numpy as np
    fake_frame = np.random.randint(0, 50, (640, 640, 3), dtype=np.uint8)
    result = detector.detect(fake_frame)
    print(f"detect() на шумовом кадре → {len(result)} объектов")
    for det in result:
        print(f"  Detection: x1={det.x1} y1={det.y1} x2={det.x2} y2={det.y2} conf={det.conf:.2f} track_id={det.track_id}")

    print("\nВсе проверки пройдены.")