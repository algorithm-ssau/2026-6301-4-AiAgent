from dataclasses import dataclass
from typing import List

from ultralytics import YOLO


def _select_device() -> str:
    """Автоопределение устройства для PyTorch-моделей (.pt)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel Arc (PyTorch 2.4+)
            return "xpu"
    except ImportError:
        pass
    return "cpu"


def _patch_ort_speed():
    """Инжектирует оптимизированные SessionOptions в ORT до загрузки модели Ultralytics.

    Ultralytics создаёт InferenceSession без SessionOptions (дефолты ORT).
    С ORT_ENABLE_ALL граф оптимизируется один раз при загрузке → быстрее инференс.
    """
    try:
        import onnxruntime as ort
        if getattr(ort.InferenceSession, "_speed_patched", False):
            return

        _Orig = ort.InferenceSession

        class _FastSession(_Orig):
            _speed_patched = True

            def __init__(self, path, sess_options=None,
                         providers=None, provider_options=None, **kw):
                if sess_options is None:
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                super().__init__(path, sess_options, providers, provider_options, **kw)

        ort.InferenceSession = _FastSession
    except Exception:
        pass


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
        self._conf = conf
        is_onnx = str(model_path).endswith(".onnx")

        if is_onnx:
            _patch_ort_speed()          # до YOLO(), чтобы патч попал в сессию
            self._device = "cpu"        # DML не работает с этой моделью
            self._classes = None        # своя модель, без фильтра по COCO
        else:
            if device == "auto":
                device = _select_device()
            self._device = device
            self._classes = [39]        # COCO: бутылка = класс 39

        self._model = YOLO(model_path, task="detect")

    def track(self, frame_bgr) -> List[Detection]:
        # predict() вместо track(): пропускаем ByteTrack (TrackSmoother делает tracking сам),
        # экономим ~10 ms/кадр на его overhead.
        results = self._model.predict(
            frame_bgr,
            conf=self._conf,
            classes=self._classes,
            verbose=False,
            device=self._device,
        )
        return self._parse(results)

    def detect(self, frame_bgr) -> List[Detection]:
        results = self._model.predict(
            frame_bgr,
            conf=self._conf,
            classes=self._classes,
            verbose=False,
            device=self._device,
        )
        return self._parse(results)

    def set_conf(self, conf: float) -> None:
        """Обновить порог без перезапуска — действует со следующего кадра."""
        self._conf = float(conf)

    @staticmethod
    def _parse(results) -> List[Detection]:
        boxes = results[0].boxes
        return [
            Detection(
                x1=int(xyxy[0]), y1=int(xyxy[1]),
                x2=int(xyxy[2]), y2=int(xyxy[3]),
                conf=float(conf),
                track_id=None,  # TrackSmoother присваивает ID через IoU-матчинг
            )
            for xyxy, conf in zip(boxes.xyxy, boxes.conf)
        ]

    def reset_tracker(self):
        p = getattr(self._model, "predictor", None)
        if p and getattr(p, "trackers", None):
            p.trackers[0].reset()
