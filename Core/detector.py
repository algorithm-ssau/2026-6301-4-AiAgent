import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

# Настройка логгера для диагностики (по умолчанию тихий)
_logger = logging.getLogger(__name__)


def _select_device() -> str:
    """Автоопределение лучшего устройства для PyTorch-моделей (.pt)."""
    try:
        import torch

        # CUDA (NVIDIA)
        if torch.cuda.is_available():
            _logger.debug("PyTorch: CUDA available")
            return "cuda"

        # Intel XPU (Arc, PyTorch 2.4+)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            _logger.debug("PyTorch: Intel XPU available")
            return "xpu"

        # Apple Metal (MPS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _logger.debug("PyTorch: Apple MPS available")
            return "mps"

    except ImportError:
        pass
    return "cpu"


def _get_ort_providers() -> List[Tuple[str, dict]]:
    """Возвращает список провайдеров ONNX Runtime в порядке приоритета (GPU → CPU)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return []

    available = ort.get_available_providers()
    priority = []

    # Приоритет: TensorRT > CUDA > CoreML > DirectML > CPU
    for provider in ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                     "CoreMLExecutionProvider", "DirectMLExecutionProvider"]:
        if provider in available:
            _logger.debug(f"ONNX Runtime: {provider} available")
            priority.append((provider, {}))

    # CPU всегда доступен
    if "CPUExecutionProvider" in available:
        priority.append(("CPUExecutionProvider", {}))

    return priority


def _patch_ort_speed(providers: Optional[List[Tuple[str, dict]]] = None):
    """
    Патчит ONNX Runtime для использования оптимизированных настроек и GPU-провайдеров.
    Вызывается ДО создания YOLO модели.
    """
    try:
        import onnxruntime as ort

        if getattr(ort.InferenceSession, "_optimized_patched", False):
            return

        if providers is None:
            providers = _get_ort_providers()

        original_session = ort.InferenceSession

        class _OptimizedSession(original_session):
            _optimized_patched = True

            def __init__(self, path, sess_options=None, providers=None,
                         provider_options=None, **kwargs):
                # Создаём оптимальные SessionOptions, если не переданы
                if sess_options is None:
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    # Дополнительные оптимизации: использование потоков
                    sess_options.intra_op_num_threads = 4
                    sess_options.inter_op_num_threads = 2

                # Переданные провайдеры или глобальные
                if providers is None:
                    providers = _get_ort_providers()
                    # Преобразуем в список строк, если нужно (совместимость)
                    if providers and isinstance(providers[0], tuple):
                        provider_names = [p[0] for p in providers]
                    else:
                        provider_names = providers
                else:
                    provider_names = providers

                super().__init__(path, sess_options, providers=provider_names,
                                 provider_options=provider_options, **kwargs)

        ort.InferenceSession = _OptimizedSession
        _logger.info(f"ONNX Runtime patched with providers: {providers}")

    except Exception as e:
        _logger.warning(f"Failed to patch ONNX Runtime: {e}")


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    track_id: int | None


class BottleDetector:
    """
    Детектор бутылок с поддержкой GPU для PyTorch (.pt) и ONNX (.onnx).
    Автоматически выбирает CUDA/TensorRT/DirectML где возможно.
    """

    def __init__(self, model_path: str, conf: float = 0.35, device: str = "auto"):
        """
        :param model_path: путь к модели .pt или .onnx
        :param conf: порог уверенности
        :param device: "auto", "cuda", "cpu", "xpu", "mps" (для .pt)
                       для ONNX игнорируется, GPU определяется через ONNX Runtime провайдеры
        """
        self._conf = float(conf)
        self._is_onnx = str(model_path).lower().endswith(".onnx")

        # Важное изменение: улучшенное определение устройства и поддержка GPU для ONNX
        if self._is_onnx:
            # Патчим ONNX Runtime с автовыбором лучших провайдеров (GPU → CPU)
            _patch_ort_speed()
            self._device = None      # device не используется для ONNX в predict
            self._classes = None     # своя модель, классы не фильтруем
            _logger.info("ONNX model loaded. GPU acceleration will be used if available.")
        else:
            # PyTorch модель: выбираем устройство
            if device == "auto":
                self._device = _select_device()
            else:
                self._device = device

            self._classes = [39]     # COCO: бутылка = класс 39
            _logger.info(f"PyTorch model loaded. Device: {self._device}")

        # Загружаем модель
        self._model = YOLO(model_path, task="detect")

    def track(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Выполняет детекцию (без встроенного трекинга Ultralytics).
        Возвращает список Detection с track_id = None (трекинг выполняется внешним TrackSmoother).
        """
        return self.detect(frame_bgr)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Базовый метод детекции."""
        kwargs = {
            "conf": self._conf,
            "classes": self._classes,
            "verbose": False,
        }
        # Для PyTorch модели передаём device, для ONNX - нет
        if not self._is_onnx and self._device is not None:
            kwargs["device"] = self._device

        results = self._model.predict(frame_bgr, **kwargs)
        return self._parse(results)

    def set_conf(self, conf: float) -> None:
        """Обновляет порог уверенности для следующих кадров."""
        self._conf = float(conf)

    def reset_tracker(self) -> None:
        """Сбрасывает внутреннее состояние трекера (если используется)."""
        predictor = getattr(self._model, "predictor", None)
        if predictor and getattr(predictor, "trackers", None):
            for tr in predictor.trackers:
                if hasattr(tr, "reset"):
                    tr.reset()
            _logger.debug("Tracker reset")

    @staticmethod
    def _parse(results) -> List[Detection]:
        """Преобразует результаты YOLO в список Detection."""
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # Оптимизация: работаем напрямую с тензорами
        xyxy_tensor = boxes.xyxy
        conf_tensor = boxes.conf

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy_tensor[i].tolist()
            detections.append(Detection(
                x1=int(x1), y1=int(y1),
                x2=int(x2), y2=int(y2),
                conf=float(conf_tensor[i]),
                track_id=None
            ))
        return detections