import numpy as np
import pytest
from Core.detector import BottleDetector, Detection, _select_device

MODEL = "Models/best.onnx"


def test_detect_returns_list():
    detector = BottleDetector(MODEL, device="cpu")
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert isinstance(result, list)


def test_detect_track_id_is_none():
    detector = BottleDetector(MODEL, device="cpu")
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    for det in detector.detect(frame):
        assert det.track_id is None


def test_reset_tracker_does_not_raise():
    detector = BottleDetector(MODEL, device="cpu")
    detector.reset_tracker()


def test_select_device_returns_string():
    device = _select_device()
    assert device in ("cuda", "cpu")


def test_detector_device_auto():
    """Автовыбор устройства не должен падать (NVIDIA/AMD/Intel/CPU)."""
    detector = BottleDetector(MODEL, device="auto")
    # .onnx: "dml" | "cuda" | "cpu"  (ONNX Runtime провайдер)
    # .pt:   "cuda" | "xpu" | "cpu"  (PyTorch устройство)
    assert isinstance(detector._device, str) and detector._device


def test_detector_device_cpu_explicit():
    detector = BottleDetector(MODEL, device="cpu")
    assert detector._device == "cpu"
