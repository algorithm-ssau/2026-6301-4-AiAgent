import numpy as np
from Core.detector import BottleDetector, Detection

MODEL = "Models/best.onnx"


def test_detect_returns_list():
    detector = BottleDetector(MODEL)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert isinstance(result, list)


def test_detect_track_id_is_none():
    detector = BottleDetector(MODEL)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    for det in detector.detect(frame):
        assert det.track_id is None


def test_reset_tracker_does_not_raise():
    detector = BottleDetector(MODEL)
    detector.reset_tracker()
