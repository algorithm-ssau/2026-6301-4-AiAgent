"""
Максимально упрощенный трекер - только преобразование координат
Вся логика удержания теперь в оверлее
"""

from typing import List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core.detector import Detection


class TrackSmoother:
    """
    Упрощенный трекер - только преобразует координаты.
    Удержание и стабилизация - в оверлее.
    """

    def __init__(self, ema_alpha: float = 0.5, min_confidence: float = 0.20, **kwargs):
        self.ema_alpha = ema_alpha
        self.min_confidence = min_confidence
        self._last_boxes = []

    def update(self, detections: List[Detection], screen_w: int, screen_h: int,
               monitor_left: int = 0, monitor_top: int = 0,
               input_w: Optional[int] = None, input_h: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
        """
        Простое преобразование детекций в экранные координаты
        """
        # Вычисляем масштаб
        if input_w and input_h:
            scale_x = screen_w / input_w
            scale_y = screen_h / input_h
        else:
            scale_x = scale_y = 1.0

        result = []
        for det in detections:
            if det.conf < self.min_confidence:
                continue

            x1 = int(det.x1 * scale_x) + monitor_left
            y1 = int(det.y1 * scale_y) + monitor_top
            x2 = int(det.x2 * scale_x) + monitor_left
            y2 = int(det.y2 * scale_y) + monitor_top

            # Нормализуем координаты
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Проверка минимального размера
            if (x2 - x1) < 15 or (y2 - y1) < 15:
                continue

            # Клиппинг
            x1 = max(monitor_left, min(monitor_left + screen_w, x1))
            y1 = max(monitor_top, min(monitor_top + screen_h, y1))
            x2 = max(monitor_left, min(monitor_left + screen_w, x2))
            y2 = max(monitor_top, min(monitor_top + screen_h, y2))

            if x1 < x2 and y1 < y2:
                result.append((x1, y1, x2, y2))

        return result

    def reset(self):
        self._last_boxes = []