"""
Модуль сглаживания треков для стабилизации прямоугольников между кадрами.

Принимает сырые детекции от нейросети (в координатах 640x640),
преобразует их в экранные координаты и применяет EMA-фильтрацию для
плавности движения. Также реализует механизм decay для временно
пропавших объектов.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import sys
import os

# Импортируем реальный Detection от Участника 2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core.detector import Detection


@dataclass
class _TrackState:
    """Внутреннее состояние одного трека."""
    x1: float  # сглаженные координаты (floats для точности EMA)
    y1: float
    x2: float
    y2: float
    age: int  # сколько кадров объект не появлялся
    track_id: int  # ID трека


class TrackSmoother:
    """
    Сглаживание треков с помощью EMA и обратной проекцией в экранные координаты.

    Args:
        ema_alpha: Коэффициент сглаживания (0.0 - очень медленно, 1.0 - мгновенно)
        decay_frames: Сколько кадров держать прямоугольник после пропажи объекта
    """

    def __init__(self, ema_alpha: float = 0.6, decay_frames: int = 10):
        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha должен быть в [0, 1], получено {ema_alpha}")
        if decay_frames < 0:
            raise ValueError(f"decay_frames должен быть >= 0, получено {decay_frames}")

        self.ema_alpha = ema_alpha
        self.decay_frames = decay_frames
        self._tracks: Dict[int, _TrackState] = {}  # track_id -> состояние

    def update(
            self,
            detections: List[Detection],
            screen_w: int,
            screen_h: int,
            monitor_left: int = 0,
            monitor_top: int = 0,
            input_w: int = 640,
            input_h: int = 640
    ) -> List[Tuple[int, int, int, int]]:
        """
        Обновить треки новыми детекциями и получить сглаженные прямоугольники.

        Args:
            detections: Список детекций от нейросети (в координатах input_w x input_h)
            screen_w, screen_h: Размер экрана в пикселях
            monitor_left, monitor_top: Смещение монитора от левого верхнего угла
            input_w, input_h: Размер входного изображения для нейросети (обычно 640x640)

        Returns:
            Список прямоугольников (x1, y1, x2, y2) в экранных координатах
        """
        # Шаг 1: Вычислить масштабные коэффициенты
        scale_x = screen_w / input_w
        scale_y = screen_h / input_h

        # Шаг 2: Пометить все существующие треки как "не обновлённые" в этом кадре
        # Увеличиваем age у всех, потом обнулим у тех, что появились
        for track in self._tracks.values():
            track.age += 1

        # Шаг 3: Обработать новые детекции
        for det in detections:
            if det.track_id is None:
                continue  # Игнорируем детекции без track_id

            # Проецируем в экранные координаты
            x1_screen = int(det.x1 * scale_x) + monitor_left
            y1_screen = int(det.y1 * scale_y) + monitor_top
            x2_screen = int(det.x2 * scale_x) + monitor_left
            y2_screen = int(det.y2 * scale_y) + monitor_top

            # Обновляем или создаём трек
            if det.track_id in self._tracks:
                # Существующий трек — применяем EMA
                track = self._tracks[det.track_id]
                track.x1 = self.ema_alpha * x1_screen + (1 - self.ema_alpha) * track.x1
                track.y1 = self.ema_alpha * y1_screen + (1 - self.ema_alpha) * track.y1
                track.x2 = self.ema_alpha * x2_screen + (1 - self.ema_alpha) * track.x2
                track.y2 = self.ema_alpha * y2_screen + (1 - self.ema_alpha) * track.y2
                track.age = 0  # сброс счётчика пропусков
            else:
                # Новый трек — инициализируем
                self._tracks[det.track_id] = _TrackState(
                    x1=float(x1_screen),
                    y1=float(y1_screen),
                    x2=float(x2_screen),
                    y2=float(y2_screen),
                    age=0,
                    track_id=det.track_id
                )

        # Шаг 4: Удалить старые треки (превысившие decay_frames)
        to_delete = [tid for tid, track in self._tracks.items()
                     if track.age > self.decay_frames]
        for tid in to_delete:
            del self._tracks[tid]

        # Шаг 5: Сформировать выходной список (включая треки с age > 0)
        result = []
        for track in self._tracks.values():
            # Округляем float координаты до int для отрисовки
            result.append((
                int(round(track.x1)),
                int(round(track.y1)),
                int(round(track.x2)),
                int(round(track.y2))
            ))

        return result

    def reset(self) -> None:
        """Полностью сбросить все треки."""
        self._tracks.clear()

    def get_track_count(self) -> int:
        """Вернуть количество активных треков (для отладки)."""
        return len(self._tracks)

    def get_track_ids(self) -> List[int]:
        """Вернуть список активных track_id (для отладки)."""
        return list(self._tracks.keys())


# Для обратной совместимости и удобства импорта
Box = Tuple[int, int, int, int]