"""
Модуль сглаживания треков для стабилизации прямоугольников между кадрами.
Оптимизированная версия с устранением мигания.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import sys
import os
import time
from collections import deque

# Импортируем реальный Detection
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core.detector import Detection


@dataclass
class _TrackState:
    """Внутреннее состояние одного трека."""
    x1: float
    y1: float
    x2: float
    y2: float
    age: int  # сколько кадров объект не появлялся
    track_id: int
    last_update: float = field(default_factory=time.time)
    confidence: float = 0.0  # последняя уверенность
    history: deque = field(default_factory=lambda: deque(maxlen=5))  # история позиций


class TrackSmoother:
    """
    Сглаживание треков с устранением мигания.
    """

    def __init__(self, ema_alpha: float = 0.7, decay_frames: int = 5,
                 min_confidence: float = 0.25, max_tracks: int = 50,
                 min_box_size: int = 20):
        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha должен быть в [0, 1], получено {ema_alpha}")
        if decay_frames < 0:
            raise ValueError(f"decay_frames должен быть >= 0, получено {decay_frames}")

        self.ema_alpha = ema_alpha
        self.decay_frames = decay_frames  # Увеличено для лучшего удержания
        self.min_confidence = min_confidence
        self.max_tracks = max_tracks
        self.min_box_size = min_box_size

        self._tracks: Dict[int, _TrackState] = {}
        self._last_boxes = []  # Кэш последних боксов для стабильности

        # Статистика
        self._total_tracks_created = 0
        self._total_tracks_removed = 0
        self._frames_processed = 0

    def _apply_ema(self, new_value: float, old_value: float) -> float:
        """Применить EMA-фильтр с приоритетом новым значениям"""
        return self.ema_alpha * new_value + (1 - self.ema_alpha) * old_value

    def _apply_smart_smoothing(self, new_box: Tuple[float, float, float, float],
                                old_box: Tuple[float, float, float, float],
                                confidence: float) -> Tuple[float, float, float, float]:
        """Умное сглаживание с учетом уверенности"""
        if confidence < 0.5:
            # Низкая уверенность - больше доверия старому значению
            alpha = self.ema_alpha * (confidence / 0.5)
        else:
            alpha = self.ema_alpha

        x1 = alpha * new_box[0] + (1 - alpha) * old_box[0]
        y1 = alpha * new_box[1] + (1 - alpha) * old_box[1]
        x2 = alpha * new_box[2] + (1 - alpha) * old_box[2]
        y2 = alpha * new_box[3] + (1 - alpha) * old_box[3]

        return x1, y1, x2, y2

    def _find_matching_track(self, box: Tuple[float, float, float, float],
                             updated_tracks: set) -> Optional[int]:
        """Найти подходящий трек по IoU и расстоянию"""
        best_track_id = None
        best_iou = 0.0
        best_distance = float("inf")

        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        max_distance = max(100.0, (box_w ** 2 + box_h ** 2) ** 0.5 * 0.5)

        for track_id, track in self._tracks.items():
            if track_id in updated_tracks:
                continue

            track_box = (track.x1, track.y1, track.x2, track.y2)

            # Вычисляем IoU
            iou = self._calculate_iou(box, track_box)

            # Вычисляем расстояние между центрами
            box_center_x = (box[0] + box[2]) / 2
            box_center_y = (box[1] + box[3]) / 2
            track_center_x = (track.x1 + track.x2) / 2
            track_center_y = (track.y1 + track.y2) / 2
            distance = ((box_center_x - track_center_x) ** 2 +
                       (box_center_y - track_center_y) ** 2) ** 0.5

            # Комбинированная метрика
            if iou > 0.2:
                score = iou
            else:
                score = 1.0 / (distance + 1.0) * 0.5

            if score > best_iou:
                best_track_id = track_id
                best_iou = score
                best_distance = distance

        if best_track_id and best_iou > 0.1:
            return best_track_id

        return None

    def _calculate_iou(self, box1: Tuple[float, float, float, float],
                       box2: Tuple[float, float, float, float]) -> float:
        """Вычислить Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, detections: List[Detection], screen_w: int, screen_h: int,
               monitor_left: int = 0, monitor_top: int = 0,
               input_w: Optional[int] = None, input_h: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
        """
        Обновить треки новыми детекциями и получить сглаженные прямоугольники.
        Улучшенная версия с устранением мигания.
        """
        self._frames_processed += 1

        # Вычисляем масштаб
        if input_w is None or input_h is None:
            scale_x = scale_y = 1.0
        else:
            scale_x = screen_w / input_w
            scale_y = screen_h / input_h

        # Увеличиваем age у всех треков
        for track in self._tracks.values():
            track.age += 1

        updated_tracks = set()
        current_boxes = []

        # Обрабатываем детекции
        for det in detections:
            if det.conf < self.min_confidence:
                continue

            # Преобразуем в экранные координаты
            x1_screen = int(det.x1 * scale_x) + monitor_left
            y1_screen = int(det.y1 * scale_y) + monitor_top
            x2_screen = int(det.x2 * scale_x) + monitor_left
            y2_screen = int(det.y2 * scale_y) + monitor_top

            # Гарантируем правильный порядок
            x1_screen, x2_screen = min(x1_screen, x2_screen), max(x1_screen, x2_screen)
            y1_screen, y2_screen = min(y1_screen, y2_screen), max(y1_screen, y2_screen)

            # Проверка минимального размера
            if (x2_screen - x1_screen) < self.min_box_size or (y2_screen - y1_screen) < self.min_box_size:
                continue

            # Клиппинг
            x1_screen = max(monitor_left, min(monitor_left + screen_w, x1_screen))
            y1_screen = max(monitor_top, min(monitor_top + screen_h, y1_screen))
            x2_screen = max(monitor_left, min(monitor_left + screen_w, x2_screen))
            y2_screen = max(monitor_top, min(monitor_top + screen_h, y2_screen))

            if x1_screen >= x2_screen or y1_screen >= y2_screen:
                continue

            current_box = (float(x1_screen), float(y1_screen), float(x2_screen), float(y2_screen))
            current_boxes.append(current_box)

            track_id = det.track_id

            # Поиск подходящего трека
            if track_id is None or track_id not in self._tracks:
                matching_id = self._find_matching_track(current_box, updated_tracks)
                if matching_id is not None:
                    track_id = matching_id

            # Обновляем или создаём трек
            if track_id is not None and track_id in self._tracks:
                track = self._tracks[track_id]

                # Умное сглаживание с учетом уверенности
                smoothed_box = self._apply_smart_smoothing(
                    current_box,
                    (track.x1, track.y1, track.x2, track.y2),
                    det.conf
                )

                track.x1, track.y1, track.x2, track.y2 = smoothed_box
                track.age = 0
                track.last_update = time.time()
                track.confidence = max(track.confidence, det.conf)

                # Сохраняем в историю
                track.history.append(smoothed_box)

                updated_tracks.add(track_id)
            else:
                # Создаём новый трек
                if track_id is None:
                    track_id = -self._total_tracks_created - 1

                if len(self._tracks) < self.max_tracks:
                    self._tracks[track_id] = _TrackState(
                        x1=current_box[0],
                        y1=current_box[1],
                        x2=current_box[2],
                        y2=current_box[3],
                        age=0,
                        track_id=track_id,
                        confidence=det.conf,
                    )
                    self._tracks[track_id].history.append(current_box)
                    updated_tracks.add(track_id)
                    self._total_tracks_created += 1

        # Удаляем старые треки (только если они долго не появлялись)
        to_delete = []
        for tid, track in self._tracks.items():
            if track.age > self.decay_frames:
                to_delete.append(tid)
                self._total_tracks_removed += 1

        for tid in to_delete:
            del self._tracks[tid]

        # Формируем результат - используем все активные треки (даже те, что временно пропали)
        result = []
        for track in self._tracks.values():
            x1 = int(round(min(track.x1, track.x2)))
            y1 = int(round(min(track.y1, track.y2)))
            x2 = int(round(max(track.x1, track.x2)))
            y2 = int(round(max(track.y1, track.y2)))

            # Клиппинг
            x1 = max(monitor_left, min(monitor_left + screen_w, x1))
            y1 = max(monitor_top, min(monitor_top + screen_h, y1))
            x2 = max(monitor_left, min(monitor_left + screen_w, x2))
            y2 = max(monitor_top, min(monitor_top + screen_h, y2))

            if x1 < x2 and y1 < y2:
                result.append((x1, y1, x2, y2))

        # Если нет детекций, но были в прошлом - удерживаем последние
        if len(result) == 0 and len(self._last_boxes) > 0:
            # Уменьшаем уверенность старых боксов
            result = self._last_boxes

        # Сохраняем для следующего кадра
        self._last_boxes = result

        return result

    def reset(self) -> None:
        """Полностью сбросить все треки."""
        self._tracks.clear()
        self._last_boxes = []

    def get_track_count(self) -> int:
        return len(self._tracks)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "active_tracks": len(self._tracks),
            "total_created": self._total_tracks_created,
            "total_removed": self._total_tracks_removed,
            "frames_processed": self._frames_processed,
            "max_tracks_limit": self.max_tracks,
        }