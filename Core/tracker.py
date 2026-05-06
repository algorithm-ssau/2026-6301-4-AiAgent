"""
Модуль сглаживания треков для стабилизации прямоугольников между кадрами.

Принимает сырые детекции от нейросети (в координатах 640x640),
преобразует их в экранные координаты и применяет EMA-фильтрацию для
плавности движения. Также реализует механизм decay для временно
пропавших объектов.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import sys
import os
import time
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

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
    created_at: float = field(default_factory=time.time)  # время создания трека
    last_update: float = field(default_factory=time.time)  # время последнего обновления


@dataclass
class TrackStats:
    """Статистика по треку для отладки."""
    track_id: int
    age: int
    lifetime: float  # время жизни в секундах
    last_update_age: float  # время с последнего обновления в секундах
    current_box: Tuple[int, int, int, int]  # текущий прямоугольник


class TrackSmoother:
    """
    Сглаживание треков с помощью EMA и обратной проекцией в экранные координаты.

    Args:
        ema_alpha: Коэффициент сглаживания (0.0 - очень медленно, 1.0 - мгновенно)
        decay_frames: Сколько кадров держать прямоугольник после пропажи объекта
        enable_cache: Включить кэширование масштаба (улучшает производительность)
        min_confidence: Минимальная уверенность для учёта детекции (0.0-1.0)
    """

    def __init__(self, ema_alpha: float = 0.6, decay_frames: int = 10,
                 enable_cache: bool = True, min_confidence: float = 0.0):
        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha должен быть в [0, 1], получено {ema_alpha}")
        if decay_frames < 0:
            raise ValueError(f"decay_frames должен быть >= 0, получено {decay_frames}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence должен быть в [0, 1], получено {min_confidence}")

        self.ema_alpha = ema_alpha
        self.decay_frames = decay_frames
        self.min_confidence = min_confidence
        self.enable_cache = enable_cache
        self._tracks: Dict[int, _TrackState] = {}  # track_id -> состояние

        # Кэш для масштабирования (оптимизация производительности)
        self._scale_cache: Dict[Tuple[int, int, int, int, int, int], Tuple[float, float]] = {}
        self._last_params: Optional[Tuple[int, int, int, int, int, int]] = None
        self._last_scale: Optional[Tuple[float, float]] = None

        # Статистика
        self._total_tracks_created = 0
        self._total_tracks_removed = 0
        self._frames_processed = 0
        self._last_log_time = time.time()

    def _get_scale(self, screen_w: int, screen_h: int, input_w: int, input_h: int) -> Tuple[float, float]:
        """
        Получить масштабные коэффициенты с кэшированием.

        Args:
            screen_w, screen_h: Размер экрана
            input_w, input_h: Размер входного изображения

        Returns:
            (scale_x, scale_y)
        """
        if not self.enable_cache:
            return screen_w / input_w, screen_h / input_h

        current_params = (screen_w, screen_h, input_w, input_h)

        # Проверяем, изменились ли параметры
        if self._last_params == current_params and self._last_scale is not None:
            return self._last_scale

        # Вычисляем новые коэффициенты
        scale_x = screen_w / input_w
        scale_y = screen_h / input_h
        self._last_scale = (scale_x, scale_y)
        self._last_params = current_params

        return scale_x, scale_y

    def _apply_ema(self, new_value: float, old_value: float) -> float:
        """
        Применить EMA-фильтр к одному значению.

        Args:
            new_value: Новое значение (из детекции)
            old_value: Старое сглаженное значение

        Returns:
            Сглаженное значение
        """
        return self.ema_alpha * new_value + (1 - self.ema_alpha) * old_value

    def _should_ignore_detection(self, det: Detection) -> bool:
        """
        Проверить, нужно ли игнорировать детекцию.

        Args:
            det: Детекция для проверки

        Returns:
            True если детекцию нужно игнорировать
        """
        # Игнорируем детекции без track_id
        if det.track_id is None:
            return True

        # Игнорируем детекции с низкой уверенностью
        if det.conf < self.min_confidence:
            logger.debug(f"Игнорируем детекцию {det.track_id} с низкой уверенностью {det.conf:.2f}")
            return True

        # Проверяем валидность координат
        if det.x1 >= det.x2 or det.y1 >= det.y2:
            logger.warning(f"Некорректные координаты детекции {det.track_id}: ({det.x1}, {det.y1}, {det.x2}, {det.y2})")
            return True

        return False

    def _log_statistics_if_needed(self):
        """Периодически логировать статистику."""
        current_time = time.time()
        if current_time - self._last_log_time >= 5.0:  # Каждые 5 секунд
            logger.info(f"Трекер статистика: активных треков={len(self._tracks)}, "
                       f"всего создано={self._total_tracks_created}, "
                       f"удалено={self._total_tracks_removed}, "
                       f"кадров={self._frames_processed}")
            self._last_log_time = current_time

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
        self._frames_processed += 1

        # Шаг 1: Вычислить масштабные коэффициенты (с кэшированием)
        scale_x, scale_y = self._get_scale(screen_w, screen_h, input_w, input_h)

        # Шаг 2: Пометить все существующие треки как "не обновлённые" в этом кадре
        # Увеличиваем age у всех, потом обнулим у тех, что появились
        for track in self._tracks.values():
            track.age += 1

        # Шаг 3: Обработать новые детекции
        updated_tracks = set()
        for det in detections:
            # Проверяем, нужно ли игнорировать детекцию
            if self._should_ignore_detection(det):
                continue

            # Проецируем в экранные координаты
            x1_screen = int(det.x1 * scale_x) + monitor_left
            y1_screen = int(det.y1 * scale_y) + monitor_top
            x2_screen = int(det.x2 * scale_x) + monitor_left
            y2_screen = int(det.y2 * scale_y) + monitor_top

            # Обновляем или создаём трек
            if det.track_id in self._tracks:
                # Существующий трек — применяем EMA
                track = self._tracks[det.track_id]
                old_box = (track.x1, track.y1, track.x2, track.y2)

                track.x1 = self._apply_ema(x1_screen, track.x1)
                track.y1 = self._apply_ema(y1_screen, track.y1)
                track.x2 = self._apply_ema(x2_screen, track.x2)
                track.y2 = self._apply_ema(y2_screen, track.y2)
                track.age = 0  # сброс счётчика пропусков
                track.last_update = time.time()
                updated_tracks.add(det.track_id)

                # Логируем значительные изменения
                if abs(track.x1 - old_box[0]) > 100 or abs(track.y1 - old_box[1]) > 100:
                    logger.debug(f"Трек {det.track_id} значительно переместился: "
                               f"{old_box[:2]} -> ({track.x1:.1f}, {track.y1:.1f})")
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
                updated_tracks.add(det.track_id)
                self._total_tracks_created += 1
                logger.info(f"Создан новый трек {det.track_id} с уверенностью {det.conf:.2f}")

        # Шаг 4: Удалить старые треки (превысившие decay_frames)
        to_delete = []
        for tid, track in self._tracks.items():
            if track.age > self.decay_frames:
                to_delete.append(tid)
                self._total_tracks_removed += 1
                logger.debug(f"Удалён трек {tid} (возраст {track.age} > {self.decay_frames})")

        for tid in to_delete:
            del self._tracks[tid]

        # Шаг 5: Сформировать выходной список (включая треки с age > 0)
        result = []
        for track in self._tracks.values():
            # Обеспечиваем валидность координат (x1 <= x2, y1 <= y2)
            x1 = int(round(min(track.x1, track.x2)))
            y1 = int(round(min(track.y1, track.y2)))
            x2 = int(round(max(track.x1, track.x2)))
            y2 = int(round(max(track.y1, track.y2)))

            # Клипируем к экрану
            x1 = max(monitor_left, min(monitor_left + screen_w, x1))
            y1 = max(monitor_top, min(monitor_top + screen_h, y1))
            x2 = max(monitor_left, min(monitor_left + screen_w, x2))
            y2 = max(monitor_top, min(monitor_top + screen_h, y2))

            result.append((x1, y1, x2, y2))

        # Логируем статистику при необходимости
        self._log_statistics_if_needed()

        return result

    def reset(self) -> None:
        """Полностью сбросить все треки."""
        removed_count = len(self._tracks)
        self._tracks.clear()
        self._total_tracks_removed += removed_count
        logger.info(f"Трекер сброшен. Удалено {removed_count} треков")

        # Сбрасываем кэш
        self._last_params = None
        self._last_scale = None

    def get_track_count(self) -> int:
        """Вернуть количество активных треков (для отладки)."""
        return len(self._tracks)

    def get_track_ids(self) -> List[int]:
        """Вернуть список активных track_id (для отладки)."""
        return list(self._tracks.keys())

    def get_track_stats(self, track_id: int) -> Optional[TrackStats]:
        """
        Получить статистику по конкретному треку.

        Args:
            track_id: ID трека

        Returns:
            TrackStats или None если трек не найден
        """
        track = self._tracks.get(track_id)
        if track is None:
            return None

        current_time = time.time()
        return TrackStats(
            track_id=track.track_id,
            age=track.age,
            lifetime=current_time - track.created_at,
            last_update_age=current_time - track.last_update,
            current_box=(int(track.x1), int(track.y1), int(track.x2), int(track.y2))
        )

    def get_all_stats(self) -> List[TrackStats]:
        """Получить статистику по всем активным трекам."""
        return [self.get_track_stats(tid) for tid in self._tracks.keys()
                if self.get_track_stats(tid) is not None]

    def get_statistics(self) -> Dict[str, int]:
        """
        Получить общую статистику работы трекера.

        Returns:
            Словарь со статистикой
        """
        return {
            "active_tracks": len(self._tracks),
            "total_created": self._total_tracks_created,
            "total_removed": self._total_tracks_removed,
            "frames_processed": self._frames_processed,
        }

    def set_params(self, ema_alpha: Optional[float] = None,
                   decay_frames: Optional[int] = None,
                   min_confidence: Optional[float] = None):
        """
        Динамически изменить параметры трекера.

        Args:
            ema_alpha: Новый коэффициент сглаживания (если указан)
            decay_frames: Новое количество кадров для decay (если указан)
            min_confidence: Новый порог уверенности (если указан)
        """
        if ema_alpha is not None:
            if not 0.0 <= ema_alpha <= 1.0:
                raise ValueError(f"ema_alpha должен быть в [0, 1], получено {ema_alpha}")
            self.ema_alpha = ema_alpha
            logger.info(f"EMA alpha изменён на {ema_alpha}")

        if decay_frames is not None:
            if decay_frames < 0:
                raise ValueError(f"decay_frames должен быть >= 0, получено {decay_frames}")
            self.decay_frames = decay_frames
            logger.info(f"Decay frames изменён на {decay_frames}")

        if min_confidence is not None:
            if not 0.0 <= min_confidence <= 1.0:
                raise ValueError(f"min_confidence должен быть в [0, 1], получено {min_confidence}")
            self.min_confidence = min_confidence
            logger.info(f"Min confidence изменён на {min_confidence}")

    def clear_cache(self):
        """Очистить кэш масштабирования."""
        self._scale_cache.clear()
        self._last_params = None
        self._last_scale = None
        logger.debug("Кэш масштаба очищен")