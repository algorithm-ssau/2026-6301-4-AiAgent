"""
Модуль сглаживания треков для стабилизации прямоугольников между кадрами.

Принимает сырые детекции от нейросети (в координатах 640x640),
преобразует их в экранные координаты и применяет EMA-фильтрацию для
плавности движения. Также реализует механизм decay для временно
пропавших объектов.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import sys
import os
import time
import logging
from collections import deque

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
    history: deque = field(default_factory=lambda: deque(maxlen=10))  # история позиций для анализа
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=10))  # история уверенности


@dataclass
class TrackStats:
    """Статистика по треку для отладки."""
    track_id: int
    age: int
    lifetime: float  # время жизни в секундах
    last_update_age: float  # время с последнего обновления в секундах
    current_box: Tuple[int, int, int, int]  # текущий прямоугольник
    stability_score: float = 0.0  # показатель стабильности трека (0-1)
    avg_confidence: float = 0.0  # средняя уверенность


class TrackSmoother:
    """
    Сглаживание треков с помощью EMA и обратной проекцией в экранные координаты.

    Args:
        ema_alpha: Коэффициент сглаживания (0.0 - очень медленно, 1.0 - мгновенно)
        decay_frames: Сколько кадров держать прямоугольник после пропажи объекта
        enable_cache: Включить кэширование масштаба (улучшает производительность)
        min_confidence: Минимальная уверенность для учёта детекции (0.0-1.0)
        enable_velocity_prediction: Включить предсказание движения (экспериментально)
        max_tracks: Максимальное количество одновременно отслеживаемых треков
    """

    def __init__(self, ema_alpha: float = 0.6, decay_frames: int = 10,
                 enable_cache: bool = True, min_confidence: float = 0.0,
                 enable_velocity_prediction: bool = False, max_tracks: int = 100):
        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha должен быть в [0, 1], получено {ema_alpha}")
        if decay_frames < 0:
            raise ValueError(f"decay_frames должен быть >= 0, получено {decay_frames}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence должен быть в [0, 1], получено {min_confidence}")
        if max_tracks < 1:
            raise ValueError(f"max_tracks должен быть >= 1, получено {max_tracks}")

        self.ema_alpha = ema_alpha
        self.decay_frames = decay_frames
        self.min_confidence = min_confidence
        self.enable_cache = enable_cache
        self.enable_velocity_prediction = enable_velocity_prediction
        self.max_tracks = max_tracks
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

        # Для velocity prediction
        self._last_frame_time = time.time()
        self._next_synthetic_track_id = -1
        self._frame_times = deque(maxlen=30)  # история времени кадров для FPS

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

    def _predict_position(self, track: _TrackState, dt: float) -> Tuple[float, float, float, float]:
        """
        Предсказать позицию на основе истории движения.

        Args:
            track: Состояние трека
            dt: Время с последнего обновления в секундах

        Returns:
            Предсказанные координаты (x1, y1, x2, y2)
        """
        if not self.enable_velocity_prediction or len(track.history) < 3:
            return track.x1, track.y1, track.x2, track.y2

        # Вычисляем скорость по истории
        velocities = []
        history_list = list(track.history)

        for i in range(1, len(history_list)):
            prev_box = history_list[i-1]
            curr_box = history_list[i]
            # Считаем смещение центра
            prev_center_x = (prev_box[0] + prev_box[2]) / 2
            prev_center_y = (prev_box[1] + prev_box[3]) / 2
            curr_center_x = (curr_box[0] + curr_box[2]) / 2
            curr_center_y = (curr_box[1] + curr_box[3]) / 2

            velocities.append((curr_center_x - prev_center_x, curr_center_y - prev_center_y))

        if velocities:
            # Берём среднюю скорость
            avg_vx = sum(v[0] for v in velocities) / len(velocities)
            avg_vy = sum(v[1] for v in velocities) / len(velocities)

            # Предсказываем новое положение
            current_center_x = (track.x1 + track.x2) / 2
            current_center_y = (track.y1 + track.y2) / 2

            predicted_center_x = current_center_x + avg_vx * dt * 30  # примерно 30 FPS
            predicted_center_y = current_center_y + avg_vy * dt * 30

            # Сохраняем размеры
            width = track.x2 - track.x1
            height = track.y2 - track.y1

            predicted_x1 = predicted_center_x - width / 2
            predicted_y1 = predicted_center_y - height / 2
            predicted_x2 = predicted_center_x + width / 2
            predicted_y2 = predicted_center_y + height / 2

            return predicted_x1, predicted_y1, predicted_x2, predicted_y2

        return track.x1, track.y1, track.x2, track.y2

    def _should_ignore_detection(self, det: Detection) -> bool:
        """
        Проверить, нужно ли игнорировать детекцию.

        Args:
            det: Детекция для проверки

        Returns:
            True если детекцию нужно игнорировать
        """
        # Игнорируем детекции без track_id
        # Игнорируем детекции с низкой уверенностью
        if det.conf < self.min_confidence:
            logger.debug(f"Игнорируем детекцию {det.track_id} с низкой уверенностью {det.conf:.2f}")
            return True

        # Проверяем валидность координат
        if det.x1 == det.x2 or det.y1 == det.y2:
            logger.warning(f"Некорректные координаты детекции {det.track_id}: ({det.x1}, {det.y1}, {det.x2}, {det.y2})")
            return True

        return False

    def _box_iou(
            self,
            a: Tuple[float, float, float, float],
            b: Tuple[float, float, float, float]
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area

        return 0.0 if union <= 0 else inter_area / union

    def _center_distance(
            self,
            a: Tuple[float, float, float, float],
            b: Tuple[float, float, float, float]
    ) -> float:
        ax = (a[0] + a[2]) / 2
        ay = (a[1] + a[3]) / 2
        bx = (b[0] + b[2]) / 2
        by = (b[1] + b[3]) / 2

        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def _find_matching_track(
            self,
            box: Tuple[float, float, float, float],
            updated_tracks: set
    ) -> Optional[int]:
        best_track_id = None
        best_iou = 0.0
        best_distance = float("inf")

        box_w = max(1.0, box[2] - box[0])
        box_h = max(1.0, box[3] - box[1])
        max_distance = max(80.0, (box_w ** 2 + box_h ** 2) ** 0.5 * 0.75)

        for track_id, track in self._tracks.items():
            if track_id in updated_tracks:
                continue

            track_box = (track.x1, track.y1, track.x2, track.y2)
            iou = self._box_iou(box, track_box)
            distance = self._center_distance(box, track_box)

            if iou > best_iou or (best_iou == 0.0 and distance < best_distance):
                best_track_id = track_id
                best_iou = iou
                best_distance = distance

        if best_track_id is None:
            return None

        if best_iou >= 0.15 or best_distance <= max_distance:
            return best_track_id

        return None

    def _new_synthetic_track_id(self) -> int:
        track_id = self._next_synthetic_track_id
        self._next_synthetic_track_id -= 1
        return track_id

    def _log_statistics_if_needed(self):
        """Периодически логировать статистику."""
        current_time = time.time()
        if current_time - self._last_log_time >= 5.0:  # Каждые 5 секунд
            # Вычисляем средний FPS
            avg_fps = 0
            if len(self._frame_times) > 1:
                time_diffs = [self._frame_times[i] - self._frame_times[i-1]
                             for i in range(1, len(self._frame_times))]
                if time_diffs:
                    avg_fps = 1.0 / (sum(time_diffs) / len(time_diffs))

            logger.info(f"Трекер статистика: активных треков={len(self._tracks)}, "
                       f"всего создано={self._total_tracks_created}, "
                       f"удалено={self._total_tracks_removed}, "
                       f"кадров={self._frames_processed}, "
                       f"FPS={avg_fps:.1f}")
            self._last_log_time = current_time

    def _prune_old_tracks(self):
        """Удалить треки, если их слишком много (по количеству или возрасту)."""
        if len(self._tracks) <= self.max_tracks:
            return

        # Сортируем треки по приоритету (молодые и недавно обновлённые важнее)
        sorted_tracks = sorted(
            self._tracks.items(),
            key=lambda x: (x[1].age, time.time() - x[1].last_update)
        )

        # Удаляем самые старые
        to_remove = [tid for tid, _ in sorted_tracks[self.max_tracks:]]
        for tid in to_remove:
            del self._tracks[tid]
            self._total_tracks_removed += 1
            logger.debug(f"Принудительно удалён трек {tid} (превышен лимит)")

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
        # Замеряем время кадра для FPS
        current_time = time.time()
        self._frame_times.append(current_time)
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
            det_x1 = min(det.x1, det.x2)
            det_y1 = min(det.y1, det.y2)
            det_x2 = max(det.x1, det.x2)
            det_y2 = max(det.y1, det.y2)

            x1_screen = int(det_x1 * scale_x) + monitor_left
            y1_screen = int(det_y1 * scale_y) + monitor_top
            x2_screen = int(det_x2 * scale_x) + monitor_left
            y2_screen = int(det_y2 * scale_y) + monitor_top

            box = (float(x1_screen), float(y1_screen), float(x2_screen), float(y2_screen))
            track_id = det.track_id
            if track_id is None:
                track_id = self._find_matching_track(box, updated_tracks)
                if track_id is None:
                    track_id = self._new_synthetic_track_id()
            elif track_id not in self._tracks:
                matching_track_id = self._find_matching_track(box, updated_tracks)
                if matching_track_id is not None and matching_track_id < 0:
                    self._tracks[track_id] = self._tracks.pop(matching_track_id)
                    self._tracks[track_id].track_id = track_id

            # Обновляем или создаём трек
            if track_id in self._tracks:
                # Существующий трек — применяем EMA
                track = self._tracks[track_id]
                old_box = (track.x1, track.y1, track.x2, track.y2)

                track.x1 = self._apply_ema(x1_screen, track.x1)
                track.y1 = self._apply_ema(y1_screen, track.y1)
                track.x2 = self._apply_ema(x2_screen, track.x2)
                track.y2 = self._apply_ema(y2_screen, track.y2)
                track.age = 0  # сброс счётчика пропусков
                track.last_update = time.time()

                # Сохраняем в историю
                track.history.append((track.x1, track.y1, track.x2, track.y2))
                track.confidence_history.append(det.conf)

                updated_tracks.add(track_id)

                # Логируем значительные изменения
                if abs(track.x1 - old_box[0]) > 100 or abs(track.y1 - old_box[1]) > 100:
                    logger.debug(f"Трек {det.track_id} значительно переместился: "
                               f"{old_box[:2]} -> ({track.x1:.1f}, {track.y1:.1f})")
            else:
                # Проверяем лимит треков
                if len(self._tracks) >= self.max_tracks:
                    self._prune_old_tracks()

                # Новый трек — инициализируем
                new_track = _TrackState(
                    x1=float(x1_screen),
                    y1=float(y1_screen),
                    x2=float(x2_screen),
                    y2=float(y2_screen),
                    age=0,
                    track_id=track_id
                )
                new_track.history.append((float(x1_screen), float(y1_screen),
                                         float(x2_screen), float(y2_screen)))
                new_track.confidence_history.append(det.conf)

                self._tracks[track_id] = new_track
                updated_tracks.add(track_id)
                self._total_tracks_created += 1
                if len(self._tracks) > self.max_tracks:
                    self._prune_old_tracks()
                logger.info(f"Создан новый трек {det.track_id} с уверенностью {det.conf:.2f}")

        # Шаг 4: Применить velocity prediction для пропавших треков
        if self.enable_velocity_prediction:
            dt = current_time - self._last_frame_time
            for track in self._tracks.values():
                if track.age > 0 and track.age <= self.decay_frames:
                    # Предсказываем позицию для временно пропавших объектов
                    pred_x1, pred_y1, pred_x2, pred_y2 = self._predict_position(track, dt)
                    # Смешиваем предсказание с текущей позицией
                    mix_factor = min(0.3, track.age * 0.05)  # Чем дольше пропал, тем больше предсказание
                    track.x1 = track.x1 * (1 - mix_factor) + pred_x1 * mix_factor
                    track.y1 = track.y1 * (1 - mix_factor) + pred_y1 * mix_factor
                    track.x2 = track.x2 * (1 - mix_factor) + pred_x2 * mix_factor
                    track.y2 = track.y2 * (1 - mix_factor) + pred_y2 * mix_factor

        self._last_frame_time = current_time

        # Шаг 5: Удалить старые треки (превысившие decay_frames)
        to_delete = []
        for tid, track in self._tracks.items():
            if track.age > self.decay_frames:
                to_delete.append(tid)
                self._total_tracks_removed += 1
                logger.debug(f"Удалён трек {tid} (возраст {track.age} > {self.decay_frames})")

        for tid in to_delete:
            del self._tracks[tid]

        # Шаг 6: Сформировать выходной список (включая треки с age > 0)
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
        self._frame_times.clear()

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

        # Вычисляем стабильность трека (насколько мало меняется позиция)
        stability_score = 1.0
        if len(track.history) > 1:
            movements = []
            history_list = list(track.history)
            for i in range(1, len(history_list)):
                prev_center = ((history_list[i-1][0] + history_list[i-1][2]) / 2,
                              (history_list[i-1][1] + history_list[i-1][3]) / 2)
                curr_center = ((history_list[i][0] + history_list[i][2]) / 2,
                              (history_list[i][1] + history_list[i][3]) / 2)
                movement = ((curr_center[0] - prev_center[0])**2 +
                           (curr_center[1] - prev_center[1])**2)**0.5
                movements.append(movement)
            if movements:
                avg_movement = sum(movements) / len(movements)
                # Чем меньше движение, тем стабильнее
                stability_score = max(0.0, 1.0 - min(1.0, avg_movement / 100.0))

        # Средняя уверенность
        avg_confidence = sum(track.confidence_history) / max(1, len(track.confidence_history))

        return TrackStats(
            track_id=track.track_id,
            age=track.age,
            lifetime=current_time - track.created_at,
            last_update_age=current_time - track.last_update,
            current_box=(int(track.x1), int(track.y1), int(track.x2), int(track.y2)),
            stability_score=stability_score,
            avg_confidence=avg_confidence
        )

    def get_all_stats(self) -> List[TrackStats]:
        """Получить статистику по всем активным трекам."""
        return [self.get_track_stats(tid) for tid in self._tracks.keys()
                if self.get_track_stats(tid) is not None]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получить общую статистику работы трекера.

        Returns:
            Словарь со статистикой
        """
        # Вычисляем средний FPS
        avg_fps = 0.0
        if len(self._frame_times) > 1:
            time_diffs = [self._frame_times[i] - self._frame_times[i-1]
                         for i in range(1, len(self._frame_times))]
            if time_diffs:
                avg_fps = 1.0 / (sum(time_diffs) / len(time_diffs))

        return {
            "active_tracks": len(self._tracks),
            "total_created": self._total_tracks_created,
            "total_removed": self._total_tracks_removed,
            "frames_processed": self._frames_processed,
            "current_fps": avg_fps,
            "max_tracks_limit": self.max_tracks,
            "velocity_prediction_enabled": self.enable_velocity_prediction,
        }

    def set_params(self, ema_alpha: Optional[float] = None,
                   decay_frames: Optional[int] = None,
                   min_confidence: Optional[float] = None,
                   max_tracks: Optional[int] = None,
                   enable_velocity_prediction: Optional[bool] = None):
        """
        Динамически изменить параметры трекера.

        Args:
            ema_alpha: Новый коэффициент сглаживания (если указан)
            decay_frames: Новое количество кадров для decay (если указан)
            min_confidence: Новый порог уверенности (если указан)
            max_tracks: Новый лимит треков (если указан)
            enable_velocity_prediction: Включить/выключить предсказание (если указан)
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

        if max_tracks is not None:
            if max_tracks < 1:
                raise ValueError(f"max_tracks должен быть >= 1, получено {max_tracks}")
            self.max_tracks = max_tracks
            self._prune_old_tracks()  # применяем новый лимит
            logger.info(f"Max tracks изменён на {max_tracks}")

        if enable_velocity_prediction is not None:
            self.enable_velocity_prediction = enable_velocity_prediction
            logger.info(f"Velocity prediction изменён на {enable_velocity_prediction}")

    def clear_cache(self):
        """Очистить кэш масштабирования."""
        self._scale_cache.clear()
        self._last_params = None
        self._last_scale = None
        logger.debug("Кэш масштаба очищен")

    def get_track_history(self, track_id: int) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Получить историю позиций трека.

        Args:
            track_id: ID трека

        Returns:
            Список прямоугольников из истории или None
        """
        track = self._tracks.get(track_id)
        if track is None:
            return None
        return list(track.history)
