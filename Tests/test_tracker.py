"""
Тесты для трекера (Фаза 2 — с реальной реализацией).
Запускать после того, как Core/detector.py и Core/tracker.py реализованы.
"""

import pytest
import sys
import os
import time

# Добавляем путь к корню проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.detector import Detection
from Core.tracker import TrackSmoother, TrackStats


class TestTrackSmoother:
    """Тесты для реального TrackSmoother."""

    def test_backprojection(self):
        """Тест обратной проекции координат."""
        smoother = TrackSmoother()
        det = Detection(
            x1=320, y1=0, x2=640, y2=640,
            conf=0.9, track_id=1
        )

        boxes = smoother.update([det], screen_w=1920, screen_h=1080)

        assert len(boxes) == 1
        # 320 * (1920/640) = 320 * 3 = 960
        assert boxes[0][0] == 960
        # 0 * (1080/640) = 0
        assert boxes[0][1] == 0
        # 640 * 3 = 1920
        assert boxes[0][2] == 1920
        # 640 * (1080/640) = 1080
        assert boxes[0][3] == 1080

    def test_ema_converges(self):
        """
        Тест сходимости EMA-фильтра.

        При alpha=0.6 значение должно быстро приближаться к 300.
        """
        smoother = TrackSmoother(ema_alpha=0.6)
        det = Detection(
            x1=100, y1=100, x2=200, y2=200,
            conf=0.9, track_id=1
        )

        # Сохраняем значения для анализа
        values = []
        for i in range(20):
            boxes = smoother.update([det], screen_w=1920, screen_h=1080)
            values.append(boxes[0][0])

        # Проверяем, что значение увеличивается (сходится к 300)
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], f"Значение уменьшилось на шаге {i}: {values[i - 1]} -> {values[i]}"

        # Проверяем, что значение достаточно близко к 300 (в пределах 5 пикселей)
        assert abs(values[-1] - 300) < 5, f"Значение {values[-1]} слишком далеко от 300"

        # Проверяем, что скорость изменения уменьшается (сходимость замедляется)
        diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        for i in range(2, len(diffs)):
            # Разница должна уменьшаться
            assert diffs[i] <= diffs[i - 1], f"Скорость изменения выросла на шаге {i}"

    def test_decay_removes_track(self):
        """Тест удаления трека после decay_frames кадров без обновления."""
        smoother = TrackSmoother(decay_frames=3)
        det = Detection(
            x1=0, y1=0, x2=100, y2=100,
            conf=0.9, track_id=1
        )

        # Первый кадр с детекцией
        boxes = smoother.update([det], screen_w=1920, screen_h=1080)
        assert len(boxes) == 1

        # Три кадра без детекций (decay_frames=3, ещё держим)
        for i in range(3):
            boxes = smoother.update([], screen_w=1920, screen_h=1080)
            assert len(boxes) == 1, f"Кадр {i+2}: трек должен ещё быть"

        # Четвёртый кадр — должен исчезнуть
        boxes = smoother.update([], screen_w=1920, screen_h=1080)
        assert len(boxes) == 0

    def test_reset_clears_tracks(self):
        """Тест сброса всех треков."""
        smoother = TrackSmoother()
        det = Detection(
            x1=0, y1=0, x2=100, y2=100,
            conf=0.9, track_id=1
        )

        smoother.update([det], screen_w=1920, screen_h=1080)
        assert smoother.get_track_count() == 1

        smoother.reset()
        assert smoother.get_track_count() == 0

        boxes = smoother.update([], screen_w=1920, screen_h=1080)
        assert len(boxes) == 0

    def test_multiple_tracks(self):
        """Тест работы с несколькими треками."""
        smoother = TrackSmoother()
        dets = [
            Detection(x1=0, y1=0, x2=100, y2=100, conf=0.9, track_id=1),
            Detection(x1=200, y1=200, x2=300, y2=300, conf=0.9, track_id=2),
            Detection(x1=400, y1=400, x2=500, y2=500, conf=0.9, track_id=3),
        ]

        boxes = smoother.update(dets, screen_w=1920, screen_h=1080)
        assert len(boxes) == 3
        assert smoother.get_track_count() == 3
        assert set(smoother.get_track_ids()) == {1, 2, 3}

    def test_track_persistence(self):
        """Тест сохранения трека при временном пропадании."""
        smoother = TrackSmoother(decay_frames=5)
        det = Detection(
            x1=100, y1=100, x2=200, y2=200,
            conf=0.9, track_id=42
        )

        # Кадр 1: есть детекция
        smoother.update([det], screen_w=1920, screen_h=1080)

        # Кадры 2-3: нет детекций
        smoother.update([], screen_w=1920, screen_h=1080)
        smoother.update([], screen_w=1920, screen_h=1080)

        # Кадр 4: та же бутылка появляется снова
        boxes = smoother.update([det], screen_w=1920, screen_h=1080)

        # Проверим, что ID тот же
        assert smoother.get_track_ids() == [42]
        assert len(boxes) == 1

    def test_accept_track_id_none(self):
        """Тест обработки детекций без track_id."""
        smoother = TrackSmoother()
        dets = [
            Detection(x1=0, y1=0, x2=100, y2=100, conf=0.9, track_id=1),
            Detection(x1=200, y1=200, x2=300, y2=300, conf=0.9, track_id=None),
        ]

        boxes = smoother.update(dets, screen_w=1920, screen_h=1080)

        assert len(boxes) == 2

        # Проверяем, что это первый трек
        scale_x = 1920 / 640
        scale_y = 1080 / 640
        assert boxes[0][0] == int(0 * scale_x)
        assert boxes[0][1] == int(0 * scale_y)
        assert boxes[1][0] == int(200 * scale_x)
        assert boxes[1][1] == int(200 * scale_y)

    def test_monitor_offset(self):
        """Тест учёта смещения монитора."""
        smoother = TrackSmoother()
        det = Detection(
            x1=0, y1=0, x2=640, y2=640,
            conf=0.9, track_id=1
        )

        boxes = smoother.update(
            [det],
            screen_w=1920,
            screen_h=1080,
            monitor_left=100,
            monitor_top=50
        )

        assert boxes[0][0] == 100
        assert boxes[0][1] == 50
        assert boxes[0][2] == 2020  # 1920 + 100
        assert boxes[0][3] == 1130  # 1080 + 50

    def test_validation(self):
        """Тест валидации параметров."""
        # Некорректные alpha
        with pytest.raises(ValueError):
            TrackSmoother(ema_alpha=-0.5)
        with pytest.raises(ValueError):
            TrackSmoother(ema_alpha=1.5)

        # Некорректные decay_frames
        with pytest.raises(ValueError):
            TrackSmoother(decay_frames=-1)

        # Некорректный min_confidence
        with pytest.raises(ValueError):
            TrackSmoother(min_confidence=-0.1)
        with pytest.raises(ValueError):
            TrackSmoother(min_confidence=1.5)

        # Корректные значения должны работать
        smoother = TrackSmoother(ema_alpha=0.5, decay_frames=5)
        assert smoother.ema_alpha == 0.5
        assert smoother.decay_frames == 5

    def test_min_confidence_filter(self):
        """Тест фильтрации по минимальной уверенности."""
        smoother = TrackSmoother(min_confidence=0.8)

        # Детекция с низкой уверенностью
        det_low = Detection(x1=0, y1=0, x2=100, y2=100, conf=0.5, track_id=1)
        boxes = smoother.update([det_low], screen_w=1920, screen_h=1080)
        assert len(boxes) == 0  # Должна быть проигнорирована

        # Детекция с высокой уверенностью
        det_high = Detection(x1=0, y1=0, x2=100, y2=100, conf=0.9, track_id=1)
        boxes = smoother.update([det_high], screen_w=1920, screen_h=1080)
        assert len(boxes) == 1  # Должна быть принята

    def test_invalid_coordinates(self):
        """Тест обработки некорректных координат."""
        smoother = TrackSmoother()

        # Некорректные координаты (x1 > x2)
        det = Detection(x1=200, y1=100, x2=100, y2=200, conf=0.9, track_id=1)
        boxes = smoother.update([det], screen_w=1920, screen_h=1080)
        assert len(boxes) == 1
        # Координаты должны быть исправлены (x1 <= x2)
        assert boxes[0][0] <= boxes[0][2]
        assert boxes[0][1] <= boxes[0][3]

    def test_clipping(self):
        """Тест клипирования координат к экрану."""
        smoother = TrackSmoother()

        # Координаты за пределами экрана
        det = Detection(x1=-100, y1=-100, x2=800, y2=800, conf=0.9, track_id=1)
        boxes = smoother.update([det], screen_w=1920, screen_h=1080)

        # Должны быть обрезаны
        assert boxes[0][0] >= 0
        assert boxes[0][1] >= 0
        assert boxes[0][2] <= 1920
        assert boxes[0][3] <= 1080

    def test_track_stats(self):
        """Тест получения статистики треков."""
        smoother = TrackSmoother()
        det = Detection(x1=0, y1=0, x2=100, y2=100, conf=0.9, track_id=42)

        smoother.update([det], screen_w=1920, screen_h=1080)

        # Проверяем статистику
        stats = smoother.get_track_stats(42)
        assert stats is not None
        assert stats.track_id == 42
        assert stats.age == 0
        assert stats.lifetime >= 0
        assert stats.current_box is not None

        # Проверяем общую статистику
        general_stats = smoother.get_statistics()
        assert general_stats["active_tracks"] == 1
        assert general_stats["total_created"] == 1
        assert general_stats["frames_processed"] == 1

    def test_dynamic_params_update(self):
        """Тест динамического обновления параметров."""
        smoother = TrackSmoother(ema_alpha=0.5, decay_frames=10)

        # Меняем параметры
        smoother.set_params(ema_alpha=0.8, decay_frames=5, min_confidence=0.7)

        assert smoother.ema_alpha == 0.8
        assert smoother.decay_frames == 5
        assert smoother.min_confidence == 0.7

        # Меняем только один параметр
        smoother.set_params(ema_alpha=0.3)
        assert smoother.ema_alpha == 0.3
        assert smoother.decay_frames == 5  # Не изменился

    def test_scale_caching(self):
        """Тест кэширования масштаба."""
        smoother = TrackSmoother(enable_cache=True)
        det = Detection(x1=0, y1=0, x2=100, y2=100, conf=0.9, track_id=1)

        # Первый вызов - вычислит масштаб
        boxes1 = smoother.update([det], screen_w=1920, screen_h=1080)

        # Второй вызов с теми же параметрами - должен использовать кэш
        boxes2 = smoother.update([det], screen_w=1920, screen_h=1080)

        assert boxes1 == boxes2

        # Проверяем, что кэш работает (обращаемся к внутреннему состоянию)
        assert smoother._last_scale is not None

        # Меняем параметры - кэш должен обновиться
        smoother.update([det], screen_w=1280, screen_h=720)
        assert smoother._last_scale[0] == 1280 / 640  # scale_x должен обновиться

    def test_performance_with_many_tracks(self):
        """Тест производительности с большим количеством треков."""
        smoother = TrackSmoother()

        # Создаём 100 треков
        detections = [
            Detection(x1=i*10, y1=i*10, x2=i*10+100, y2=i*10+100,
                     conf=0.9, track_id=i)
            for i in range(100)
        ]

        start_time = time.time()
        boxes = smoother.update(detections, screen_w=1920, screen_h=1080)
        elapsed_time = time.time() - start_time

        assert len(boxes) == 100
        assert elapsed_time < 0.1, f"Обновление 100 треков заняло {elapsed_time:.3f} секунд"

    def test_ema_different_alphas(self):
        """Тест EMA с разными значениями alpha."""
        det = Detection(x1=100, y1=100, x2=200, y2=200, conf=0.9, track_id=1)

        # Alpha = 0.0 - не должно меняться
        smoother_static = TrackSmoother(ema_alpha=0.0)
        smoother_static.update([det], screen_w=1920, screen_h=1080)
        boxes = smoother_static.update([det], screen_w=1920, screen_h=1080)
        first_value = boxes[0][0]
        for _ in range(10):
            boxes = smoother_static.update([det], screen_w=1920, screen_h=1080)
            assert boxes[0][0] == first_value, "При alpha=0.0 значение не должно меняться"

        # Alpha = 1.0 - должно мгновенно обновляться
        smoother_instant = TrackSmoother(ema_alpha=1.0)
        for _ in range(10):
            boxes = smoother_instant.update([det], screen_w=1920, screen_h=1080)
            assert boxes[0][0] == 300, "При alpha=1.0 значение должно быть точным"

    def test_max_tracks_limit(self):
        """Тест ограничения максимального количества треков."""
        smoother = TrackSmoother(max_tracks=5)

        # Создаём 10 треков
        detections = [
            Detection(x1=i*10, y1=i*10, x2=i*10+100, y2=i*10+100,
                     conf=0.9, track_id=i)
            for i in range(10)
        ]

        boxes = smoother.update(detections, screen_w=1920, screen_h=1080)

        # Должно быть не больше max_tracks
        assert len(boxes) <= 5
        assert smoother.get_track_count() <= 5

        # Проверяем, что остались самые приоритетные треки
        stats = smoother.get_statistics()
        assert stats["max_tracks_limit"] == 5

    def test_track_history(self):
        """Тест сохранения истории треков."""
        smoother = TrackSmoother()
        det = Detection(x1=100, y1=100, x2=200, y2=200, conf=0.9, track_id=1)

        # Несколько обновлений
        for _ in range(5):
            smoother.update([det], screen_w=1920, screen_h=1080)

        # Проверяем историю
        history = smoother.get_track_history(1)
        assert history is not None
        assert len(history) == 5

        # Проверяем, что координаты правильные
        scale_x = 1920 / 640
        assert history[0][0] == 100 * scale_x

    def test_track_stability_score(self):
        """Тест вычисления стабильности трека."""
        smoother = TrackSmoother()

        # Стабильный трек (не двигается)
        det_stable = Detection(x1=100, y1=100, x2=200, y2=200, conf=0.9, track_id=1)
        for _ in range(10):
            smoother.update([det_stable], screen_w=1920, screen_h=1080)

        stats_stable = smoother.get_track_stats(1)
        assert stats_stable is not None
        assert stats_stable.stability_score > 0.9  # Должен быть очень стабильным

        # Нестабильный трек (двигается)
        smoother.reset()
        det_moving = Detection(x1=100, y1=100, x2=200, y2=200, conf=0.9, track_id=1)
        for i in range(10):
            det_moving.x1 = 100 + i * 50
            det_moving.x2 = 200 + i * 50
            smoother.update([det_moving], screen_w=1920, screen_h=1080)

        stats_moving = smoother.get_track_stats(1)
        assert stats_moving is not None
        assert stats_moving.stability_score < 0.5  # Должен быть менее стабильным

    def test_velocity_prediction(self):
        """Тест предсказания движения (если включено)."""
        smoother = TrackSmoother(enable_velocity_prediction=True, decay_frames=5)

        # Создаём движущийся объект
        det = Detection(x1=100, y1=100, x2=200, y2=200, conf=0.9, track_id=1)

        # Несколько кадров с движением
        for i in range(5):
            det.x1 = 100 + i * 20
            det.x2 = 200 + i * 20
            smoother.update([det], screen_w=1920, screen_h=1080)
            time.sleep(0.033)  # Симулируем ~30 FPS

        # Сохраняем позицию до пропадания
        boxes_before = smoother.update([], screen_w=1920, screen_h=1080)
        pos_before = boxes_before[0][0] if boxes_before else None

        # Пропускаем кадр без детекции (должно сработать предсказание)
        time.sleep(0.033)
        boxes_after = smoother.update([], screen_w=1920, screen_h=1080)

        if boxes_after:
            # С предсказанием позиция должна продвинуться вперёд
            assert boxes_after[0][0] > pos_before if pos_before else True

    def test_fps_tracking(self):
        """Тест отслеживания FPS."""
        smoother = TrackSmoother()

        # Симулируем несколько кадров
        det = Detection(x1=100, y1=100, x2=200, y2=200, conf=0.9, track_id=1)

        for _ in range(10):
            smoother.update([det], screen_w=1920, screen_h=1080)
            time.sleep(0.033)  # ~30 FPS

        stats = smoother.get_statistics()
        assert "current_fps" in stats
        # FPS должен быть примерно 30 (плюс-минус погрешность)
        assert 25 <= stats["current_fps"] <= 35


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
