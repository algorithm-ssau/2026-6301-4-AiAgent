"""
Тесты для трекера (Фаза 2 — с реальной реализацией).
Запускать после того, как Core/detector.py и Core/tracker.py реализованы.
"""

import pytest
import sys
import os

# Добавляем путь к корню проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.detector import Detection
from Core.tracker import TrackSmoother


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

        При alpha=0.5 значение должно приближаться к 300, но не достигать его.
        """
        smoother = TrackSmoother(ema_alpha=0.5)
        det = Detection(
            x1=100, y1=100, x2=200, y2=200,
            conf=0.9, track_id=1
        )

        # Сохраняем значения для анализа
        values = []
        for i in range(30):
            boxes = smoother.update([det], screen_w=1920, screen_h=1080)
            values.append(boxes[0][0])

        # Проверяем, что значение увеличивается (сходится к 300)
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], f"Значение уменьшилось на шаге {i}: {values[i - 1]} -> {values[i]}"

        # Проверяем, что значение не превышает 300
        assert values[-1] <= 300, f"Значение {values[-1]} превышает 300"

        # Проверяем, что значение достаточно близко к 300 (в пределах 20%)
        assert values[-1] > 240, f"Значение {values[-1]} слишком далеко от 300"

        # Проверяем, что скорость изменения уменьшается (сходимость замедляется)
        diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        for i in range(2, len(diffs)):
            # Не строгое условие, но хотя бы не увеличивается резко
            assert diffs[i] <= diffs[i - 1] * 1.1, f"Скорость изменения выросла на шаге {i}"

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

    def test_ignore_track_id_none(self):
        """Тест игнорирования детекций без track_id."""
        smoother = TrackSmoother()
        dets = [
            Detection(x1=0, y1=0, x2=100, y2=100, conf=0.9, track_id=1),
            Detection(x1=200, y1=200, x2=300, y2=300, conf=0.9, track_id=None),
        ]

        boxes = smoother.update(dets, screen_w=1920, screen_h=1080)

        # Должен быть только 1 трек (с track_id=1)
        assert len(boxes) == 1

        # Проверяем, что это первый трек
        scale_x = 1920 / 640
        scale_y = 1080 / 640
        assert boxes[0][0] == int(0 * scale_x)
        assert boxes[0][1] == int(0 * scale_y)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])