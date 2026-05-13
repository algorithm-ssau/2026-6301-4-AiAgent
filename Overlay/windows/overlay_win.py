"""
Windows оверлей с принудительным удержанием боксов для полного устранения мерцания
"""
import threading
import time
from typing import List, Tuple, Dict
from collections import deque

import win32api
import win32con
import win32gui

Box = Tuple[int, int, int, int]


class TrackedBox:
    """Бокс с историей и таймером жизни"""
    def __init__(self, box: Box, confidence: float = 1.0):
        self.box = box
        self.confidence = confidence
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.seen_count = 1
        self.alpha = 1.0  # Прозрачность для плавного исчезновения

    def update(self, box: Box, confidence: float = 1.0):
        """Обновить бокс с усреднением"""
        # Экспоненциальное скользящее среднее для плавности
        alpha = 0.3
        self.box = (
            int(alpha * box[0] + (1 - alpha) * self.box[0]),
            int(alpha * box[1] + (1 - alpha) * self.box[1]),
            int(alpha * box[2] + (1 - alpha) * self.box[2]),
            int(alpha * box[3] + (1 - alpha) * self.box[3]),
        )
        self.confidence = max(self.confidence, confidence)
        self.last_seen = time.time()
        self.seen_count += 1
        self.alpha = 1.0

    def is_alive(self, max_age_seconds: float = 0.5) -> bool:
        """Жив ли бокс (не слишком ли давно не обновлялся)"""
        return (time.time() - self.last_seen) < max_age_seconds

    def get_fade_alpha(self, fade_duration: float = 0.3) -> float:
        """Получить альфу для плавного исчезновения"""
        age = time.time() - self.last_seen
        if age <= 0:
            return 1.0
        if age >= fade_duration:
            return 0.0
        return 1.0 - (age / fade_duration)


class WindowsOverlay:
    """Windows оверлей с принудительным удержанием боксов"""

    def __init__(self, width: int, height: int, update_rate: int = 30,
                 hold_seconds: float = 0.5,      # Держим бокс 0.5 секунды после пропажи
                 fade_seconds: float = 0.2):     # Плавное исчезновение 0.2 секунды
        self.width = width
        self.height = height
        self.update_rate = update_rate
        self.hold_seconds = hold_seconds
        self.fade_seconds = fade_seconds
        self._min_update_interval = 1.0 / update_rate

        self._hwnd = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Хранилище боксов с историей
        self._boxes: Dict[str, TrackedBox] = {}
        self._last_update_time = 0
        self._dirty = True

        self._class_name = f"AlcoholCensorOverlay_{id(self)}"
        self._transparent_color = win32api.RGB(255, 0, 255)

    def _box_key(self, box: Box) -> str:
        """Уникальный ключ для бокса по его позиции"""
        # Округляем до 20 пикселей для группировки близких боксов
        x1 = int(box[0] / 20)
        y1 = int(box[1] / 20)
        x2 = int(box[2] / 20)
        y2 = int(box[3] / 20)
        return f"{x1},{y1},{x2},{y2}"

    def _boxes_intersect(self, box1: Box, box2: Box, threshold: float = 0.3) -> bool:
        """Проверяет, пересекаются ли два бокса"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return False

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        iou = inter / union if union > 0 else 0
        return iou > threshold

    def _find_matching_box(self, box: Box) -> str | None:
        """Найти существующий бокс, который пересекается с новым"""
        for key, tracked in self._boxes.items():
            if self._boxes_intersect(box, tracked.box):
                return key
        return None

    def update_boxes(self, boxes: List[Box]) -> None:
        """Обновить список боксов с принудительным удержанием"""
        current_time = time.time()

        # Ограничение частоты обновления
        if current_time - self._last_update_time < self._min_update_interval:
            return

        self._last_update_time = current_time

        with self._lock:
            # Обновляем существующие боксы и добавляем новые
            processed_keys = set()

            for box in boxes:
                # Проверяем минимальный размер
                if (box[2] - box[0]) < 20 or (box[3] - box[1]) < 20:
                    continue

                # Ищем подходящий существующий бокс
                matching_key = self._find_matching_box(box)

                if matching_key and matching_key in self._boxes:
                    # Обновляем существующий
                    self._boxes[matching_key].update(box)
                    processed_keys.add(matching_key)
                else:
                    # Создаем новый
                    new_key = self._box_key(box)
                    self._boxes[new_key] = TrackedBox(box)
                    processed_keys.add(new_key)

            # Удаляем только те боксы, которые давно не обновлялись
            to_delete = []
            for key, tracked in self._boxes.items():
                if not tracked.is_alive(self.hold_seconds):
                    to_delete.append(key)

            for key in to_delete:
                del self._boxes[key]

            self._dirty = True

        # Запрашиваем перерисовку
        if self._hwnd:
            try:
                win32gui.InvalidateRect(self._hwnd, None, False)
            except Exception:
                pass

    def start(self) -> None:
        """Создать окно"""
        try:
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = self._window_procedure
            wc.lpszClassName = self._class_name
            wc.hInstance = win32api.GetModuleHandle(None)
            wc.hbrBackground = win32gui.GetStockObject(win32con.NULL_BRUSH)

            try:
                win32gui.RegisterClass(wc)
            except win32gui.error as e:
                if e.winerror != 1410:
                    raise

            ex_style = (
                win32con.WS_EX_LAYERED
                | win32con.WS_EX_TRANSPARENT
                | win32con.WS_EX_TOPMOST
                | win32con.WS_EX_NOACTIVATE
            )

            style = win32con.WS_POPUP

            self._hwnd = win32gui.CreateWindowEx(
                ex_style,
                self._class_name,
                "Alcohol Censor Overlay",
                style,
                0,
                0,
                self.width,
                self.height,
                None,
                None,
                wc.hInstance,
                None,
            )

            if not self._hwnd:
                raise RuntimeError("Не удалось создать окно оверлея")

            win32gui.SetLayeredWindowAttributes(
                self._hwnd,
                self._transparent_color,
                0,
                win32con.LWA_COLORKEY,
            )

            win32gui.ShowWindow(self._hwnd, win32con.SW_SHOW)
            win32gui.UpdateWindow(self._hwnd)

            self._running = True
            self._thread = threading.Thread(target=self._message_loop, daemon=True)
            self._thread.start()

        except Exception as e:
            print(f"Ошибка запуска оверлея: {e}")
            raise

    def stop(self) -> None:
        """Закрыть окно"""
        self._running = False
        if self._hwnd:
            try:
                win32gui.DestroyWindow(self._hwnd)
            except Exception:
                pass
            self._hwnd = None

    def is_running(self) -> bool:
        return self._running

    def _draw_rectangles(self, hdc):
        """Нарисовать прямоугольники с учетом прозрачности"""
        with self._lock:
            boxes = list(self._boxes.values())
            self._dirty = False

        if not boxes:
            return

        brush = None
        old_brush = None
        old_pen = None

        try:
            brush = win32gui.CreateSolidBrush(win32api.RGB(0, 0, 0))
            old_brush = win32gui.SelectObject(hdc, brush)

            null_pen = win32gui.GetStockObject(win32con.NULL_PEN)
            old_pen = win32gui.SelectObject(hdc, null_pen)

            for tracked in boxes:
                # Если бокс жив - рисуем его
                if tracked.is_alive(self.hold_seconds):
                    x1, y1, x2, y2 = tracked.box

                    x1 = max(0, min(int(x1), self.width))
                    y1 = max(0, min(int(y1), self.height))
                    x2 = max(0, min(int(x2), self.width))
                    y2 = max(0, min(int(y2), self.height))

                    if x1 < x2 and y1 < y2:
                        win32gui.Rectangle(hdc, x1, y1, x2, y2)
                else:
                    # Бокс умирает - рисуем с затуханием (если поддерживается)
                    # В Win32 API просто пропускаем для чистоты
                    pass

        except Exception as e:
            print(f"Ошибка рисования: {e}")
        finally:
            try:
                if old_pen:
                    win32gui.SelectObject(hdc, old_pen)
                if old_brush:
                    win32gui.SelectObject(hdc, old_brush)
                if brush:
                    win32gui.DeleteObject(brush)
            except Exception:
                pass

    def _window_procedure(self, hwnd, msg, wparam, lparam):
        """Обработчик сообщений"""
        if msg == win32con.WM_PAINT:
            hdc, paint_struct = win32gui.BeginPaint(hwnd)

            # Двойная буферизация
            mem_dc = win32gui.CreateCompatibleDC(hdc)
            bitmap = win32gui.CreateCompatibleBitmap(hdc, self.width, self.height)
            old_bitmap = win32gui.SelectObject(mem_dc, bitmap)

            try:
                # Заливаем прозрачным фоном
                bg_brush = win32gui.CreateSolidBrush(self._transparent_color)
                rect = (0, 0, self.width, self.height)
                win32gui.FillRect(mem_dc, rect, bg_brush)
                win32gui.DeleteObject(bg_brush)

                # Рисуем прямоугольники
                self._draw_rectangles(mem_dc)

                # Копируем на экран
                win32gui.BitBlt(
                    hdc, 0, 0, self.width, self.height,
                    mem_dc, 0, 0, win32con.SRCCOPY
                )
            finally:
                win32gui.SelectObject(mem_dc, old_bitmap)
                win32gui.DeleteObject(bitmap)
                win32gui.DeleteDC(mem_dc)
                win32gui.EndPaint(hwnd, paint_struct)

            return 0

        elif msg == win32con.WM_DESTROY:
            self._running = False
            return 0

        elif msg == win32con.WM_ERASEBKGND:
            return 1

        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def _message_loop(self):
        """Цикл сообщений"""
        while self._running:
            try:
                win32gui.PumpWaitingMessages()
                time.sleep(0.005)
            except Exception as e:
                if self._running:
                    time.sleep(0.01)