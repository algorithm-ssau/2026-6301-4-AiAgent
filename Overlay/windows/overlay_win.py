"""
Windows оверлей с отрисовкой черных прямоугольников
"""
import threading
import time
from typing import List, Tuple

import win32api
import win32con
import win32gui


Box = Tuple[int, int, int, int]


class WindowsOverlay:
    """Windows оверлей с прозрачным окном и черной заливкой"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._hwnd = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._boxes: List[Box] = []

        self._class_name = f"AlcoholCensorOverlay_{id(self)}"

        # Цвет, который будет прозрачным.
        # ВАЖНО: не черный, иначе черные прямоугольники тоже станут прозрачными.
        self._transparent_color = win32api.RGB(255, 0, 255)

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
                # 1410 = класс окна уже существует
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

            # Фиолетовый цвет будет прозрачным.
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

            print(f"Оверлей запущен: {self.width}x{self.height}")

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

        print("Оверлей остановлен")

    def update_boxes(self, boxes: List[Box]) -> None:
        """Обновить список боксов и вызвать перерисовку"""
        with self._lock:
            self._boxes = list(boxes)

        if self._hwnd:
            try:
                win32gui.InvalidateRect(self._hwnd, None, True)
                win32gui.UpdateWindow(self._hwnd)
            except Exception:
                pass

    def is_running(self) -> bool:
        return self._running

    def _fill_rectangles(self, hdc):
        """Нарисовать черные залитые прямоугольники"""
        if not hdc:
            return

        with self._lock:
            boxes = list(self._boxes)

        if not boxes:
            return

        brush = None
        old_brush = None
        old_pen = None

        try:
            brush = win32gui.CreateSolidBrush(win32api.RGB(0, 0, 0))
            old_brush = win32gui.SelectObject(hdc, brush)

            # Убираем обводку, оставляем только заливку.
            null_pen = win32gui.GetStockObject(win32con.NULL_PEN)
            old_pen = win32gui.SelectObject(hdc, null_pen)

            for box in boxes:
                if len(box) != 4:
                    continue

                x1, y1, x2, y2 = box

                x1 = max(0, min(int(x1), self.width))
                y1 = max(0, min(int(y1), self.height))
                x2 = max(0, min(int(x2), self.width))
                y2 = max(0, min(int(y2), self.height))

                if x1 < x2 and y1 < y2:
                    win32gui.Rectangle(hdc, x1, y1, x2, y2)

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

            try:
                # Заполняем фон прозрачным цветом.
                bg_brush = win32gui.CreateSolidBrush(self._transparent_color)
                rect = (0, 0, self.width, self.height)
                win32gui.FillRect(hdc, rect, bg_brush)
                win32gui.DeleteObject(bg_brush)

                # Поверх рисуем черные залитые прямоугольники.
                self._fill_rectangles(hdc)

            finally:
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
                time.sleep(0.01)
            except Exception as e:
                if self._running:
                    print(f"Ошибка в цикле сообщений: {e}")
                    time.sleep(0.1)