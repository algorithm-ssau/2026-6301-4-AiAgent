"""
Windows оверлей с отрисовкой черных прямоугольников
Оптимизированная версия для высокой производительности
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
        self._dirty = True
        self._last_boxes_hash = 0

        self._class_name = f"AlcoholCensorOverlay_{id(self)}"
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

    def update_boxes(self, boxes: List[Box]) -> None:
        """Обновить список боксов с проверкой изменений"""
        # Хэшируем для быстрого сравнения
        boxes_hash = hash(tuple(tuple(b) for b in boxes))

        if boxes_hash == self._last_boxes_hash:
            return  # Нет изменений

        self._last_boxes_hash = boxes_hash

        with self._lock:
            self._boxes = list(boxes)
            self._dirty = True

        if self._hwnd:
            try:
                win32gui.InvalidateRect(self._hwnd, None, False)
            except Exception:
                pass

    def is_running(self) -> bool:
        return self._running

    def _fill_rectangles(self, hdc):
        """Оптимизированная отрисовка с batch-операциями"""
        if not hdc:
            return

        with self._lock:
            boxes = self._boxes
            self._dirty = False

        if not boxes:
            return

        brush = None
        old_brush = None
        old_pen = None

        try:
            # Создаем черную кисть один раз
            brush = win32gui.CreateSolidBrush(win32api.RGB(0, 0, 0))
            old_brush = win32gui.SelectObject(hdc, brush)

            # Убираем обводку
            null_pen = win32gui.GetStockObject(win32con.NULL_PEN)
            old_pen = win32gui.SelectObject(hdc, null_pen)

            # Отрисовка всех прямоугольников
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

            # Двойная буферизация для плавности
            mem_dc = win32gui.CreateCompatibleDC(hdc)
            bitmap = win32gui.CreateCompatibleBitmap(hdc, self.width, self.height)
            old_bitmap = win32gui.SelectObject(mem_dc, bitmap)

            try:
                # Заполняем фон прозрачным цветом
                bg_brush = win32gui.CreateSolidBrush(self._transparent_color)
                rect = (0, 0, self.width, self.height)
                win32gui.FillRect(mem_dc, rect, bg_brush)
                win32gui.DeleteObject(bg_brush)

                # Рисуем прямоугольники
                self._fill_rectangles(mem_dc)

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
                time.sleep(0.005)  # Уменьшена задержка для более быстрого отклика
            except Exception as e:
                if self._running:
                    time.sleep(0.01)