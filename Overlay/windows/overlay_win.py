"""
Windows оверлей с отрисовкой рамок (без заливки)
"""
import threading
import time
from typing import List, Tuple

import win32api
import win32con
import win32gui


Box = Tuple[int, int, int, int]


class WindowsOverlay:
    """Windows оверлей с прозрачным окном и отрисовкой рамок"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._hwnd = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._boxes: List[Box] = []

        # Цвет рамки (зеленый в формате RGB)
        self._pen_color = (0, 255, 0)
        self._pen_width = 3

    def start(self) -> None:
        """Создать окно"""
        try:
            # Регистрируем класс окна
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = self._window_procedure
            self._class_name = f"AlcoholCensorOverlay_{id(self)}"
            wc.lpszClassName = self._class_name
            wc.hInstance = win32api.GetModuleHandle(None)
            wc.hbrBackground = win32gui.GetStockObject(win32con.NULL_BRUSH)  # Прозрачный фон

            try:
                win32gui.RegisterClass(wc)
            except win32gui.error as e:
                # 1410 = класс окна уже существует
                if e.winerror != 1410:
                    raise

            ex_style = (win32con.WS_EX_LAYERED |
                        win32con.WS_EX_TRANSPARENT |
                        win32con.WS_EX_TOPMOST |
                        win32con.WS_EX_NOACTIVATE)

            style = win32con.WS_POPUP

            self._hwnd = win32gui.CreateWindowEx(
                ex_style,
                self._class_name,
                "Alcohol Censor Overlay",
                style,
                0, 0, self.width, self.height,
                None, None, wc.hInstance, None
            )

            if not self._hwnd:
                raise RuntimeError("Не удалось создать окно")

            # Устанавливаем прозрачность (ключевой цвет - черный)
            win32gui.SetLayeredWindowAttributes(
                self._hwnd,
                0,  # черный цвет будет прозрачным
                0,  # альфа не используется при LWA_COLORKEY
                win32con.LWA_COLORKEY  # используем цветовой ключ вместо альфа
            )

            # Делаем окно кликабельным (пропускаем клики)
            win32gui.SetWindowLong(
                self._hwnd,
                win32con.GWL_EXSTYLE,
                win32gui.GetWindowLong(self._hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_TRANSPARENT
            )

            win32gui.ShowWindow(self._hwnd, win32con.SW_SHOW)

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
            win32gui.DestroyWindow(self._hwnd)
            self._hwnd = None
        print("Оверлей остановлен")

    def update_boxes(self, boxes: List[Box]) -> None:
        """Обновить список боксов и вызвать перерисовку"""
        with self._lock:
            self._boxes = list(boxes)
        # Принудительно перерисовываем окно
        if self._hwnd:
            win32gui.InvalidateRect(self._hwnd, None, True)
            win32gui.UpdateWindow(self._hwnd)

    def _draw_rectangles(self, hdc):
        """Нарисовать прямоугольники на контексте устройства"""
        if not hdc:
            return

        with self._lock:
            boxes = list(self._boxes)

        if not boxes:
            return

        try:
            # Создаем перо (кисть) для рисования
            pen = win32gui.CreatePen(
                win32con.PS_SOLID,  # сплошная линия
                self._pen_width,  # толщина
                win32api.RGB(self._pen_color[0], self._pen_color[1], self._pen_color[2])  # цвет
            )

            # Выбираем перо в контекст
            old_pen = win32gui.SelectObject(hdc, pen)


            # Рисуем каждый прямоугольник ТОЛЬКО КОНТУР
            for box in boxes:
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    # Убеждаемся, что координаты в пределах окна
                    x1 = max(0, min(x1, self.width))
                    y1 = max(0, min(y1, self.height))
                    x2 = max(0, min(x2, self.width))
                    y2 = max(0, min(y2, self.height))

                    if x1 < x2 and y1 < y2:
                        # Рисуем только контур прямоугольника
                        win32gui.MoveToEx(hdc, x1, y1)
                        win32gui.LineTo(hdc, x2, y1)  # верхняя линия
                        win32gui.LineTo(hdc, x2, y2)  # правая линия
                        win32gui.LineTo(hdc, x1, y2)  # нижняя линия
                        win32gui.LineTo(hdc, x1, y1)  # левая линия

            # Восстанавливаем старые объекты
            win32gui.SelectObject(hdc, old_pen)

            # Удаляем созданное перо
            win32gui.DeleteObject(pen)

        except Exception as e:
            print(f"Ошибка рисования: {e}")

    def _window_procedure(self, hwnd, msg, wparam, lparam):
        """Обработчик сообщений"""
        if msg == win32con.WM_PAINT:
            # Начинаем рисование
            hdc, paint_struct = win32gui.BeginPaint(hwnd)
            try:
                # Очищаем фон (делаем прозрачным)
                # Создаем прямоугольник для всей области окна
                rect = (0, 0, self.width, self.height)
                # Заполняем черным цветом (который станет прозрачным через LWA_COLORKEY)
                win32gui.PatBlt(hdc, 0, 0, self.width, self.height, win32con.BLACKNESS)
                # Рисуем прямоугольники поверх
                self._draw_rectangles(hdc)
            finally:
                # Завершаем рисование
                win32gui.EndPaint(hwnd, paint_struct)
            return 0

        elif msg == win32con.WM_DESTROY:
            self._running = False
            return 0

        elif msg == win32con.WM_ERASEBKGND:
            # Возвращаем 1, чтобы не стирать фон
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
