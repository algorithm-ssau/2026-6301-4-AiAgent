"""
Windows оверлей с использованием pywin32
"""

import win32api
import win32con
import win32gui
import threading
import time
from typing import List, Tuple

from Overlay.base import OverlayBase, Box


class WindowsOverlay(OverlayBase):
    """Windows оверлей с прозрачным окном поверх всех приложений"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Параметры окна
        self._hwnd = None
        self._running = False
        self._thread = None

        # Потокобезопасные данные
        self._lock = threading.Lock()
        self._boxes: List[Box] = []

    def start(self) -> None:
        """Создать окно и запустить рендер-цикл в отдельном потоке."""
        try:
            # Регистрируем класс окна
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = self._window_procedure
            wc.lpszClassName = "AlcoholCensorOverlay"
            wc.hInstance = win32api.GetModuleHandle(None)

            class_atom = win32gui.RegisterClass(wc)
            if not class_atom:
                raise RuntimeError("Не удалось зарегистрировать класс окна")

            # Создаём обычное окно (пока без расширенных стилей)
            style = win32con.WS_POPUP
            ex_style = 0

            self._hwnd = win32gui.CreateWindowEx(
                ex_style,
                "AlcoholCensorOverlay",
                "Alcohol Censor Overlay",
                style,
                0, 0, self.width, self.height,
                None, None, wc.hInstance, None
            )

            if not self._hwnd:
                raise RuntimeError("Не удалось создать окно")

            # Показываем окно
            win32gui.ShowWindow(self._hwnd, win32con.SW_SHOW)

            # Запускаем поток обработки сообщений
            self._running = True
            self._thread = threading.Thread(target=self._message_loop, daemon=True)
            self._thread.start()

            print(f"Оверлей запущен: {self.width}x{self.height}")

        except Exception as e:
            print(f"Ошибка запуска оверлея: {e}")
            self._running = False

    def stop(self) -> None:
        """Завершить поток и закрыть окно."""
        self._running = False

        if self._hwnd:
            win32gui.PostMessage(self._hwnd, win32con.WM_QUIT, 0, 0)
            time.sleep(0.1)
            win32gui.DestroyWindow(self._hwnd)
            self._hwnd = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        print("Оверлей остановлен")

    def update_boxes(self, boxes: List[Box]) -> None:
        """Обновить список прямоугольников. Вызывается из любого потока."""
        with self._lock:
            self._boxes = list(boxes)
        # Пока просто печатаем для отладки
        print(f"Получено рамок: {len(boxes)}")

    def is_running(self) -> bool:
        """Возвращает True, если рендер-поток жив."""
        return self._running and self._thread and self._thread.is_alive()

    def _window_procedure(self, hwnd, msg, wparam, lparam):
        """Обработчик сообщений Win32 окна."""
        if msg == win32con.WM_DESTROY:
            self._running = False
            return 0
        elif msg == win32con.WM_PAINT:
            # Пока просто белый фон для видимости окна
            hdc, paint_struct = win32gui.BeginPaint(hwnd)
            win32gui.FillRect(hdc, (0, 0, self.width, self.height), win32gui.GetStockObject(win32con.WHITE_BRUSH))
            win32gui.EndPaint(hwnd, paint_struct)
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def _message_loop(self):
        """Цикл обработки сообщений Windows."""
        while self._running:
            try:
                msg = win32gui.GetMessage(None, 0, 0)
                if msg and msg[0] != 0:
                    win32gui.TranslateMessage(msg[1])
                    win32gui.DispatchMessage(msg[1])
                else:
                    time.sleep(0.001)
            except Exception as e:
                if self._running:
                    print(f"Ошибка в цикле сообщений: {e}")