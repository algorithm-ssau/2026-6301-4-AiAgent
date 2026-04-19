"""
Windows оверлей с использованием pywin32
"""
import ctypes

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
        self._hdc = None  # Контекст устройства
        self._hbitmap = None  # Handle битмапа
        self._bits = None  # Указатель на пиксельные данные
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

            ex_style = (win32con.WS_EX_LAYERED |
                        win32con.WS_EX_TRANSPARENT |
                        win32con.WS_EX_TOPMOST |
                        win32con.WS_EX_NOACTIVATE)

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

            self._create_dib_section()

            # Запускаем поток обработки сообщений
            self._running = True
            self._thread = threading.Thread(target=self._message_loop, daemon=True)
            self._thread.start()

            print(f"Оверлей с DIBSection запущен: {self.width}x{self.height}")

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

        if self._hbitmap:
            win32gui.DeleteObject(self._hbitmap)
            self._hbitmap = None

        if self._hdc:
            win32gui.DeleteDC(self._hdc)
            self._hdc = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        print("Оверлей остановлен")

    def update_boxes(self, boxes: List[Box]) -> None:
        """Обновить список прямоугольников. Вызывается из любого потока."""
        with self._lock:
            self._boxes = list(boxes)


    def is_running(self) -> bool:
        """Возвращает True, если рендер-поток жив."""
        return self._running and self._thread and self._thread.is_alive()

    def _create_dib_section(self):
        screen_dc = win32gui.GetDC(0)
        try:
            self._hdc =win32gui.CreateCompatibleDC(screen_dc)

            # Настройки 32-bit битмапа
            bi = win32gui.BITMAPINFO()
            bi.bmiHeader.biSize = win32gui.GetBitmapInfoHeaderSize()
            bi.bmiHeader.biWidth = self.width
            bi.bmiHeader.biHeight = -self.height  # Отрицательная = сверху вниз
            bi.bmiHeader.biPlanes = 1
            bi.bmiHeader.biBitCount = 32
            bi.bmiHeader.biCompression = win32con.BI_RGB
            bi.bmiHeader.biSizeImage = 0
            # Создаём DIBSection и получаем указатель на пиксели
            self._hbitmap, self._bits = win32gui.CreateDIBSection(
                screen_dc, bi, win32con.DIB_RGB_COLORS, None, 0, 0
            )

            if not self._hbitmap:
                raise RuntimeError("Не удалось создать DIBSection")

            win32gui.SelectObject(self._hdc, self._hbitmap)
            self._clear_bitmap()

        finally:
            win32gui.ReleaseDC(0, screen_dc)

    def _clear_bitmap(self):
        """Очистка битмапа (заполнение прозрачным цветом)."""
        if self._bits:
            bitmap_size = self.width * self.height * 4
            ctypes.memset(self._bits, 0, bitmap_size)

    def _draw_rectangle(self, x1: int, y1: int, x2: int, y2: int, r: int, g: int, b: int):
        """Рисование прямоугольника на битмапе."""
        if not self._bits:
            return

        x1 = max(0, min(x1, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        x2 = max(0, min(x2, self.width - 1))
        y2 = max(0, min(y2, self.height - 1))

        if x1 >= x2 or y1 >= y2:
            return

        color = (255 << 24) | (r << 16) | (g << 8) | b
        bits_ptr = ctypes.cast(self._bits, ctypes.POINTER(ctypes.c_uint32))

        thickness = 3

        # Вертикальные линии
        for offset in range(thickness):
            left_x = x1 + offset
            right_x = x2 - offset
            if left_x >= self.width or right_x < 0:
                continue
            for y in range(y1, y2 + 1):
                if 0 <= y < self.height:
                    if 0 <= left_x < self.width:
                        idx = y * self.width + left_x
                        bits_ptr[idx] = color
                    if 0 <= right_x < self.width and right_x != left_x:
                        idx = y * self.width + right_x
                        bits_ptr[idx] = color

        # Горизонтальные линии
        for offset in range(thickness):
            top_y = y1 + offset
            bottom_y = y2 - offset
            if top_y >= self.height or bottom_y < 0:
                continue
            for x in range(x1, x2 + 1):
                if 0 <= x < self.width:
                    if 0 <= top_y < self.height:
                        idx = top_y * self.width + x
                        bits_ptr[idx] = color
                    if 0 <= bottom_y < self.height and bottom_y != top_y:
                        idx = bottom_y * self.width + x
                        bits_ptr[idx] = color

    def _render(self):
        """Отрисовка всех рамок на битмапе и обновление окна."""
        self._clear_bitmap()

        # Копируем данные с блокировкой
        with self._lock:
            boxes = list(self._boxes)

        # Рисуем все рамки (зелёным цветом)
        for box in boxes:
            x1, y1, x2, y2 = box
            self._draw_rectangle(x1, y1, x2, y2, 0, 255, 0)

        self._update_layered_window()

    def _update_layered_window(self):
        """Обновление layered окна с новым содержимым через UpdateLayeredWindow."""
        if not self._hwnd or not self._hdc:
            return

        pt = (0, 0)
        size = (self.width, self.height)

        blend = win32api.BLENDFUNCTION()
        blend.BlendOp = win32con.AC_SRC_OVER
        blend.BlendFlags = 0
        blend.SourceConstantAlpha = 255
        blend.AlphaFormat = win32con.AC_SRC_ALPHA

        try:
            win32gui.UpdateLayeredWindow(
                self._hwnd, None, pt, size,
                self._hdc, pt, 0, blend, win32con.ULW_ALPHA
            )
        except Exception as e:
            print(f"Ошибка UpdateLayeredWindow: {e}")

    def _window_procedure(self, hwnd, msg, wparam, lparam):
        """Обработчик сообщений Win32 окна."""
        if msg == win32con.WM_DESTROY:
            self._running = False
            return 0
        elif msg == win32con.WM_ERASEBKGND:
            return 1
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