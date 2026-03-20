import ctypes
import ctypes.wintypes
import threading
import time
from collections import deque

# --- Константы Win32 ---
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
WS_EX_TOPMOST = 0x8
WS_POPUP = 0x80000000

WM_PAINT = 0x000F
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_SIZE = 0x0005
WM_ERASEBKGND = 0x0014

# ... и так далее

class WindowsOverlay:
    def __init__(self):
        self.hwnd = None
        self.running = False
        self.rectangles_to_draw = []  # Сюда будут класться новые координаты
        self.fps = 0
        self.fps_buffer = deque(maxlen=30)  # Для сглаживания FPS
        self.last_time = time.perf_counter()

    def create_overlay(self):
        # 1. Зарегистрировать класс окна
        # 2. Создать окно с расширенными стилями (WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST)
        # 3. Сделать окно видимым
        # 4. Вернуть hwnd
        pass

    def window_procedure(self, hwnd, msg, wParam, lParam):
        # Обработчик сообщений
        if msg == WM_PAINT:
            self.on_paint()
        elif msg == WM_DESTROY:
            ctypes.windll.user32.PostQuitMessage(0)
        return ctypes.windll.user32.DefWindowProcW(hwnd, msg, wParam, lParam)

    def on_paint(self):
        # Рисуем:
        # 1. Заливаем фон прозрачным
        # 2. Рисуем все прямоугольники из self.rectangles_to_draw
        # 3. Рисуем текст с self.fps
        pass

    def update_rectangles(self, new_rects):
        """Вызывается из главного потока (Участник 6)"""
        self.rectangles_to_draw = new_rects
        # Просим окно перерисоваться
        ctypes.windll.user32.InvalidateRect(self.hwnd, None, True)

    def calculate_fps(self):
        """Считаем FPS внутри цикла сообщений"""
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps_buffer.append(current_fps)
            self.fps = sum(self.fps_buffer) / len(self.fps_buffer)

    def run_message_loop(self):
        """Запускается в отдельном потоке, чтобы не блокировать основной GUI"""
        self.create_overlay()
        self.running = True

        msg = ctypes.wintypes.MSG()
        while self.running:
            # PeekMessage не блокирует поток, если нет сообщений
            ret = ctypes.windll.user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1)
            if ret != 0:
                ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
                ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))

            # Здесь мы можем обновлять FPS и рисовать, даже если нет сообщений
            self.calculate_fps()
            # Можно принудительно просить перерисовку, если координаты обновились
            # InvalidateRect уже вызывается в update_rectangles

            # Небольшая задержка, чтобы не нагружать процессор в простое
            time.sleep(0.001)

    def start(self):
        thread = threading.Thread(target=self.run_message_loop, daemon=True)
        thread.start()

    def stop(self):
        self.running = False
        ctypes.windll.user32.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)