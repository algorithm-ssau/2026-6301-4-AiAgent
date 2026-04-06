"""
Windows оверлей с использованием pywin32
"""

import win32api
import win32con
import win32gui
import threading
from typing import List, Tuple, Optional

from Overlay.base import OverlayBase


class WindowsOverlay(OverlayBase):
    """Windows оверлей с прозрачным окном поверх всех приложений"""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

        # Параметры окна
        self._hwnd = None
        self._running = False
        self._thread = None

    def initialize(self) -> bool:
        """Создание и инициализация оверлейного окна"""

        pass

    def draw_boxes(self, boxes: List[Tuple[int, int, int, int]],
                   labels: Optional[List[str]] = None,
                   colors: Optional[List[Tuple[int, int, int]]] = None):
        """Отрисовка рамок"""

        pass

    def clear(self):
        """Очистка всех рамок"""
        pass

    def update(self):
        """Обновление оверлея"""
        pass

    def hide(self):
        """Скрыть оверлей"""
        pass

    def show(self):
        """Показать оверлей"""
        pass

    def close(self):
        """Закрыть оверлей"""
        pass

    def is_running(self) -> bool:
        """Проверить, работает ли оверлей"""
        return self._running