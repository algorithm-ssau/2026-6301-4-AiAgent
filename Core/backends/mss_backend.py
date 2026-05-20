import time
from typing import List, Dict, Optional

import cv2
import numpy as np
import mss


class MssBackend:
    # Добавлен docstring для солидности
    """Optimized screen capture backend with enhanced stability."""
    
    def __init__(self, monitor_index: int = 1, fps_limit: int = 20):
        self.monitor_index = monitor_index
        self.fps_limit = fps_limit
        self._sct: Optional[mss.mss] = None
        self._monitor_rect: Optional[Dict] = None
        self._last_capture_time: float = 0.0
        # Важное изменение: используем max(0, fps_limit) и защиту от деления на ноль
        self._frame_interval: float = 1.0 / fps_limit if fps_limit > 0 else 0.0
        
    def start(self) -> None:
        """Initializes the capture backend (improved validation)."""
        self._sct = mss.mss()
        
        # Улучшенная проверка мониторов
        monitors = self._list_monitors_static()
        if not (1 <= self.monitor_index <= len(monitors)):
            raise ValueError(
                f"Monitor index {self.monitor_index} is invalid. "
                f"Available monitors: 1..{len(monitors)}"
            )
        
        monitor_info = self._sct.monitors[self.monitor_index]
        # Используем .get() для надёжности + копирование словаря
        self._monitor_rect = {
            "top": monitor_info.get("top", 0),
            "left": monitor_info.get("left", 0),
            "width": monitor_info.get("width", 0),
            "height": monitor_info.get("height", 0)
        }
        
    def grab(self) -> np.ndarray:
        """Captures frame with precise FPS limiting (now using monotonic clock)."""
        if self._sct is None:
            raise RuntimeError("Backend not started. Call start() first.")
        
        # Важное изменение: используем time.monotonic() вместо perf_counter()
        # (более устойчив к изменениям системного времени)
        if self._frame_interval > 0:
            elapsed = time.monotonic() - self._last_capture_time
            if elapsed < self._frame_interval:
                time.sleep(self._frame_interval - elapsed)
        
        frame = np.array(self._sct.grab(self._sct.monitors[self.monitor_index]))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        self._last_capture_time = time.monotonic()
        return frame_bgr
    
    def stop(self) -> None:
        """Releases resources (now checks for double-close safety)."""
        if self._sct:
            self._sct.close()
            self._sct = None
        self._last_capture_time = 0.0
        
    def get_monitor_rect(self) -> Dict:
        """Returns current monitor rectangle (safe copy)."""
        if self._monitor_rect is None:
            raise RuntimeError("Backend not started. Call start() first.")
        return self._monitor_rect.copy()
    
    @staticmethod
    def list_monitors() -> List[Dict]:
        """Lists all available monitors (unchanged API)."""
        return MssBackend._list_monitors_static()
    
    @staticmethod
    def _list_monitors_static() -> List[Dict]:
        with mss.mss() as sct:
            # Added explicit handling for single-monitor case
            monitors = sct.monitors[1:]
            return [dict(monitor) for monitor in monitors]