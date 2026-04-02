import time
from typing import List, Dict, Optional

import cv2
import numpy as np
import mss


class MssBackend:
    
    def __init__(self, monitor_index: int = 1, fps_limit: int = 20):
        self.monitor_index = monitor_index
        self.fps_limit = fps_limit
        self._sct: Optional[mss.mss] = None
        self._monitor_rect: Optional[Dict] = None
        self._last_capture_time: float = 0
        self._frame_interval: float = 1.0 / fps_limit if fps_limit > 0 else 0
        
    def start(self) -> None:
        self._sct = mss.mss()
        
        # Validate monitor index
        monitors = self._list_monitors_static()  # Changed to static method call
        if self.monitor_index < 1 or self.monitor_index > len(monitors):
            raise ValueError(
                f"Monitor index {self.monitor_index} is invalid. "
                f"Available monitors: 1..{len(monitors)}"
            )
        
        # Get monitor rect (MSS uses 1-based indexing)
        monitor_info = self._sct.monitors[self.monitor_index]
        self._monitor_rect = {
            "top": monitor_info["top"],
            "left": monitor_info["left"],
            "width": monitor_info["width"],
            "height": monitor_info["height"]
        }
        
    def grab(self) -> np.ndarray:

        if self._sct is None:
            raise RuntimeError("Backend not started. Call start() first.")
        
        # FPS limiting
        if self._frame_interval > 0:
            elapsed = time.perf_counter() - self._last_capture_time
            if elapsed < self._frame_interval:
                time.sleep(self._frame_interval - elapsed)
        
        # Capture frame (returns BGRA)
        frame = np.array(self._sct.grab(self._sct.monitors[self.monitor_index]))
        
        # Convert BGRA to BGR (remove alpha channel)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        self._last_capture_time = time.perf_counter()
        return frame_bgr
    
    def stop(self) -> None:
        if self._sct:
            self._sct.close()
            self._sct = None
        self._last_capture_time = 0
        
    def get_monitor_rect(self) -> Dict:

        if self._monitor_rect is None:
            raise RuntimeError("Backend not started. Call start() first.")
        return self._monitor_rect.copy()
    
    @staticmethod
    def list_monitors() -> List[Dict]:
        return MssBackend._list_monitors_static()
    
    @staticmethod
    def _list_monitors_static() -> List[Dict]:
        with mss.mss() as sct:
            # Skip monitor[0] which is the combined virtual screen
            return [dict(monitor) for monitor in sct.monitors[1:]]