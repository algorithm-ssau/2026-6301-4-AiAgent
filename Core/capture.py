import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class CaptureBackend(ABC):
    
    @abstractmethod
    def start(self) -> None:
        pass
    
    @abstractmethod
    def grab(self) -> np.ndarray:

        pass
    
    @abstractmethod
    def stop(self) -> None:
        pass
    
    @abstractmethod
    def get_monitor_rect(self) -> Dict:

        pass


class ScreenCapturer:
    
    def __init__(self, monitor_index: int = 1, fps_limit: int = 20):

        self.monitor_index = monitor_index
        self.fps_limit = fps_limit
        self._backend: Optional[CaptureBackend] = None
        self._backend_type: str = ""
        
        # Auto-detect and initialize backend
        self._init_backend()
        
    def _init_backend(self) -> None:
        if os.name == "posix":  # Unix-like system
            xdg_session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
            if xdg_session_type == "wayland":
                self._init_wayland_backend()
                return
        
        # Default to MSS (works on Windows, macOS, X11)
        self._init_mss_backend()
        
    def _init_mss_backend(self) -> None:
        from Core.backends.mss_backend import MssBackend
        
        self._backend = MssBackend(
            monitor_index=self.monitor_index,
            fps_limit=self.fps_limit
        )
        self._backend_type = "MSS"
        
    def _init_wayland_backend(self) -> None:
        raise NotImplementedError(
            "Wayland backend is not implemented yet. "
            "Please use X11 or switch to MSS backend by unsetting XDG_SESSION_TYPE"
        )
    
    def start(self) -> None:

        if self._backend is None:
            raise RuntimeError("No backend available")
        self._backend.start()
        
    def grab(self) -> np.ndarray:

        if self._backend is None:
            raise RuntimeError("No backend available")
        return self._backend.grab()
    
    def stop(self) -> None:
        if self._backend:
            self._backend.stop()
            
    def get_monitor_rect(self) -> Dict:

        if self._backend is None:
            raise RuntimeError("No backend available")
        return self._backend.get_monitor_rect()
    
    @property
    def backend_type(self) -> str:
        return self._backend_type
    
    @staticmethod
    def list_monitors() -> list:

        from Core.backends.mss_backend import MssBackend
        return MssBackend.list_monitors()