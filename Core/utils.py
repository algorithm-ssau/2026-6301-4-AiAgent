import threading
import time

class FpsCounter:
    def __init__(self, window=2.0):
        # window — за сколько последних секунд считать FPS
        self._lock = threading.Lock()
        self._timestamps = []
        self._window = window

    def tick(self) -> None:
        """Вызывать каждый раз когда обработан кадр."""
        now = time.perf_counter()
        with self._lock:
            self._timestamps.append(now)
            # удалить старые метки за пределами окна
            cutoff = now - self._window
            self._timestamps = [t for t in self._timestamps if t >= cutoff]

    def get(self) -> float:
        """Вернуть текущий FPS."""
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            span = self._timestamps[-1] - self._timestamps[0]
            if span == 0:
                return 0.0
            return (len(self._timestamps) - 1) / span