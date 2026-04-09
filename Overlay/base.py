from abc import ABC, abstractmethod
from typing import List, Tuple

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


class OverlayBase(ABC):
    @abstractmethod
    def start(self) -> None:
        """Создать окно и запустить рендер-цикл в отдельном потоке."""

    @abstractmethod
    def stop(self) -> None:
        """Завершить поток и закрыть окно."""

    @abstractmethod
    def update_boxes(self, boxes: List[Box]) -> None:
        """Обновить список прямоугольников. Вызывается из любого потока."""

    @abstractmethod
    def is_running(self) -> bool:
        """Возвращает True, если рендер-поток жив."""
