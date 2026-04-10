# Техническое задание: AI-агент цензуры алкоголя

Приложение захватывает экран в реальном времени, находит на нём бутылки алкоголя
с помощью нейросети и рисует поверх них чёрные прямоугольники через прозрачный
оверлей. Работает на Windows и Linux (Hyprland).

---

## Структура проекта

```
2026-6301-4-AiAgent/
├── main.py                          — точка входа, запускает всё
├── gui.py                           — панель управления (tkinter)
├── requirements.txt                 — зависимости для запуска
├── CLAUDE.md                        — заметки для разработчиков
│
├── Core/
│   ├── capture.py                   — захват экрана (общий интерфейс + диспетчер)
│   ├── detector.py                  — детекция бутылок через YOLO26
│   ├── tracker.py                   — сглаживание координат между кадрами
│   ├── utils.py                     — FpsCounter и другие утилиты
│   └── backends/
│       ├── mss_backend.py           — захват экрана на Windows / X11
│       └── wayland_backend.py       — захват экрана на Hyprland через PipeWire
│
├── Overlay/
│   ├── base.py                      — общий интерфейс оверлея
│   ├── windows/
│   │   └── overlay_win.py           — оверлей для Windows
│   └── linux/
│       └── overlay_lin.py           — оверлей для Linux (GTK3 + gtk-layer-shell)
│
├── Models/
│   └── best.onnx                    — обученная модель (~10 МБ)
│
└── Tests/
    ├── test_detector.py
    └── test_tracker.py
```

## Файлы по участникам

| Участник | Файлы |
|----------|-------|
| 1 — Артем Виряскин | `Core/capture.py`, `Core/backends/mss_backend.py` |
| 2 — Игорь | `Core/detector.py`, `Tests/test_detector.py` |
| 3 — Миша | `Core/tracker.py`, `Tests/test_tracker.py` |
| 4 — Саша | `train.py`, `Overlay/base.py`, `Core/backends/wayland_backend.py`, `Overlay/linux/overlay_lin.py`, `Models/best.onnx` |
| 5 — Артем Сурков | `Core/utils.py`, `Overlay/windows/overlay_win.py` |
| 6 — Макс | `gui.py`, `main.py`, `requirements.txt` |

---

## Порядок работы

```
ФАЗА 0 — только Участник 4 (Linux, самый мощный компьютер)
  Найти датасет на Roboflow → написать train.py → запустить обучение
  Обучение идёт само (несколько часов). Пока оно идёт — начинается Фаза 1.

ФАЗА 1 — параллельно, пока модель обучается
  Участник 1: Core/capture.py + Core/backends/mss_backend.py
  Участник 2: Core/detector.py
  Участник 3: Tests/test_tracker.py (заглушки)  ← не ждёт никого
  Участник 4: Overlay/base.py + Core/backends/wayland_backend.py + Overlay/linux/overlay_lin.py
  Участник 5: Core/utils.py (FpsCounter)  ← не ждёт никого

  Когда обучение завершилось: Участник 4 экспортирует best.onnx и коммитит его.
  С этого момента все тестируют на реальной модели.

ФАЗА 2 — после фазы 1
  Участник 3: Core/tracker.py               (нужен Detection от Участника 2)
  Участник 5: Overlay/windows/overlay_win.py (нужен base.py от Участника 4)

ФАЗА 3 — когда все остальные закончили
  Участник 6: gui.py + main.py + requirements.txt
```

---

## Участник 1 (Артем Виряскин) — Захват экрана (Windows/X11)

**Система:** Windows
**Файлы:** `Core/capture.py`, `Core/backends/mss_backend.py`

### Что нужно сделать

Написать код, который делает скриншот экрана и возвращает его как картинку
(numpy array в формате BGR). Плюс — общий интерфейс, который позволит
автоматически выбирать нужный бэкенд в зависимости от системы.

## Подробная документация

### **`Core/capture.py`** — два класса:

1. **`CaptureBackend`** — абстрактный класс (шаблон) с методами:
   - `start()` — инициализация бэкенда, проверка доступности монитора
   - `grab() -> np.ndarray` — захват текущего кадра, возвращает numpy array в формате BGR (height, width, channels)
   - `stop()` — освобождение ресурсов, закрытие соединений
   - `get_monitor_rect() -> dict` — возвращает словарь с ключами `{"top", "left", "width", "height"}` для текущего монитора

2. **`ScreenCapturer`** — обёртка с автоопределением бэкенда:
   - **Автоматический выбор бэкенда:**
     - Проверяет `os.environ.get("XDG_SESSION_TYPE") == "wayland"` для определения Wayland
     - На Wayland пытается использовать `WaylandBackend` (пока не реализован, выбрасывает `NotImplementedError`)
     - Во всех остальных случаях (Windows, macOS, X11) использует `MssBackend`
   - **Публичные методы:**
     - `__init__(monitor_index=1, fps_limit=20)` — инициализация с указанием монитора и ограничением FPS
     - `start()` — запуск захвата, инициализация выбранного бэкенда
     - `grab() -> np.ndarray` — захват кадра в формате BGR
     - `stop()` — остановка захвата и очистка ресурсов
     - `get_monitor_rect() -> dict` — получение размеров и позиции текущего монитора
   - **Дополнительные возможности:**
     - `backend_type` — свойство (property), возвращающее имя используемого бэкенда ("MSS" или "Wayland")
     - `list_monitors() -> list` — статический метод для получения списка всех мониторов без инициализации

### **`Core/backends/mss_backend.py`** — класс `MssBackend`:

- **Назначение:** захват экрана с помощью библиотеки `mss` (работает на Windows, macOS, X11)
- **Инициализация:** `__init__(monitor_index=1, fps_limit=20)`
  - `monitor_index` — индекс монитора (1 = основной, 2, 3... для дополнительных)
  - `fps_limit` — максимальное количество кадров в секунду (0 = без ограничений)

- **Метод `start()`:**
  - Создаёт экземпляр `mss.mss()`
  - Проверяет валидность `monitor_index` через `list_monitors()`
  - При неверном индексе выбрасывает `ValueError` с информацией о доступных мониторах
  - Сохраняет информацию о мониторе в `self._monitor_rect`

- **Метод `grab() -> np.ndarray`:**
  - Реализует ограничение FPS через `time.perf_counter()` и `time.sleep()`
  - Захватывает кадр через `self._sct.grab()` — возвращает изображение в формате BGRA
  - Конвертирует BGRA → BGR с помощью `cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)`
  - Возвращает numpy array размерности (height, width, 3) с dtype uint8
  - Выбрасывает `RuntimeError`, если `start()` не был вызван

- **Метод `stop()`:**
  - Закрывает соединение через `self._sct.close()`
  - Обнуляет внутренние переменные

- **Метод `get_monitor_rect() -> dict`:**
  - Возвращает копию словаря с ключами `top`, `left`, `width`, `height`
  - Выбрасывает `RuntimeError`, если `start()` не был вызван

- **Статический метод `list_monitors() -> List[dict]`:**
  - Создаёт временный экземпляр `mss.mss()` в контекстном менеджере
  - Возвращает список мониторов, пропуская индекс 0 (виртуальный экран, объединяющий все мониторы)
  - Каждый монитор представлен словарём с ключами: `left`, `top`, `width`, `height`
  - Может использоваться без вызова `start()`, удобно для GUI

### **Особенности реализации:**

1. **Обработка ошибок:**
   - Проверка существования монитора при старте
   - Проверка инициализации бэкенда перед вызовом методов
   - Информативные сообщения об ошибках

2. **Управление FPS:**
   - Вычисление интервала между кадрами: `1.0 / fps_limit`
   - Замер времени через `time.perf_counter()` для высокой точности
   - Принудительная задержка через `time.sleep()` при превышении FPS

3. **Формат изображения:**
   - Вход: BGRA (32-bit, 4 канала) от MSS
   - Выход: BGR (24-bit, 3 канала) через OpenCV
   - Совместимость с большинством CV алгоритмов

### **Пример использования:**

```python
from Core.capture import ScreenCapturer

# Автоматическое определение бэкенда
capturer = ScreenCapturer(monitor_index=1, fps_limit=30)
capturer.start()

try:
    # Получение информации о мониторе
    rect = capturer.get_monitor_rect()
    print(f"Захват монитора: {rect['width']}x{rect['height']}")
    
    # Бесконечный захват кадров
    while True:
        frame = capturer.grab()  # BGR numpy array
        # Обработка кадра...
finally:
    capturer.stop()
```

### **Зависимости:**
- `mss` — захват экрана
- `opencv-python` (cv2) — конвертация цветовых пространств
- `numpy` — работа с массивами изображений

### Коммиты
1. `feat: добавлен абстрактный CaptureBackend и диспетчер ScreenCapturer`
2. `feat: добавлен MssBackend для захвата экрана через mss`
3. `feat: добавлено ограничение FPS в MssBackend`
4. `feat: добавлена валидация монитора и list_monitors()`

---

## Участник 2 (Игорь) — Детекция объектов

**Система:** Windows
**Файлы:** `Core/detector.py`, `Tests/test_detector.py`

### Что нужно сделать

Написать код детектора — принимает кадр экрана 640×640, передаёт в нейросеть,
возвращает список найденных бутылок с координатами. Обучение модели делает
Участник 4, но код пишется параллельно — тестировать можно на `yolo26n.pt`
(COCO), потом заменить на `best.onnx` через `git pull`.

### Подробно

**Датакласс `Detection`** — структура для одной найденной бутылки:
```python
@dataclass
class Detection:
    x1: int       # левый край
    y1: int       # верхний край
    x2: int       # правый край
    y2: int       # нижний край
    conf: float   # уверенность модели (0.0–1.0)
    track_id: int | None  # ID трека (None если трекинг не использовался)
```

**Класс `BottleDetector`**:
- `__init__(model_path, conf=0.35)`
  - `model_path` — путь к файлу модели (.pt или .onnx)
  - `conf` — минимальная уверенность для детекции
  - Загрузить модель: `self._model = YOLO(model_path)`
  - Определить классы автоматически:
    - Если файл `.pt` (COCO) → `self._classes = [39]` (bottle)
    - Если файл `.onnx` (кастомная модель) → `self._classes = None` (все классы)
  - Почему: при обучении на своём датасете бутылка получает класс `0`, а не `39`.
    Передача `classes=[39]` с кастомной моделью даст пустой результат.

- `track(frame_bgr) -> List[Detection]`
  - Запустить детекцию + трекинг ByteTrack
  - Вызов: `self._model.track(frame_bgr, persist=True, conf=..., classes=..., tracker="bytetrack.yaml", verbose=False)`
  - Из результата взять `boxes.xyxy` (координаты), `boxes.conf`, `boxes.id`
  - Если `boxes.id` равен None (треков нет) — вернуть пустой список
  - Координаты привести к `int`

- `detect(frame_bgr) -> List[Detection]`
  - Только детекция без трекинга, `track_id=None`
  - Нужен для тестов

- `reset_tracker()`
  - Сбросить состояние ByteTrack при перезапуске
  - Если `self._model.predictor` существует → вызвать `trackers[0].reset()`

> Важно: координаты в `Detection` — в пространстве входного кадра (640×640),
> не в экранных координатах. Обратная проекция — задача Участника 3.

### Проверка работы

Написать `Tests/test_detector.py` и запустить `pytest Tests/test_detector.py`:

```python
from Core.detector import BottleDetector, Detection
import numpy as np

def test_detect_returns_list():
    # детектор не падает и возвращает список
    detector = BottleDetector("yolo26n.pt")
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert isinstance(result, list)

def test_detect_track_id_is_none():
    # detect() не запускает трекинг — track_id всегда None
    detector = BottleDetector("yolo26n.pt")
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    for det in detector.detect(frame):
        assert det.track_id is None

def test_reset_tracker_does_not_raise():
    detector = BottleDetector("yolo26n.pt")
    detector.reset_tracker()  # не должен падать
```

### Коммиты
1. `feat: добавлен датакласс Detection`
2. `feat: добавлен BottleDetector с автоопределением классов для pt/onnx`
3. `feat: добавлены методы track() и detect()`
4. `feat: добавлен метод reset_tracker()`
5. `test: добавлены тесты детектора`

---

## Участник 3 (Миша) — Сглаживание треков

**Система:** Windows
**Файлы:** `Core/tracker.py`, `Tests/test_tracker.py`
**Ждёт:** класс `Detection` от Участника 2 (только для Фазы 2)

### Что нужно сделать

**Фаза 1** (пока Участник 2 пишет детектор) — написать заглушки в `Tests/test_tracker.py`.
Функции с `pass` внутри, чтобы зафиксировать что именно будет проверяться.

**Фаза 2** (после Участника 2) — написать сам `Core/tracker.py` и наполнить тесты реальной логикой.

Код берёт "сырые" координаты от нейросети (в пространстве
640×640) и превращает их в стабильные координаты на экране. Без этого
прямоугольники будут дёргаться от кадра к кадру.

### Подробно

**Класс `TrackSmoother`**:
- `__init__(ema_alpha=0.6, decay_frames=10)`
  - `ema_alpha` — насколько быстро прямоугольник следует за объектом
    (0.0 = не двигается, 1.0 = мгновенно следует)
  - `decay_frames` — сколько кадров держать прямоугольник, если бутылка
    временно не видна

- `update(detections, screen_w, screen_h, monitor_left=0, monitor_top=0, input_w=640, input_h=640) -> List[Tuple[int,int,int,int]]`
  - Принимает список `Detection` из детектора
  - Возвращает список `(x1, y1, x2, y2)` в **экранных координатах**
  - Внутри хранит словарь `_tracks` по `track_id`

- `reset()` — очистить все треки

**Шаг 1 — обратная проекция** (640×640 → экран):
```
scale_x = screen_w / input_w
scale_y = screen_h / input_h
x1_screen = int(det.x1 * scale_x) + monitor_left
y1_screen = int(det.y1 * scale_y) + monitor_top
```

**Шаг 2 — EMA-сглаживание** для каждого `track_id`:
```
# Первый раз — просто запомнить
# Каждый следующий кадр:
new_x1 = int(alpha * new_x1 + (1 - alpha) * old_x1)
# то же для y1, x2, y2
```

**Шаг 3 — Decay** (плавное исчезновение):
- Каждый кадр увеличивать счётчик `age` у треков, которые не были обнаружены
- Если `age > decay_frames` — удалить трек
- Треки с `age > 0` всё равно включать в вывод (держим прямоугольник)

### Проверка работы

Написать `Tests/test_tracker.py` и запустить `pytest Tests/test_tracker.py`:

```python
from Core.tracker import TrackSmoother
from Core.detector import Detection

def test_backprojection():
    # пиксель (320, 0) в 640×640 → x=960 на экране 1920×1080
    smoother = TrackSmoother()
    det = Detection(x1=320, y1=0, x2=640, y2=640, conf=0.9, track_id=1)
    boxes = smoother.update([det], screen_w=1920, screen_h=1080)
    assert boxes[0][0] == 960

def test_ema_converges():
    # после 10 одинаковых кадров x1=100 в 640×640 → x=300 на экране 1920×1080
    # (scale_x = 1920/640 = 3.0, EMA сходится к 100 * 3.0 = 300)
    smoother = TrackSmoother()
    det = Detection(x1=100, y1=100, x2=200, y2=200, conf=0.9, track_id=1)
    for _ in range(10):
        boxes = smoother.update([det], screen_w=1920, screen_h=1080)
    assert boxes[0][0] == 300

def test_decay_removes_track():
    # трек исчезает через decay_frames кадров без обновления
    smoother = TrackSmoother(decay_frames=3)
    det = Detection(x1=0, y1=0, x2=100, y2=100, conf=0.9, track_id=1)
    smoother.update([det], screen_w=1920, screen_h=1080)
    for _ in range(4):  # 4 кадра без детекции
        boxes = smoother.update([], screen_w=1920, screen_h=1080)
    assert len(boxes) == 0

def test_reset_clears_tracks():
    smoother = TrackSmoother()
    det = Detection(x1=0, y1=0, x2=100, y2=100, conf=0.9, track_id=1)
    smoother.update([det], screen_w=1920, screen_h=1080)
    smoother.reset()
    boxes = smoother.update([], screen_w=1920, screen_h=1080)
    assert len(boxes) == 0
```

### Коммиты
1. `test: добавлены заглушки тестов трекера`
2. `feat: добавлен TrackSmoother с хранением состояния треков`
3. `feat: добавлена обратная проекция координат в экранное пространство`
4. `feat: добавлено EMA-сглаживание для каждого track_id`
5. `feat: добавлена логика decay для потерянных треков`
6. `test: добавлены тесты трекера`

---

## Участник 4 (Саша) — Обучение модели + Оверлей Linux + Захват Wayland

**Система:** Linux (Hyprland)
**Файлы:** `train.py`, `Overlay/base.py`, `Core/backends/wayland_backend.py`, `Overlay/linux/overlay_lin.py`

### Что нужно сделать

Четыре задачи в таком порядке:
1. **Фаза 0** — найти датасет, написать `train.py`, запустить обучение на своём железе
2. **Фаза 1, пока идёт обучение** — написать `Overlay/base.py` (нужен Участнику 5)
3. **Фаза 1** — написать `WaylandBackend` для захвата экрана
4. **Фаза 1** — написать `LinuxOverlay`

### Подробно

#### Часть 1 — Датасет и обучение модели (Фаза 0)

#### Шаг 1 — Датасет

Используется датасет **Alcohol Computer Vision Model** (adonantonin/alcohol-iaeeq, версия 4):
- 7332 изображения, split: 81% train / 10% val / 9% test
- 5 классов: `alcohol-bottle`, `beer-bottle`, `beer-glass`, `shot`, `wine-glass`
- Формат: **YOLO26**

#### Шаг 2 — `train.py`

Запустить:
```bash
pip install -r requirements-train.txt
python train.py
```

После обучения скрипт спросит:
```
Сохранить как основную модель (best.onnx)? [y/N]:
```
- `y` — старая `best.onnx` уходит в `Models/archive/best_YYYYMMDD.onnx`, новая становится `Models/best.onnx`
- `n` / Enter — новая сохраняется в `Models/archive/model_YYYYMMDD.onnx`, `best.onnx` не трогается

Содержимое `requirements-train.txt`:
```
ultralytics>=8.3.0
roboflow>=1.1.0
```

Структура `Models/`:
```
Models/
├── best.onnx             ← в git (актуальная модель)
└── archive/              ← в .gitignore (локальные архивы)
    ├── best_20260321.onnx
    └── model_20260321.onnx
```

Закоммитить и сообщить команде:
```bash
git add Models/best.onnx train.py
git commit -m "feat: добавлена обученная модель детекции алкоголя"
git push
# Написать команде — пусть делают git pull
```

#### Часть 2 — `Overlay/base.py` ✅ ГОТОВО

Абстрактный класс `OverlayBase` с методами:
- `start()` — создать окно и запустить в отдельном потоке
- `stop()` — завершить поток
- `update_boxes(boxes)` — обновить список прямоугольников (thread-safe)
- `is_running() -> bool` — жив ли поток

Тип `Box = Tuple[int, int, int, int]` — координаты `(x1, y1, x2, y2)`.

Отправить Участнику 5 (нужен для `overlay_win.py`).

#### Часть 3 — `Core/backends/wayland_backend.py`

Класс `WaylandBackend` для захвата экрана на Hyprland.

Как это работает:
1. Python обращается к `xdg-desktop-portal-hyprland` через D-Bus
2. Hyprland показывает диалог "разрешить захват экрана?" — пользователь выбирает монитор
3. Portal возвращает PipeWire stream (node_id + file descriptor)
4. GStreamer читает кадры из этого потока через `pipewiresrc`
5. `appsink` отдаёт кадры в Python как numpy array

Методы те же, что у `MssBackend`: `start()`, `grab()`, `stop()`, `get_monitor_rect()`.

Внутри `start()`:
- Открыть D-Bus сессию с `org.freedesktop.portal.Desktop`
- Создать ScreenCast сессию, выбрать источник (монитор), запустить
- Получить `node_id` и `pipewire_fd`
- Запустить GStreamer pipeline:
  ```
  pipewiresrc fd={fd} path={node_id} ! videoconvert ! video/x-raw,format=BGR ! appsink
  ```
- Из первого кадра узнать разрешение экрана и сохранить в `_monitor_rect`

Внутри `grab()`:
- `appsink.emit("pull-sample")` → достать буфер → numpy array BGR
- Если сэмпл None — вернуть пустой кадр того же размера

Установка зависимостей:
```bash
sudo pacman -S xdg-desktop-portal-hyprland pipewire gstreamer \
               gst-plugin-pipewire gst-plugins-base python-dbus python-gobject
```

#### Часть 4 — `Overlay/linux/overlay_lin.py`

Класс `LinuxOverlay(OverlayBase)` — прозрачное окно поверх всего экрана.

Используется `gtk-layer-shell` (протокол wlr-layer-shell, Hyprland поддерживает).

```bash
sudo pacman -S gtk-layer-shell python-gobject
```

Создание окна:
```python
import gi
gi.require_version("Gtk", "3.0")
gi.require_version("GtkLayerShell", "0.1")
from gi.repository import Gtk, GtkLayerShell, Gdk, GLib
import cairo

window = Gtk.Window()
GtkLayerShell.init_for_window(window)
GtkLayerShell.set_layer(window, GtkLayerShell.Layer.OVERLAY)
GtkLayerShell.set_anchor(window, GtkLayerShell.Edge.TOP, True)
GtkLayerShell.set_anchor(window, GtkLayerShell.Edge.LEFT, True)
GtkLayerShell.set_exclusive_zone(-1)
```

RGBA прозрачность:
```python
screen = window.get_screen()
visual = screen.get_rgba_visual()
if visual:
    window.set_visual(visual)
window.set_app_paintable(True)
```

Click-through (мышь проходит сквозь окно):
```python
region = cairo.Region()
window.input_shape_combine_region(region)
```

Draw-обработчик (рисует чёрные прямоугольники):
```python
def _on_draw(self, widget, cr):
    cr.set_operator(cairo.OPERATOR_CLEAR)
    cr.paint()                    # прозрачный фон
    cr.set_operator(cairo.OPERATOR_OVER)
    cr.set_source_rgba(0, 0, 0, 1)
    for x1, y1, x2, y2 in self._boxes:
        cr.rectangle(x1, y1, x2 - x1, y2 - y1)
        cr.fill()
```

GTK работает в отдельном потоке. Из pipeline-потока **нельзя** вызывать GTK напрямую.
Обновление прямоугольников только через:
```python
def update_boxes(self, boxes):
    with self._lock:
        self._boxes = list(boxes)
    GLib.idle_add(self._window.queue_draw)
```

### Коммиты
1. `feat: добавлен train.py со скачиванием датасета и обучением модели`
2. `feat: добавлена обученная модель детекции алкоголя (best.onnx)`
3. `feat: добавлен абстрактный OverlayBase и тип Box`
4. `feat: добавлен WaylandBackend с захватом экрана через xdg-portal`
5. `feat: добавлен GStreamer PipeWire пайплайн для захвата кадров`
6. `feat: добавлено окно LinuxOverlay на gtk-layer-shell`
7. `feat: добавлена Cairo-отрисовка и click-through`

---

## Участник 5 (Артем Сурков) — Оверлей Windows + FpsCounter

**Система:** Windows
**Файлы:** `Overlay/windows/overlay_win.py`, `Core/utils.py`
**Ждёт:** `Overlay/base.py` от Участника 4 (только для overlay)

### Что нужно сделать

Две задачи, первую можно делать сразу в Фазе 1 не дожидаясь никого:
1. Написать `Core/utils.py` с `FpsCounter`
2. Написать оверлей для Windows (после получения `base.py` от Участника 4)

### Подробно

#### Часть 1 — `Core/utils.py` с `FpsCounter` (делать первым, в Фазе 1)

`FpsCounter` — простой потокобезопасный счётчик FPS. Используется в `pipeline_loop`,
читается GUI для отображения текущей скорости.

```python
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
```

#### Часть 2 — `Overlay/windows/overlay_win.py` (после получения base.py от Участника 4)

**Класс `WindowsOverlay(OverlayBase)`**.

Используется библиотека `pywin32`.

**Создание окна с нужными флагами:**
```python
import win32api, win32con, win32gui
import ctypes

ex_style = (win32con.WS_EX_LAYERED    # попиксельная прозрачность
          | win32con.WS_EX_TRANSPARENT # мышь проходит насквозь
          | win32con.WS_EX_TOPMOST     # поверх всех окон
          | win32con.WS_EX_NOACTIVATE) # не перехватывать фокус
style = win32con.WS_POPUP              # без рамки и заголовка
```

**Рисование** через 32-bit ARGB DIBSection:
- Создать bitmap в памяти
- Заполнить `0x00000000` (полностью прозрачно)
- Нарисовать прямоугольники цветом `0xFF000000` (чёрный непрозрачный)
- Передать в систему через `UpdateLayeredWindow` с флагом `ULW_ALPHA`

**Message loop** в отдельном потоке:
```python
WM_APP_UPDATE = win32con.WM_APP + 1

def _message_loop(self):
    while True:
        # получать сообщения
        # если WM_QUIT — выйти
        # если WM_APP_UPDATE — вызвать self._render()
```

**`update_boxes(boxes)`** из pipeline-потока:
```python
with self._lock:
    self._boxes = list(boxes)
win32gui.PostMessage(self._hwnd, WM_APP_UPDATE, 0, 0)
```

`PostMessage` — единственный безопасный способ общения с Win32 окном
из другого потока. Не вызывать Win32 функции рисования напрямую из pipeline.

### Коммиты
1. `feat: добавлен FpsCounter в Core/utils.py`
2. `feat: добавлен WindowsOverlay с регистрацией Win32 окна`
3. `feat: добавлено прозрачное layered окно с WS_EX_LAYERED`
4. `feat: добавлена отрисовка через DIBSection и UpdateLayeredWindow`
5. `feat: добавлено сообщение WM_APP_UPDATE для потокобезопасных обновлений`

---

## Участник 6 (Макс) — GUI + Сборка

**Система:** Windows
**Файлы:** `gui.py`, `main.py`, `requirements.txt`
**Ждёт:** всех остальных

### Что нужно сделать

Две задачи:
1. Написать панель управления (`gui.py`) с кнопками включения/выключения цензуры
2. Собрать всё вместе в `main.py`

### Подробно

#### `gui.py` — панель управления

Небольшое окно на `tkinter` (встроен в Python, ничего устанавливать не нужно).

**Как выглядит:**
```
┌─────────────────────────────┐
│   Alcohol Censor            │
│                             │
│   Модель: yolo26n.pt  [...]  │
│   Порог:  [──●──────] 0.35  │
│   Монитор: [1 ▾]            │
│                             │
│   [  Включить цензуру  ]    │
│                             │
│   Статус: выключено         │
│   FPS: —                    │
└─────────────────────────────┘
```

**Элементы:**
- Поле с путём к модели + кнопка "..." для выбора файла через диалог
- Слайдер порога уверенности (`conf`) от 0.1 до 0.9, шаг 0.05
- Выпадающий список для выбора монитора
- Большая кнопка "Включить цензуру" / "Выключить цензуру" (текст меняется)
- Строка статуса: "выключено" / "работает"
- Строка с текущим FPS (обновляется раз в секунду через `after()`)

**Как работает кнопка:**

При нажатии "Включить":
```python
def _on_toggle(self):
    if not self._running:
        self._start_censorship()
    else:
        self._stop_censorship()

def _start_censorship(self):
    # 1. Создать ScreenCapturer и запустить
    # 2. Создать BottleDetector с выбранной моделью и conf
    # 3. Создать TrackSmoother
    # 4. Создать и запустить оверлей (LinuxOverlay или WindowsOverlay)
    # 5. Запустить потоки capture_loop и pipeline_loop
    # 6. Обновить кнопку и статус
    self._running = True
    self._btn.config(text="Выключить цензуру")
    self._status.config(text="Статус: работает")

def _stop_censorship(self):
    # 1. stop_event.set()
    # 2. overlay.stop()
    # 3. capturer.stop()
    # 4. join потоки с timeout=2
    # 5. Сбросить всё в None
    # 6. Обновить кнопку и статус
    self._running = False
    self._btn.config(text="Включить цензуру")
    self._status.config(text="Статус: выключено")
    self._fps_label.config(text="FPS: —")
```

**FPS-счётчик** — pipeline-поток пишет текущий FPS в переменную, GUI читает её
через `self.after(1000, self._update_fps)`:
```python
def _update_fps(self):
    if self._running:
        fps = self._fps_counter.get()
        self._fps_label.config(text=f"FPS: {fps:.1f}")
    self.after(1000, self._update_fps)
```

> Важно: tkinter не thread-safe. Никогда не обновлять виджеты из capture/pipeline
> потоков напрямую. Только через `self.after(0, callback)`.

#### `main.py`

Просто запускает GUI:
```python
from gui import CensorApp
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = CensorApp(root)
    root.mainloop()
```

#### Функции пайплайна (в `main.py` или отдельном модуле)

**`capture_loop(capturer, frame_queue, stop_event)`**:
- В цикле: захватить кадр → положить в очередь
- Если очередь полна (maxsize=2) — выбросить старый кадр, положить новый
- Выходить при `stop_event.is_set()`

**`pipeline_loop(detector, smoother, overlay, frame_queue, stop_event, W, H, mon_rect, fps_counter)`**:
- Взять кадр из очереди → `cv2.resize(frame, (640, 640))`
- `detections = detector.track(small_frame)`
- `boxes = smoother.update(detections, W, H, mon_rect["left"], mon_rect["top"])`
- `overlay.update_boxes(boxes)`
- Считать FPS: `fps_counter.tick()`
- Выходить при `stop_event.is_set()`

#### Выбор оверлея по платформе (внутри `_start_censorship`)

```python
import sys

def create_overlay(screen_w, screen_h):
    if sys.platform == "win32":
        from Overlay.windows.overlay_win import WindowsOverlay
        return WindowsOverlay(screen_w, screen_h)
    else:
        from Overlay.linux.overlay_lin import LinuxOverlay
        return LinuxOverlay(screen_w, screen_h)
```

#### `requirements.txt` — для запуска приложения

```
ultralytics>=8.3.0
onnxruntime>=1.18.0
opencv-python-headless>=4.10.0
numpy>=1.26.0
mss>=9.0.1
pywin32>=306; sys_platform=="win32"
pytest>=8.0.0

# Linux — устанавливается через pacman, не pip:
# sudo pacman -S xdg-desktop-portal-hyprland pipewire gstreamer
#               gst-plugin-pipewire gst-plugins-base
#               python-dbus python-gobject gtk-layer-shell
```

### Коммиты
1. `feat: добавлена панель управления gui.py на tkinter`
2. `feat: добавлено включение/выключение цензуры с управлением оверлеем`
3. `feat: добавлен счётчик FPS и отображение статуса`
4. `feat: добавлен main.py и функции пайплайна`
5. `chore: добавлен requirements.txt`
