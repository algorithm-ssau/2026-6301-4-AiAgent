# Alcohol Censor — AI-агент цензуры алкоголя

Приложение захватывает экран в реальном времени, обнаруживает бутылки алкоголя с помощью нейросети YOLO и рисует поверх них чёрные прямоугольники через прозрачный оверлей. Работает на Windows и Linux.

## Запуск

### EXE на диске (скачиваемый)

Установите по ссылке и откройте его. Обученная модель находится на C:\Users\username\AppData

### EXE в проекте

Готовый `AlcoholCensor.exe` собирается через PyInstaller и включает все зависимости и модель:

```bash
pip install pyinstaller
pyinstaller AlcoholCensor.spec
```

Результат — `dist/AlcoholCensor.exe`. Запускается двойным кликом, консольного окна нет (`console=False`). Папка `Models/` упаковывается внутрь исполняемого файла автоматически через `.spec`.

### Из исходников

```bash
pip install -r requirements.txt
python main.py
```

Откроется панель управления. Выберите модель, задайте порог уверенности и целевой FPS, нажмите **«Включить цензуру»** — поверх экрана появится невидимый оверлей с чёрными прямоугольниками поверх алкоголя.

> Для запуска необходим файл `Models/best.onnx`. Если его нет — обучите модель.

### Linux

```bash
sudo pacman -S xdg-desktop-portal-hyprland pipewire gstreamer \
               gst-plugin-pipewire gst-plugins-base \
               python-dbus python-gobject gtk-layer-shell
python main.py
```

При первом запуске Hyprland покажет диалог выбора монитора для захвата. Wayland-бэкенд реализован, но не подключён в `main.py`; принудительно использовать X11: `XDG_SESSION_TYPE=x11 python main.py`.

### AMD GPU

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

После этого `_select_device()` автоматически определит видеокарту AMD через стандартный интерфейс `torch.cuda.is_available()`.

## Тесты

```bash
python -m pytest Tests/                                                   # все тесты
python -m pytest Tests/test_tracker.py                                    # только трекер (без модели)
python -m pytest Tests/test_tracker.py::TestTrackSmoother::test_backprojection
```

`Tests/test_detector.py` требует наличия `Models/best.onnx`. `Tests/test_tracker.py` работает автономно через `Detection`-датаклассы.

## Обучение модели

```bash
pip install -r requirements-train.txt
python train.py
```

Скачивает датасет **Alcohol Computer Vision Model** (adonantonin/alcohol-iaeeq, v4, 7332 изображений, 5 классов алкоголя) с Roboflow, обучает `yolo26n.pt` на 50 эпох и предлагает сохранить результат как `Models/best.onnx` или в `Models/archive/`.

## Архитектура

### Конвейер обработки

```
Захват экрана → resize 640×640 → YOLO detect() → TrackSmoother → Overlay
```

~20 FPS на CPU с ONNX-моделью.

### Потоковая модель

| Поток | Роль |
|-------|------|
| Main (tkinter) | GUI, `root.mainloop()` |
| CaptureThread | Захват кадров → `Queue(maxsize=3)` (дропает старые при переполнении) |
| PipelineThread | Детекция → сглаживание → `overlay.update_boxes()` |
| StatsThread | Мониторинг FPS, обновление статус-строки каждые 2 с |
| Overlay (daemon) | Win32 message loop, GDI-рендер |

Все обновления UI из рабочих потоков — только через `root.after(0, callback)`.

### Компоненты

**`Core/capture.py` — `ScreenCapturer`**  
Диспетчер бэкендов: MSS на Windows/X11, `NotImplementedError` на Wayland. Принимает `monitor_index` и `fps_limit` для троттлинга захвата.

**`Core/backends/mss_backend.py` — `MssBackend`**  
Захват через `mss`: BGRA → BGR (OpenCV), ограничение FPS через `time.perf_counter()`. `list_monitors()` — статический метод для GUI.

**`Core/detector.py` — `BottleDetector`**  
Оборачивает Ultralytics YOLO. `.pt`-файлы фильтруют COCO-класс 39 (bottle); `.onnx`-файлы без фильтра (кастомная модель, класс 0). `track()` — алиас `detect()`, возвращает `track_id=None`. Для ONNX автоматически патчит `ort.InferenceSession` через `_patch_ort_speed()`: выбирает лучший провайдер (TensorRT → CUDA → CoreML → DirectML → CPU) и включает `ORT_ENABLE_ALL`.

**`Core/tracker.py` — `TrackSmoother`**  
Получает `Detection`-объекты в пространстве 640×640, проецирует в экранные координаты, сопоставляет с существующими треками по IoU + центроидной дистанции, применяет EMA-сглаживание (`ema_alpha=0.6`). Держит прямоугольники `decay_frames=6` кадров после исчезновения объекта. Поддерживает `set_params()` для live-настройки и `get_statistics()` для отладки.

**`Core/utils.py` — `FpsCounter`**  
Потокобезопасный sliding-window счётчик: `.tick()` на каждый кадр, `.get()` возвращает FPS за последние 2 секунды.

**`Overlay/windows/overlay_win.py` — `WindowsOverlay`**  
Win32-окно с флагами `WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE`. Прозрачный цвет-ключ — маджента `RGB(255,0,255)`, чёрный зарезервирован под прямоугольники. Рендер через двойной буфер GDI (CreateCompatibleDC + BitBlt). Обновления из pipeline-потока — через `PostMessage(WM_APP_UPDATE)`.

**`Overlay/linux/overlay_lin.py` — `LinuxOverlay`**  
GTK3 + gtk-layer-shell (протокол wlr-layer-shell). Реализован, в `main.py` пока не подключён.

**`gui.py` — `CensorApp`**  
Панель управления: выбор модели, слайдер порога (0.10–0.90), спиннер монитора (1–8), спиннер FPS (10–60). Изменение порога во время работы передаётся live в `detector.set_conf()` и `smoother.set_params()`.

**`main.py`**  
Точка входа. Создаёт именованный мьютекс `Local\AlcoholCensorSingleInstance` (Windows) для защиты от двойного запуска, затем запускает `CensorApp`.

## Участники

### Артём Виряскин — Захват экрана (Windows/X11)
**Файлы:** `Core/capture.py`, `Core/backends/mss_backend.py`

Реализовал абстракцию `CaptureBackend` и диспетчер `ScreenCapturer` с автовыбором бэкенда по платформе. Написал `MssBackend`: захват через `mss`, конвертация BGRA→BGR, ограничение FPS через `time.perf_counter()`, `list_monitors()` для GUI. Добавил улучшение захвата и исправил определение видеокарты.

Ключевые коммиты: `f06325e` `3f202ae` `c69247f` `0955707` `c2f9305` `114969b`

---

### Игорь Бурментьев — Детекция объектов
**Файлы:** `Core/detector.py`, `Tests/test_detector.py`

Реализовал `BottleDetector` на базе Ultralytics YOLO с автоопределением устройства (CUDA/AMD ROCm/CPU). Добавил патч ONNX Runtime (`_patch_ort_speed`) для автовыбора GPU-провайдеров и включения `ORT_ENABLE_ALL`. Реализовал автовыбор фильтра классов: COCO класс 39 для `.pt`, без фильтра для `.onnx`-моделей. Написал тесты детектора, не требующие GPU. Добавил флаг `WDA_EXCLUDEFROMCAPTURE` в оверлей — чёрные прямоугольники не захватываются скриншотером, что исключает обнаружение собственного оверлея как бутылки.

Ключевые коммиты: `4488daa` (детектор) · `8148aad` (тесты) · `00896c2` (AMD GPU) · `4276a5c` (тесты без GPU) · `97aa128` (квадраты не попадают в захват)

---

### Михаил — Сглаживание треков
**Файлы:** `Core/tracker.py`, `Tests/test_tracker.py`

Реализовал `TrackSmoother`: обратная проекция координат 640×640 → экран, сопоставление детекций с треками по IoU + центроидной дистанции, EMA-сглаживание, decay-механизм для временно пропавших объектов. Написал тесты на обратную проекцию, EMA-сходимость и decay. Исправил мигание прямоугольников.

Ключевые коммиты: `0dca28d` `c4afcaa` `5f7027f` `9524753` `66c5ee2` `84200a8`

---

### Александр Солдатов — Обучение модели + Оверлей Linux + Захват Wayland
**Файлы:** `train.py`, `Overlay/base.py`, `Core/backends/wayland_backend.py`, `Overlay/linux/overlay_lin.py`, `Models/best.onnx`

Обучил модель детекции алкоголя на датасете Roboflow (7332 изображений, 5 классов), экспортировал в ONNX. Написал `train.py` с автоматическим скачиванием датасета и ротацией архивных моделей. Определил интерфейс `OverlayBase`. Реализовал `WaylandBackend` (захват через xdg-desktop-portal + GStreamer/PipeWire) и `LinuxOverlay` на GTK3 + gtk-layer-shell.

Ключевые коммиты: `651cb28` `8a0768d` `462c055` `b82757c` `1575353`

---

### Артём Сурков — Оверлей Windows + FpsCounter
**Файлы:** `Overlay/windows/overlay_win.py`, `Core/utils.py`

Реализовал `FpsCounter` с потокобезопасным sliding-window. Написал `WindowsOverlay`: регистрация Win32-класса, прозрачное layered-окно (`WS_EX_LAYERED | WS_EX_TRANSPARENT`), рендер через DIBSection и `UpdateLayeredWindow`, потокобезопасные обновления через `PostMessage(WM_APP_UPDATE)`.

Ключевые коммиты: `7b9a9e2` `d73ddbf` `958f560` `ae30e53` `c7e35bc`

---

### Максим Павлов — GUI и интеграция
**Файлы:** `gui.py`, `main.py`, `requirements.txt`

Написал панель управления на tkinter: выбор модели, слайдер порога, выбор монитора и FPS, live-обновление параметров детектора во время работы, индикация производительности. Собрал весь конвейер в `main.py` (мьютекс, потоки, очереди, обработка ошибок). Итеративно улучшал производительность: убрал 100% загрузку CPU, устранил зависание при закрытии.

Ключевые коммиты: `fa34cf4` `0b1641b` `bdb0e3b` `7e245e2`
