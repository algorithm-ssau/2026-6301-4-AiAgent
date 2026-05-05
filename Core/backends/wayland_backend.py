import re
import threading
import time
from typing import Dict, Optional

import dbus
import dbus.mainloop.glib
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

Gst.init(None)

_REQUEST_IFACE = "org.freedesktop.portal.Request"
_SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"


class WaylandBackend:
    def __init__(self, monitor_index: int = 1, fps_limit: int = 20):
        self.monitor_index = monitor_index
        self.fps_limit = fps_limit
        self._frame_interval: float = 1.0 / fps_limit if fps_limit > 0 else 0
        self._last_capture_time: float = 0

        self._pipeline: Optional[Gst.Pipeline] = None
        self._appsink: Optional[Gst.Element] = None
        self._monitor_rect: Optional[Dict] = None

        self._bus: Optional[dbus.SessionBus] = None
        self._portal = None
        self._session = None

        self._loop: Optional[GLib.MainLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        self._ready = threading.Event()
        self._error: Optional[str] = None

        self._req_counter = 0
        self._ses_counter = 0
        self._sender_name: str = ""

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def start(self) -> None:
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self._bus = dbus.SessionBus()
        self._sender_name = re.sub(r"\.", "_", self._bus.get_unique_name()[1:])
        self._portal = self._bus.get_object(
            "org.freedesktop.portal.Desktop",
            "/org/freedesktop/portal/desktop",
        )

        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(target=self._loop.run, daemon=True)
        self._loop_thread.start()

        # Запускаем асинхронное D-Bus согласование внутри главного цикла
        GLib.idle_add(self._create_session)

        # Ждём готовности пайплайна (или ошибки)
        if not self._ready.wait(timeout=30):
            raise RuntimeError("Timeout waiting for xdg-desktop-portal response")
        if self._error:
            raise RuntimeError(self._error)

    def grab(self) -> np.ndarray:
        if self._appsink is None:
            raise RuntimeError("Backend not started. Call start() first.")

        if self._frame_interval > 0:
            elapsed = time.perf_counter() - self._last_capture_time
            if elapsed < self._frame_interval:
                time.sleep(self._frame_interval - elapsed)

        sample = self._appsink.emit("pull-sample")
        self._last_capture_time = time.perf_counter()

        if sample is None:
            w = self._monitor_rect["width"] if self._monitor_rect else 1920
            h = self._monitor_rect["height"] if self._monitor_rect else 1080
            return np.zeros((h, w, 3), dtype=np.uint8)

        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        w = struct.get_int("width").value
        h = struct.get_int("height").value

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return np.zeros((h, w, 3), dtype=np.uint8)
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
        buf.unmap(mapinfo)
        return frame

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._appsink = None
        if self._loop is not None:
            self._loop.quit()
            self._loop = None

    def get_monitor_rect(self) -> Dict:
        if self._monitor_rect is None:
            raise RuntimeError("Backend not started. Call start() first.")
        return self._monitor_rect.copy()

    # ------------------------------------------------------------------
    # Согласование с D-Bus порталом (выполняется внутри GLib mainloop)
    # ------------------------------------------------------------------

    def _new_request_path(self):
        self._req_counter += 1
        token = f"u{self._req_counter}"
        path = f"/org/freedesktop/portal/desktop/request/{self._sender_name}/{token}"
        return path, token

    def _new_session_path(self):
        self._ses_counter += 1
        token = f"u{self._ses_counter}"
        path = f"/org/freedesktop/portal/desktop/session/{self._sender_name}/{token}"
        return path, token

    def _portal_call(self, method, callback, *args, options=None):
        """Регистрируем слушатель Response до вызова метода портала."""
        if options is None:
            options = {}
        request_path, request_token = self._new_request_path()
        self._bus.add_signal_receiver(
            callback,
            signal_name="Response",
            dbus_interface=_REQUEST_IFACE,
            bus_name="org.freedesktop.portal.Desktop",
            path=request_path,
        )
        options["handle_token"] = request_token
        method(*(args + (options,)), dbus_interface=_SCREENCAST_IFACE)

    def _create_session(self):
        _, session_token = self._new_session_path()
        self._portal_call(
            self._portal.CreateSession,
            self._on_create_session,
            options={"session_handle_token": session_token},
        )

    def _on_create_session(self, response, results):
        if response != 0:
            self._error = f"CreateSession failed (response={response})"
            self._ready.set()
            return
        self._session = results["session_handle"]
        self._portal_call(
            self._portal.SelectSources,
            self._on_select_sources,
            self._session,
            options={"multiple": False, "types": dbus.UInt32(1)},  # 1 = Монитор
        )

    def _on_select_sources(self, response, results):
        if response != 0:
            self._error = f"SelectSources failed (response={response})"
            self._ready.set()
            return
        self._portal_call(
            self._portal.Start,
            self._on_start,
            self._session,
            "",
        )

    def _on_start(self, response, results):
        if response != 0:
            self._error = f"Start failed (response={response})"
            self._ready.set()
            return

        streams = results.get("streams", [])
        if not streams:
            self._error = "No streams returned by portal"
            self._ready.set()
            return

        node_id = int(streams[0][0])

        fd_obj = self._portal.OpenPipeWireRemote(
            self._session,
            dbus.Dictionary(signature="sv"),
            dbus_interface=_SCREENCAST_IFACE,
        )
        fd = fd_obj.take()

        try:
            self._start_pipeline(node_id, int(fd))
            self._monitor_rect = self._detect_resolution()
        except Exception as e:
            self._error = str(e)
        finally:
            self._ready.set()

    # ------------------------------------------------------------------
    # GStreamer пайплайн
    # ------------------------------------------------------------------

    def _start_pipeline(self, node_id: int, fd: int) -> None:
        pipeline_str = (
            f"pipewiresrc fd={fd} path={node_id} ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
        )
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._appsink = self._pipeline.get_by_name("sink")
        self._pipeline.set_state(Gst.State.PLAYING)

        ret = self._pipeline.get_state(timeout=5 * Gst.SECOND)
        if ret[0] != Gst.StateChangeReturn.SUCCESS:
            raise RuntimeError("GStreamer pipeline failed to reach PLAYING state")

    def _detect_resolution(self) -> Dict:
        sample = self._appsink.emit("pull-sample")
        if sample is None:
            return {"top": 0, "left": 0, "width": 1920, "height": 1080}
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        w = struct.get_int("width").value
        h = struct.get_int("height").value
        return {"top": 0, "left": 0, "width": w, "height": h}


# ------------------------------------------------------------------
# Быстрая проверка: python Core/backends/wayland_backend.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Запрос разрешения (появится диалог Hyprland)...")
    backend = WaylandBackend(fps_limit=20)
    backend.start()

    rect = backend.get_monitor_rect()
    print(f"Монитор: {rect['width']}x{rect['height']}")

    # Показать реальный формат и framerate из пайплайна
    sample = backend._appsink.emit("pull-sample")
    if sample:
        caps = sample.get_caps()
        print(f"Caps: {caps.to_string()}")

    print("Захват 30 кадров...")
    t0 = time.perf_counter()
    for i in range(30):
        frame = backend.grab()
        assert frame.shape == (rect["height"], rect["width"], 3), \
            f"Неожиданный shape: {frame.shape}"
        assert frame.dtype == np.uint8

    elapsed = time.perf_counter() - t0
    print(f"Среднее FPS: {30 / elapsed:.1f}")
    print("OK — все проверки пройдены.")

    backend.stop()
