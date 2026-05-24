import threading
import time
from typing import List
import gi
import cairo

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
from gi.repository import Gtk, Gdk, GLib, Adw
from Overlay.base import OverlayBase, Box

class LinuxOverlay(OverlayBase):
    def __init__(self, screen_w: int, screen_h: int, monitor_index: int = 0):
        self._width = screen_w
        self._height = screen_h
        self._monitor_index = monitor_index
        self._boxes: List[Box] = []
        self._lock = threading.Lock()
        self._app = None
        self._window = None
        self._running = False
        self._thread = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_app, daemon=True)
        self._thread.start()
        # Wait for app to be ready
        timeout = 5.0
        start_time = time.time()
        while self._app is None and time.time() - start_time < timeout:
            time.sleep(0.1)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._app:
            GLib.idle_add(self._app.quit)
        if self._thread and self._thread != threading.current_thread():
            self._thread.join(timeout=2.0)

    def update_boxes(self, boxes: List[Box]) -> None:
        with self._lock:
            self._boxes = list(boxes)
        if self._window and hasattr(self, '_darea'):
            GLib.idle_add(self._darea.queue_draw)

    def is_running(self) -> bool:
        return self._running

    def _run_app(self):
        self._app = Adw.Application(application_id="com.aiagent.overlay")
        self._app.connect("activate", self._on_activate)
        self._app.run(None)

    def _on_activate(self, app):
        self._window = Gtk.ApplicationWindow(application=app)
        self._window.set_decorated(False)
        self._window.set_default_size(self._width, self._height)
        
        # Add CSS class for targeting
        self._window.add_css_class("overlay-window")
        
        # Overlay to stack UI on top of drawing area
        overlay = Gtk.Overlay()
        self._window.set_child(overlay)

        # 1. Drawing Area (Fullscreen)
        self._darea = Gtk.DrawingArea()
        self._darea.set_draw_func(self._draw_func)
        overlay.set_child(self._darea)

        # 2. Control Panel (Clickable Area)
        self._controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self._controls.add_css_class("control-panel")
        self._controls.set_halign(Gtk.Align.END)
        self._controls.set_valign(Gtk.Align.START)
        self._controls.set_margin_top(10)
        self._controls.set_margin_end(10)
        
        close_btn = Gtk.Button(label="× Close Overlay")
        close_btn.connect("clicked", lambda _: self.stop())
        self._controls.append(close_btn)
        
        overlay.add_overlay(self._controls)

        # CSS for transparency and control panel
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            window.overlay-window, 
            window.overlay-window > contents {
                background-color: transparent;
                background-image: none;
                box-shadow: none;
            }
            .control-panel {
                background-color: rgba(50, 50, 50, 0.7);
                border-radius: 8px;
                padding: 4px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
        """)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        
        self._window.connect("realize", self._on_realize)
        self._window.present()

    def _draw_func(self, area, cr, width, height, data=None):
        # Draw censorship boxes
        cr.set_source_rgba(0, 0, 0, 1.0) # Black
        with self._lock:
            boxes = list(self._boxes)

        for (x1, y1, x2, y2) in boxes:
            left = max(0, min(width, min(x1, x2)))
            top = max(0, min(height, min(y1, y2)))
            right = max(0, min(width, max(x1, x2)))
            bottom = max(0, min(height, max(y1, y2)))
            if right <= left or bottom <= top:
                continue
            cr.rectangle(left, top, right - left, bottom - top)
            cr.fill()

    def _on_realize(self, widget):
        surface = widget.get_native().get_surface()
        if surface:
            # Define a rectangle for the top-right area (control panel)
            # covers approximately 210x70 pixels in the top right corner
            rect = cairo.RectangleInt(
                x=self._width - 210, 
                y=0, 
                width=210, 
                height=70
            )
            region = cairo.Region(rect)
            surface.set_input_region(region)
