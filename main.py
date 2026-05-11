import ctypes
import multiprocessing
import sys
from ctypes import wintypes


_APP_MUTEX_HANDLE = None


def _acquire_single_instance_lock() -> bool:
    if sys.platform != "win32":
        return True

    global _APP_MUTEX_HANDLE

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    )
    kernel32.CreateMutexW.restype = wintypes.HANDLE

    _APP_MUTEX_HANDLE = kernel32.CreateMutexW(
        None,
        False,
        "Local\\AlcoholCensorSingleInstance",
    )

    if not _APP_MUTEX_HANDLE:
        return True

    return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS


def main():
    import tkinter as tk
    from gui import CensorApp

    root = tk.Tk()
    app = CensorApp(root)
    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if _acquire_single_instance_lock():
        main()
