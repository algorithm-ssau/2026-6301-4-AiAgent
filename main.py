from gui import CensorApp
import tkinter as tk


def main():
    root = tk.Tk()
    app = CensorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()