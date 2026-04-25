import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Core.capture import ScreenCapturer
import time

def show_preview_in_corner(capturer):
    """
    Бесконечно показывает превью захвата экрана в углу экрана.
    Нажмите 'q' для выхода.
    """
    # Создаем захватчик экрана
    
    # Настройки окна
    PREVIEW_WIDTH = 320  # Ширина превью
    PREVIEW_HEIGHT = 180  # Высота превью (16:9)
    CORNER_MARGIN = 20    # Отступ от края экрана
    
    try:
        # Запускаем захват
        capturer.start()
        print("Захват экрана запущен. Нажмите 'q' в окне превью для выхода.")
        print(f"Используемый бэкенд: {capturer.backend_type}")
        
        # Получаем информацию о мониторе
        monitor = capturer.get_monitor_rect()
        print(f"Монитор: {monitor['width']}x{monitor['height']}")
        
        # Создаем окно
        window_name = "Screen Capture Preview (Press 'q' to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Устанавливаем размер окна
        cv2.resizeWindow(window_name, PREVIEW_WIDTH, PREVIEW_HEIGHT)
        
        # Перемещаем окно в правый верхний угол
        # Для Windows можно использовать cv2.moveWindow()
        # Получаем размеры экрана через capturer
        screen_width = monitor['width']
        screen_height = monitor['height']
        
        # Позиция окна (правый верхний угол с отступом)
        window_x = screen_width - PREVIEW_WIDTH - CORNER_MARGIN
        window_y = CORNER_MARGIN
        
        # Перемещаем окно
        cv2.moveWindow(window_name, window_x, window_y)
        
        # Делаем окно поверх всех (работает на Windows)
        try:
            # Windows specific: make window stay on top
            import win32gui
            import win32con
            
            hwnd = win32gui.FindWindow(None, window_name)
            if hwnd:
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        except ImportError:
            print("Для закрепления окна поверх всех установите pywin32: pip install pywin32")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Захватываем кадр
            frame = capturer.grab()
            
            # Масштабируем кадр для превью
            preview = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            
            # Добавляем информационный текст
            # Текущий FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                cv2.putText(preview, f"FPS: {fps:.1f}", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Время
            current_time = time.strftime("%H:%M:%S")
            cv2.putText(preview, current_time, (10, PREVIEW_HEIGHT - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Размер кадра
            cv2.putText(preview, f"{frame.shape[1]}x{frame.shape[0]}", 
                       (PREVIEW_WIDTH - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Показываем превью
            cv2.imshow(window_name, preview)
            
            # Проверяем нажатие клавиши 'q' для выхода
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Выход по запросу пользователя")
                break
                
            # Небольшая задержка для снижения нагрузки на CPU
            cv2.waitKey(1)
            
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем (Ctrl+C)")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Очистка
        capturer.stop()
        cv2.destroyAllWindows()
        print("Программа завершена")

def show_preview_in_corner_simple(capturer):
    """
    Упрощенная версия без дополнительных зависимостей.
    """
    
    PREVIEW_WIDTH = 320
    PREVIEW_HEIGHT = 180
    
    try:
        capturer.start()
        monitor = capturer.get_monitor_rect()
        
        print("Захват экрана запущен. Нажмите 'q' в окне превью для выхода.")
        print(f"Монитор: {monitor['width']}x{monitor['height']}")
        print("Используемый бэкенд:", capturer.backend_type)
        
        window_name = "Screen Capture (Press 'q' to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, PREVIEW_WIDTH, PREVIEW_HEIGHT)
        
        # Перемещаем окно в правый верхний угол
        window_x = monitor['width'] - PREVIEW_WIDTH - 20
        window_y = 20
        cv2.moveWindow(window_name, window_x, window_y)
        
        while True:
            frame = capturer.grab()
            preview = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            
            # Добавляем простую информацию
            cv2.putText(preview, "Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.imshow(window_name, preview)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nОстановка...")
    finally:
        capturer.stop()
        cv2.destroyAllWindows()
        print("Завершено")

if __name__ == "__main__":
    a = int(input("Выберите монитор: "))
    capturer = ScreenCapturer(monitor_index=a, fps_limit=30)
    print("Выберите режим:")
    print("1. Расширенный (с FPS и временем)")
    print("2. Простой (только превью)")
    
    choice = input("Ваш выбор (1 или 2): ").strip()
    
    if choice == "1":
        show_preview_in_corner(capturer)
    else:
        show_preview_in_corner_simple()