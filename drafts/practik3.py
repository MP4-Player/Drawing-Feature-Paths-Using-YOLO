import os
import cv2
import random
import time
from ultralytics import YOLO
import numpy as np

# Устраняем конфликт OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Глобальные переменные для рисования зон
drawing = False
current_zone = []
zones = []
current_zone_type = "trigger"  # или "delete"
instructions = [
    "ЛКМ: добавить точку",
    "ПКМ: завершить зону",
    "SPACE: сменить тип зоны",
    "ENTER: начать отслеживание",
    "Q: выход"
]

def draw_zones(frame, zones):
    """Отрисовка всех зон на кадре"""
    for zone in zones:
        zone_type, points = zone
        color = (0, 255, 0) if zone_type == "trigger" else (0, 0, 255)
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points)], True, color, 2)
        for point in points:
            cv2.circle(frame, point, 5, color, -1)

def is_inside_zone(point, zone):
    """Проверка, находится ли точка внутри полигона"""
    zone_type, points = zone
    if len(points) < 3:
        return False
    return cv2.pointPolygonTest(np.array(points), point, False) >= 0

def mouse_callback(event, x, y, flags, param):
    """Обработка событий мыши для рисования зон"""
    global drawing, current_zone, zones, current_zone_type
    
    if event == cv2.EVENT_LBUTTONDOWN:
        current_zone.append((x, y))
        drawing = True
    elif event == cv2.EVENT_RBUTTONDOWN and drawing:
        if len(current_zone) > 1:
            zones.append((current_zone_type, current_zone.copy()))
        current_zone = []
        drawing = False

def select_source():
    """Выбор источника видео (камера или файл)"""
    print("Выберите источник видео:")
    print("1 - Веб-камера")
    print("2 - Видеофайл")
    choice = input("Введите номер (1/2): ").strip()
    
    if choice == "1":
        # Проверяем доступные камеры
        available_cameras = []
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.read()[0]:
                available_cameras.append(i)
                cap.release()
            else:
                cap.release()
        
        if not available_cameras:
            print("Не найдено доступных камер!")
            return None
        
        print(f"Доступные камеры: {available_cameras}")
        try:
            cam_idx = int(input(f"Выберите камеру (по умолчанию {available_cameras[0]}): ") or available_cameras[0])
            if cam_idx not in available_cameras:
                print(f"Камера {cam_idx} недоступна, будет использована {available_cameras[0]}")
                cam_idx = available_cameras[0]
            return cam_idx
        except ValueError:
            print("Некорректный ввод, будет использована камера 0")
            return available_cameras[0]
    
    elif choice == "2":
        video_files = [f for f in os.listdir() if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            print("В текущей папке нет видеофайлов!")
            return None
        
        print("Доступные видеофайлы:")
        for i, f in enumerate(video_files, 1):
            print(f"{i} - {f}")
        
        try:
            vid_idx = int(input("Выберите видеофайл (номер): ")) - 1
            if 0 <= vid_idx < len(video_files):
                return video_files[vid_idx]
            else:
                print("Некорректный номер, будет использован первый файл")
                return video_files[0]
        except ValueError:
            print("Некорректный ввод, будет использован первый файл")
            return video_files[0]
    
    print("Некорректный выбор, будет использована камера 0")
    return 0

def define_zones(frame):
    """Определение зон прямо на видео"""
    global zones, current_zone, current_zone_type
    
    window_name = 'Определение зон (рисуйте прямо на видео)'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display_frame = frame.copy()
        
        # Отрисовка текущей зоны
        if current_zone:
            if len(current_zone) > 1:
                cv2.polylines(display_frame, [np.array(current_zone)], False, 
                             (0, 255, 0) if current_zone_type == "trigger" else (0, 0, 255), 2)
            for point in current_zone:
                cv2.circle(display_frame, point, 5, 
                          (0, 255, 0) if current_zone_type == "trigger" else (0, 0, 255), -1)
        
        # Отрисовка всех завершенных зон
        draw_zones(display_frame, zones)
        
        # Отображение инструкций
        cv2.putText(display_frame, f"Текущая зона: {current_zone_type}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y = 60
        for line in instructions:
            cv2.putText(display_frame, line, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Смена типа зоны
            current_zone_type = "delete" if current_zone_type == "trigger" else "trigger"
        elif key == 13:  # Enter - завершить определение зон
            cv2.destroyWindow(window_name)
            return True
        elif key == ord('q'):  # Выход
            cv2.destroyWindow(window_name)
            return False

def main():
    # Установка корректной кодировки для консоли (Windows)
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
    
    # Инициализация модели
    model = YOLO('yolov8n.pt')
    trajectories = {}
    object_colors = {}
    active_objects = {}  # Для отслеживания активных объектов
    
    # Выбор источника видео
    video_source = select_source()
    if video_source is None:
        print("Не удалось выбрать источник видео!")
        return
    
    # Открываем видео поток
    if isinstance(video_source, int):
        print(f"Используется камера {video_source}")
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    else:
        print(f"Используется видеофайл: {video_source}")
        cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Ошибка открытия видео потока!")
        return
    
    # Получаем первый кадр для определения зон
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить первый кадр!")
        cap.release()
        return
    
    # Определение зон
    if not define_zones(frame):
        cap.release()
        return
    
    # Разделяем зоны на триггерные и удаляющие
    trigger_zones = [("trigger", zone) for zone_type, zone in zones if zone_type == "trigger"]
    delete_zones = [("delete", zone) for zone_type, zone in zones if zone_type == "delete"]
    
    print(f"Определено триггерных зон: {len(trigger_zones)}, удаляющих зон: {len(delete_zones)}")
    
    # Основной цикл обработки видео
    start_time = time.time()
    tracking_window = 'YOLOv8 Tracking (Q - выход)'
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Получаем текущее время
        current_time = time.time() - start_time
        hours, remainder = divmod(current_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # Трекинг объектов
        results = model.track(frame, persist=True, tracker="botsort.yaml")
        
        # Визуализация результатов
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []
            clss = result.boxes.cls.cpu().numpy()
            
            for box, obj_id, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Генерация цвета для объекта
                if obj_id not in object_colors:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    object_colors[obj_id] = color
                else:
                    color = object_colors[obj_id]
                
                # Отрисовка bounding box и ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'ID: {int(obj_id)}', (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Проверка триггерных зон
                for zone in trigger_zones:
                    if is_inside_zone((center_x, center_y), zone):
                        if obj_id not in trajectories:
                            trajectories[obj_id] = []
                            print(f"Объект с ID {int(obj_id)} начал рисовать траекторию в {time_str}")
                            active_objects[obj_id] = time_str
                        trajectories[obj_id].append((center_x, center_y))
                
                # Проверка удаляющих зон
                for zone in delete_zones:
                    if is_inside_zone((center_x, center_y), zone):
                        if obj_id in trajectories:
                            del trajectories[obj_id]
                            if obj_id in active_objects:
                                print(f"Объект с ID {int(obj_id)} удален (был активен с {active_objects[obj_id]}) в {time_str}")
                                del active_objects[obj_id]
                
                # Отрисовка траекторий
                if obj_id in trajectories:
                    for i in range(1, len(trajectories[obj_id])):
                        cv2.line(frame, trajectories[obj_id][i - 1], trajectories[obj_id][i], color, 2)
        
        # Отрисовка зон
        draw_zones(frame, zones)
        
        # Отображение информации
        cv2.putText(frame, f"Активных объектов: {len(active_objects)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, time_str, (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(tracking_window, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()