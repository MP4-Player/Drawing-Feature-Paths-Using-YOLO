import os
import cv2
import random
import time
from ultralytics import YOLO
import numpy as np

# Устраняем конфликт OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class TrajectoryTracker:
    def __init__(self):
        self.drawing = False
        self.current_zone = []
        self.zones = []
        self.current_zone_type = "trigger"  # или "delete"
        self.define_zones_mode = True
        self.window_name = 'YOLOv8 Tracking and Zone Drawing'
        self.model = YOLO('yolov8n.pt')
        self.trajectories = {}
        self.object_colors = {}
        self.active_objects = {}
    
    def draw_zones(self, frame):
        """Отрисовка всех зон на кадре"""
        for zone in self.zones:
            zone_type, points = zone
            color = (0, 255, 0) if zone_type == "trigger" else (0, 0, 255)
            if len(points) > 1:
                cv2.polylines(frame, [np.array(points)], True, color, 2)
            for point in points:
                cv2.circle(frame, point, 5, color, -1)
    
    def is_inside_zone(self, point, zone):
        """Проверка, находится ли точка внутри полигона"""
        zone_type, points = zone
        if len(points) < 3:
            return False
        return cv2.pointPolygonTest(np.array(points), point, False) >= 0
    
    def mouse_callback(self, event, x, y, flags, param):
        """Обработка событий мыши для рисования зон"""
        if not self.define_zones_mode:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_zone.append((x, y))
            self.drawing = True
        elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
            if len(self.current_zone) > 1:
                self.zones.append((self.current_zone_type, self.current_zone.copy()))
            self.current_zone = []
            self.drawing = False
    
    def select_source(self):
        """Выбор источника видео"""
        print("Выберите источник видео:")
        print("1 - Веб-камера")
        print("2 - Видеофайл")
        choice = input("Введите номер (1/2): ").strip()
        
        if choice == "1":
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
                return cam_idx
            except ValueError:
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
                    return video_files[0]
            except ValueError:
                return video_files[0]
        
        return 0
    
    def run(self):
        """Основной метод запуска трекера"""
        # Установка корректной кодировки для Windows
        if os.name == 'nt':
            os.system('chcp 65001 > nul')
        
        # Выбор источника
        video_source = self.select_source()
        if video_source is None:
            return
        
        cap = cv2.VideoCapture(video_source if isinstance(video_source, int) else video_source, 
                              cv2.CAP_DSHOW if isinstance(video_source, int) else 0)
        
        if not cap.isOpened():
            print("Ошибка открытия видео!")
            return
        
        # Создаем одно окно
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Убираем иконку Jupyter (только для Windows)
        if os.name == 'nt':
            import ctypes
            try:
                hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)
                ctypes.windll.user32.SetClassLongW(hwnd, -26, 0)
            except:
                pass
        
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Получаем текущее время
            current_time = time.time() - start_time
            time_str = time.strftime("%H:%M:%S", time.gmtime(current_time))
            
            display_frame = frame.copy()
            
            if self.define_zones_mode:
                # Режим определения зон
                if self.current_zone:
                    if len(self.current_zone) > 1:
                        cv2.polylines(display_frame, [np.array(self.current_zone)], False, 
                                     (0, 255, 0) if self.current_zone_type == "trigger" else (0, 0, 255), 2)
                    for point in self.current_zone:
                        cv2.circle(display_frame, point, 5, 
                                  (0, 255, 0) if self.current_zone_type == "trigger" else (0, 0, 255), -1)
                
                self.draw_zones(display_frame)
                
                # Инструкции
                cv2.putText(display_frame, "РЕЖИМ ОПРЕДЕЛЕНИЯ ЗОН", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Тип зоны: {self.current_zone_type}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "ЛКМ: добавить точку", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "ПКМ: завершить зону", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "Пробел: сменить тип", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "Enter: начать отслеживание", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Режим отслеживания
                results = self.model.track(frame, persist=True, tracker="botsort.yaml")
                
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []
                    
                    for box, obj_id in zip(boxes, ids):
                        x1, y1, x2, y2 = map(int, box)
                        center = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        if obj_id not in self.object_colors:
                            self.object_colors[obj_id] = (random.randint(0, 255), 
                                                       random.randint(0, 255), 
                                                       random.randint(0, 255))
                        
                        color = self.object_colors[obj_id]
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, f'ID: {int(obj_id)}', (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Проверка зон
                        for zone_type, zone in self.zones:
                            if self.is_inside_zone(center, (zone_type, zone)):
                                if zone_type == "trigger":
                                    if obj_id not in self.trajectories:
                                        self.trajectories[obj_id] = []
                                        print(f"Объект {int(obj_id)} начал траекторию в {time_str}")
                                        self.active_objects[obj_id] = time_str
                                    self.trajectories[obj_id].append(center)
                                else:
                                    if obj_id in self.trajectories:
                                        del self.trajectories[obj_id]
                                        if obj_id in self.active_objects:
                                            print(f"Объект {int(obj_id)} удален в {time_str}")
                                            del self.active_objects[obj_id]
                        
                        # Отрисовка траекторий
                        if obj_id in self.trajectories and len(self.trajectories[obj_id]) > 1:
                            for i in range(1, len(self.trajectories[obj_id])):
                                cv2.line(display_frame, self.trajectories[obj_id][i-1], 
                                        self.trajectories[obj_id][i], color, 2)
                
                self.draw_zones(display_frame)
                cv2.putText(display_frame, f"Объектов: {len(self.active_objects)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, time_str, (display_frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if self.define_zones_mode:
                if key == ord(' '):
                    self.current_zone_type = "delete" if self.current_zone_type == "trigger" else "trigger"
                elif key == 13:  # Enter
                    self.define_zones_mode = False
                    print("Начато отслеживание!")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = TrajectoryTracker()
    tracker.run()