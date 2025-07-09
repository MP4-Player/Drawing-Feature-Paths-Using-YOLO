import os
import cv2
import random
import time
from ultralytics import YOLO
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class TrajectoryTracker:
    def __init__(self):
        """Инициализация всех параметров трекера"""
        # Флаги состояния
        self.drawing = False              # Рисуется ли сейчас зона
        self.define_zones_mode = True     # Режим определения зон
        self.paused = True                # Пауза
        
        # Данные для зон
        self.current_zone = []            # Текущая рисуемая зона
        self.zones = []                   # Все созданные зоны
        self.current_zone_type = "trigger" # Тип текущей зоны
        self.zone_colors = {
            "trigger": (0, 255, 0),       # Зеленый для триггерных зон
            "delete": (0, 0, 255)         # Красный для зон удаления
        }
        
        # Данные для трекинга
        self.trajectories = {}            # Траектории объектов
        self.object_colors = {}           # Цвета объектов
        self.active_objects = {}          # Активные объекты
        self.object_in_trigger = {}       # Объекты в триггерных зонах
        self.last_seen = {}               # Время последнего обнаружения
        
        # Видео и модель
        self.cap = None                   # Видеопоток
        self.model = None                 # Модель YOLO
        self.current_frame = None         # Текущий кадр
        self.clean_frame = None           # Чистый кадр для YOLO
        self.video_source = None          # Источник видео
        
        # Производительность
        self.fps_counter = 0              # Счетчик кадров
        self.last_fps_time = time.time()  # Время последнего подсчета FPS
        self.fps = 0                      # Текущий FPS
        
        # Интерфейс
        self.window_name = 'YOLOv8 Tracking and Zone Drawing' # Имя окна
        
        # Классы объектов COCO
        self.class_names = {
            0: "человек",
            1: "велосипед",
            2: "автомобиль",
            3: "мотоцикл", 
            5: "автобус",
            7: "грузовик"
        }
        self.classes_to_detect = None     # Классы для детекции

    def select_source(self):
        """Выбор источника видео (камера или файл)"""
        print("\nВыберите источник видео:")
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
            except:
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
                return video_files[0]
            except:
                return video_files[0]
        
        return 0

    def select_yolo_model(self):
        """Выбор модели YOLO"""
        print("\nВыберите модель YOLO (точность vs скорость):")
        print("1 - yolov8n.pt (нано, самая быстрая)")
        print("2 - yolov8s.pt (малая)")
        print("3 - yolov8m.pt (средняя)")
        print("4 - yolov8l.pt (большая)")
        print("5 - yolov8x.pt (огромная, самая точная)")
        choice = input("Введите номер модели (1-5, по умолчанию 1): ").strip()
        
        models = {
            '1': 'yolov8n.pt',
            '2': 'yolov8s.pt',
            '3': 'yolov8m.pt',
            '4': 'yolov8l.pt',
            '5': 'yolov8x.pt'
        }
        
        model_path = models.get(choice, 'yolov8n.pt')
        print(f"Загружается {model_path}...")
        try:
            model = YOLO(model_path)
            if not hasattr(model, 'track'):
                print("Предупреждение: выбранная модель не поддерживает трекинг!")
            return model
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return None

    def select_detection_mode(self):
        """Выбор режима детекции объектов"""
        print("\nВыберите режим детекции:")
        print("1 - Все объекты (по умолчанию)")
        print("2 - Только люди")
        print("3 - Только транспорт (автомобили, автобусы, грузовики)")
        print("4 - Двухколесные (велосипеды, мотоциклы)")
        print("5 - Ручной выбор классов")
        choice = input("Введите номер режима (1-5, по умолчанию 1): ").strip()
        
        modes = {
            '1': None,  # Все классы
            '2': [0],   # Только люди
            '3': [2, 5, 7],  # Транспорт
            '4': [1, 3]  # Двухколесные
        }
        
        if choice == '5':
            print("\nДоступные классы:")
            for class_id, name in self.class_names.items():
                print(f"{class_id} - {name}")
            custom_classes = input("Введите ID классов через запятую: ").strip()
            try:
                return [int(c) for c in custom_classes.split(',')]
            except:
                print("Ошибка ввода! Используются все классы.")
                return None
        
        return modes.get(choice, None)

    def init_video_capture(self, video_source):
        """Инициализация видеопотока"""
        if isinstance(video_source, int):
            self.cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            print(f"Ошибка открытия видео источника: {video_source}")
            return False
        
        ret, frame = self.cap.read()
        if not ret:
            print("Ошибка чтения первого кадра!")
            self.cap.release()
            return False
        
        self.current_frame = frame.copy()
        self.clean_frame = frame.copy()
        return True

    def mouse_callback(self, event, x, y, flags, param):
        """Обработчик событий мыши для рисования зон"""
        if not (self.define_zones_mode or self.paused):
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_zone.append((x, y))
            self.drawing = True
            
        elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
            if len(self.current_zone) > 2:
                self.zones.append((self.current_zone_type, self.current_zone.copy()))
            self.current_zone = []
            self.drawing = False

    def draw_zones(self, frame):
        """Отрисовка всех зон на кадре"""
        for zone in self.zones:
            zone_type, points = zone
            color = self.zone_colors[zone_type]
            if len(points) > 1:
                cv2.polylines(frame, [np.array(points)], True, color, 2)
            for point in points:
                cv2.circle(frame, point, 5, color, -1)

    def is_inside_zone(self, point, zone):
        """Проверка, находится ли точка внутри зоны"""
        zone_type, points = zone
        if len(points) < 3:
            return False
        return cv2.pointPolygonTest(np.array(points), point, False) >= 0

    def delete_zones(self, zone_type=None):
        """Удаление зон"""
        if zone_type is None:
            self.zones = []
        else:
            self.zones = [zone for zone in self.zones if zone[0] != zone_type]

    def cleanup_old_objects(self):
        """Удаление старых объектов"""
        current_time = time.time()
        to_delete = []
        
        for obj_id in list(self.trajectories.keys()):
            if current_time - self.last_seen.get(obj_id, 0) > 2.0:
                to_delete.append(obj_id)
        
        for obj_id in to_delete:
            if obj_id in self.trajectories:
                del self.trajectories[obj_id]
            if obj_id in self.active_objects:
                del self.active_objects[obj_id]
            if obj_id in self.object_in_trigger:
                del self.object_in_trigger[obj_id]
            if obj_id in self.last_seen:
                del self.last_seen[obj_id]

    def process_frame(self):
        """Обработка текущего кадра"""
        if not self.paused and not self.define_zones_mode and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.video_source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return
                else:
                    self.paused = True
                    return
            
            self.clean_frame = frame.copy()
            self.current_frame = frame.copy()

    def process_detections(self):
        """Обработка обнаружений YOLO"""
        if not self.paused and not self.define_zones_mode and self.model is not None:
            # Запуск трекинга на чистом кадре
            results = self.model.track(
                self.clean_frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.5,
                classes=self.classes_to_detect,
                verbose=False
            )
    
            # Проверка наличия результатов
            if not results or not results[0].boxes:
                return
    
            # Получаем детекции
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
    
            current_ids = set()
    
            for box, obj_id, cls, conf in zip(boxes, ids, clss, confs):
                current_ids.add(obj_id)
                x1, y1, x2, y2 = map(int, box)
                center = (x1 + x2) // 2, (y1 + y2) // 2
    
                # Инициализация нового объекта
                if obj_id not in self.object_colors:
                    self.object_colors[obj_id] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                    self.object_in_trigger[obj_id] = False
    
                # Обновляем время последнего обнаружения
                self.last_seen[obj_id] = time.time()
    
                # Отрисовка bounding box и ID
                color = self.object_colors[obj_id]
                cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(self.current_frame, 
                           f'ID:{int(obj_id)} {self.class_names.get(int(cls), "unknown")}',
                           (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
                # Проверка зон и обновление траекторий
                self.check_zones_and_update_trajectories(obj_id, center)

    def update_fps(self):
        """Обновление счетчика FPS"""
        self.fps_counter += 1
        if time.time() - self.last_fps_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = time.time()

    def show_instructions(self, frame):
        """Отображение инструкций на кадре"""
        y = 30
        mode = "ПАУЗА" if self.paused else "ОПРЕДЕЛЕНИЕ ЗОН" if self.define_zones_mode else "ТРЕКИНГ"
        cv2.putText(frame, f"РЕЖИМ: {mode}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 40
        
        if self.define_zones_mode or self.paused:
            cv2.putText(frame, f"Тип зоны: {self.current_zone_type}", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "ЛКМ: добавить точку", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "ПКМ: завершить зону (мин. 3 точки)", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "C: сменить тип зоны", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "ПРОБЕЛ: продолжить/пауза", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "ENTER: начать трекинг", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "D: удалить все зоны", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "T: удалить триггерные зоны", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "E: удалить зоны удаления", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "Q: выход", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            if self.classes_to_detect:
                classes_str = ", ".join([self.class_names.get(c, str(c)) for c in self.classes_to_detect])
                cv2.putText(frame, f"Детекция: {classes_str}", (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                

    def check_zones_and_update_trajectories(self, obj_id, center):
        """Проверка зон и обновление траекторий"""
    # Проверка триггерных зон
        in_trigger = any(
            self.is_inside_zone(center, ("trigger", zone)) 
            for zone_type, zone in self.zones 
            if zone_type == "trigger"
        )
    
        if in_trigger and not self.object_in_trigger.get(obj_id, False):
            self.object_in_trigger[obj_id] = True
            self.trajectories[obj_id] = []
            self.active_objects[obj_id] = time.time() - self.start_time
    
        # Проверка зон удаления
        in_delete = any(
            self.is_inside_zone(center, ("delete", zone)) 
            for zone_type, zone in self.zones 
            if zone_type == "delete"
        )
    
        if in_delete and obj_id in self.trajectories:
            if obj_id in self.active_objects:
                del self.active_objects[obj_id]
            self.object_in_trigger[obj_id] = False
    
        # Добавление точки в траекторию
        if self.object_in_trigger.get(obj_id, False):
            if obj_id not in self.trajectories:
                self.trajectories[obj_id] = []
            self.trajectories[obj_id].append(center)


    def update_display(self):
        """Обновление отображаемого кадра"""
        if self.current_frame is None or self.clean_frame is None:
            return
            
        display_frame = self.current_frame.copy()
        current_time = time.time() - self.start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(current_time))
        
        self.cleanup_old_objects()
        self.process_detections()
        
      # Отрисовка траекторий
        for obj_id, trajectory in self.trajectories.items():
          if len(trajectory) > 1:
              color = self.object_colors.get(obj_id, (0, 0, 255))
              for i in range(1, len(trajectory)):
                  cv2.line(display_frame, 
                          trajectory[i-1], 
                          trajectory[i], 
                          color, 2)
        
        # Отрисовка текущей зоны
        if (self.define_zones_mode or self.paused) and self.current_zone:
            if len(self.current_zone) > 1:
                cv2.polylines(display_frame, [np.array(self.current_zone)], False, 
                             self.zone_colors[self.current_zone_type], 2)
            for point in self.current_zone:
                cv2.circle(display_frame, point, 5, 
                          self.zone_colors[self.current_zone_type], -1)
        
        self.draw_zones(display_frame)
        self.show_instructions(display_frame)
        self.update_fps()
        
        # Отображение информации
        cv2.putText(display_frame, f"FPS: {self.fps}", 
                   (display_frame.shape[1] - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Объекты: {len(self.active_objects)}", 
                   (display_frame.shape[1] - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, display_frame)

    def run(self):
        """Основной метод запуска трекера"""
        self.video_source = self.select_source()
        if self.video_source is None:
            return
        
        self.model = self.select_yolo_model()
        if self.model is None:
            return
            
        self.classes_to_detect = self.select_detection_mode()
        
        if not self.init_video_capture(self.video_source):
            return
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        if os.name == 'nt':
            import ctypes
            try:
                hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)
                ctypes.windll.user32.SetClassLongW(hwnd, -26, 0)
            except:
                pass
        
        self.start_time = time.time()
        
        while True:
            self.process_frame()
            self.update_display()
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                if not self.define_zones_mode:
                    self.paused = not self.paused
            elif key == ord('c'):
                if self.define_zones_mode or self.paused:
                    self.current_zone_type = "delete" if self.current_zone_type == "trigger" else "trigger"
            elif key == 13:
                if self.define_zones_mode:
                    if len([z for z in self.zones if z[0] == "trigger"]) > 0:
                        self.define_zones_mode = False
                        self.paused = False
                        print("Трекинг начат!")
                    else:
                        print("Ошибка: не определена ни одна триггерная зона!")
            elif key == ord('d'):
                self.delete_zones()
                print("Все зоны удалены")
            elif key == ord('t'):
                self.delete_zones("trigger")
                print("Триггерные зоны удалены")
            elif key == ord('e'):
                self.delete_zones("delete")
                print("Зоны удаления удалены")
            elif key == ord('q'):
                break
        
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = TrajectoryTracker()
    tracker.run()