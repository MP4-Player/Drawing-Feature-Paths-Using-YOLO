import os
import cv2
import random
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QSlider, QLabel, QFrame, QComboBox, QGroupBox, 
                            QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QPainter, QColor, QPen
from ultralytics import YOLO

# Устранение конфликта OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'  # Уменьшаем вывод ошибок OpenCV

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Trajectory Tracker")
        self.setMinimumSize(1000, 700)
        
        # Инициализация переменных трекера
        self.drawing = False
        self.current_zone = []
        self.zones = []
        self.current_zone_type = "trigger"
        self.define_zones_mode = True
        self.trajectories = {}
        self.object_colors = {}
        self.active_objects = {}
        self.object_in_trigger = {}
        self.paused = True
        self.current_frame = None
        self.cap = None
        self.model = None
        self.track_mode = "all"
        self.video_source = None
        self.start_time = 0
        self.last_seen = {}
        self.frame_count = 0
        self.fps = 30
        self.video_width = 640
        self.video_height = 480
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.original_frame_size = QSize(640, 480)
        
        # Цвета зон
        self.zone_colors = {
            "trigger": QColor(0, 255, 0),
            "delete": QColor(255, 0, 0)
        }
        
        # Фильтры классов
        self.class_filters = {
            "vehicles": [2, 3, 5, 7],  # car, motorcycle, bus, truck
            "people": [0],  # person
            "all": None
        }
        
        # Инициализация UI
        self.init_ui()
        self.apply_styles()
        self.load_available_cameras()
        
    def init_ui(self):
        # Главный виджет и layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        
        # Панель видео
        self.video_frame = QFrame()
        self.video_frame.setFrameShape(QFrame.StyledPanel)
        self.video_layout = QVBoxLayout()
        self.video_frame.setLayout(self.video_layout)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_layout.addWidget(self.video_label)
        
        # Таймлайн
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.sliderMoved.connect(self.set_position)
        self.video_layout.addWidget(self.timeline_slider)
        
        # Панель управления
        self.control_panel = QFrame()
        self.control_panel.setFrameShape(QFrame.StyledPanel)
        self.control_panel.setFixedWidth(300)
        
        self.control_layout = QVBoxLayout()
        self.control_panel.setLayout(self.control_layout)
        
        # Группа источника видео
        self.source_group = QGroupBox("Video Source")
        self.source_layout = QVBoxLayout()
        self.source_group.setLayout(self.source_layout)
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam", "Video File"])
        self.source_combo.currentIndexChanged.connect(self.update_source_ui)
        self.source_layout.addWidget(self.source_combo)
        
        self.webcam_combo = QComboBox()
        self.webcam_combo.setVisible(False)
        self.source_layout.addWidget(self.webcam_combo)
        
        self.browse_video_btn = QPushButton("Select Video File")
        self.browse_video_btn.setVisible(False)
        self.browse_video_btn.clicked.connect(self.browse_video)
        self.source_layout.addWidget(self.browse_video_btn)
        
        self.load_source_btn = QPushButton("Load Source")
        self.load_source_btn.clicked.connect(self.load_source)
        self.source_layout.addWidget(self.load_source_btn)
        
        # Группа модели YOLO
        self.model_group = QGroupBox("YOLO Model")
        self.model_layout = QVBoxLayout()
        self.model_group.setLayout(self.model_layout)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt (nano)", "yolov8s.pt (small)", 
                                 "yolov8m.pt (medium)", "yolov8l.pt (large)", 
                                 "yolov8x.pt (xlarge)"])
        self.model_layout.addWidget(self.model_combo)
        
        # Группа режима трекинга
        self.track_group = QGroupBox("Tracking Mode")
        self.track_layout = QVBoxLayout()
        self.track_group.setLayout(self.track_layout)
        
        self.track_combo = QComboBox()
        self.track_combo.addItems(["All objects", "Vehicles only", "People only"])
        self.track_layout.addWidget(self.track_combo)
        
        # Группа управления зонами
        self.zone_group = QGroupBox("Zone Management")
        self.zone_layout = QVBoxLayout()
        self.zone_group.setLayout(self.zone_layout)
        
        self.zone_type_combo = QComboBox()
        self.zone_type_combo.addItems(["Trigger Zone", "Delete Zone"])
        self.zone_type_combo.currentIndexChanged.connect(self.change_zone_type)
        self.zone_layout.addWidget(self.zone_type_combo)
        
        self.clear_zones_btn = QPushButton("Clear All Zones")
        self.clear_zones_btn.clicked.connect(lambda: self.delete_zones())
        self.zone_layout.addWidget(self.clear_zones_btn)
        
        self.clear_trigger_btn = QPushButton("Clear Trigger Zones")
        self.clear_trigger_btn.clicked.connect(lambda: self.delete_zones("trigger"))
        self.zone_layout.addWidget(self.clear_trigger_btn)
        
        self.clear_delete_btn = QPushButton("Clear Delete Zones")
        self.clear_delete_btn.clicked.connect(lambda: self.delete_zones("delete"))
        self.zone_layout.addWidget(self.clear_delete_btn)
        
        # Группа управления воспроизведением
        self.playback_group = QGroupBox("Playback")
        self.playback_layout = QHBoxLayout()
        self.playback_group.setLayout(self.playback_layout)
        
        self.play_button = QPushButton()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_button.clicked.connect(self.toggle_play)
        
        self.pause_button = QPushButton()
        self.pause_button.setIcon(QIcon.fromTheme("media-playback-pause"))
        self.pause_button.clicked.connect(self.pause_video)
        
        self.stop_button = QPushButton()
        self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_button.clicked.connect(self.stop_video)
        
        self.playback_layout.addWidget(self.play_button)
        self.playback_layout.addWidget(self.pause_button)
        self.playback_layout.addWidget(self.stop_button)
        
        # Кнопка запуска трекинга
        self.start_tracking_btn = QPushButton("Start Tracking")
        self.start_tracking_btn.clicked.connect(self.start_tracking)
        
        # Добавление групп на панель управления
        self.control_layout.addWidget(self.source_group)
        self.control_layout.addWidget(self.model_group)
        self.control_layout.addWidget(self.track_group)
        self.control_layout.addWidget(self.zone_group)
        self.control_layout.addWidget(self.playback_group)
        self.control_layout.addWidget(self.start_tracking_btn)
        self.control_layout.addStretch()
        
        # Добавление видео и панели управления в главный layout
        self.main_layout.addWidget(self.video_frame, stretch=1)
        self.main_layout.addWidget(self.control_panel)
        
        # Таймер для обновления видео
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
            QFrame {
                background-color: #3c3f41;
                border-radius: 5px;
            }
            QGroupBox {
                color: #bbbbbb;
                font-size: 14px;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #4e5254;
                color: #ffffff;
                border: none;
                padding: 8px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #5e6264;
            }
            QPushButton:pressed {
                background-color: #3e4244;
            }
            QPushButton:checked {
                background-color: #4a6ea9;
            }
            QComboBox {
                background-color: #4e5254;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 4px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #4e5254;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                margin: -5px 0;
                background: #ffffff;
                border-radius: 9px;
            }
        """)
        
        # Установка иконок для кнопок
        self.play_button.setIconSize(QSize(24, 24))
        self.pause_button.setIconSize(QSize(24, 24))
        self.stop_button.setIconSize(QSize(24, 24))
        
    def load_available_cameras(self):
        self.webcam_combo.clear()
        available_cameras = []
        
        # Проверяем доступные камеры с обработкой ошибок
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
            except:
                pass
        
        if available_cameras:
            for cam_idx in available_cameras:
                self.webcam_combo.addItem(f"Camera {cam_idx}", cam_idx)
        else:
            self.webcam_combo.addItem("No cameras found", -1)
    
    def update_source_ui(self, index):
        if index == 0:  # Webcam
            self.webcam_combo.setVisible(True)
            self.browse_video_btn.setVisible(False)
        else:  # Video File
            self.webcam_combo.setVisible(False)
            self.browse_video_btn.setVisible(True)
    
    def browse_video(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file:
            self.video_source = file
    
    def load_source(self):
        if self.source_combo.currentIndex() == 0:  # Webcam
            cam_idx = self.webcam_combo.currentData()
            if cam_idx == -1:
                QMessageBox.warning(self, "Warning", "No cameras available!")
                return
            self.video_source = cam_idx
        else:  # Video File
            if not self.video_source:
                QMessageBox.warning(self, "Warning", "Please select a video file first!")
                return
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        try:
            # Для камеры
            if isinstance(self.video_source, int):
                self.cap = cv2.VideoCapture(self.video_source)
                if not self.cap.isOpened():
                    raise Exception(f"Could not open camera {self.video_source}")
                
                # Устанавливаем размеры кадра для улучшения производительности
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Получаем первый кадр для определения размера
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("Could not read frame from camera")
                
                self.original_frame_size = QSize(frame.shape[1], frame.shape[0])
                self.current_frame = frame.copy()
                self.paused = True
                self.timer.stop()
                self.update_frame_display()
                
            # Для видеофайла
            else:
                self.cap = cv2.VideoCapture(self.video_source)
                if not self.cap.isOpened():
                    raise Exception(f"Could not open video file {self.video_source}")
                
                # Получаем информацию о видео
                self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0:
                    self.fps = 30
                
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("Could not read first frame from video")
                
                self.original_frame_size = QSize(frame.shape[1], frame.shape[0])
                self.current_frame = frame.copy()
                self.paused = True
                self.timer.stop()
                self.timeline_slider.setRange(0, self.frame_count)
                self.update_frame_display()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load source: {str(e)}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return
    
    def change_zone_type(self, index):
        self.current_zone_type = "trigger" if index == 0 else "delete"
    
    def load_model(self):
        model_map = {
            0: "yolov8n.pt",
            1: "yolov8s.pt",
            2: "yolov8m.pt",
            3: "yolov8l.pt",
            4: "yolov8x.pt"
        }
        
        model_idx = self.model_combo.currentIndex()
        model_name = model_map.get(model_idx, "yolov8n.pt")
        
        try:
            self.model = YOLO(model_name)
            self.class_names = self.model.names
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            return False
    
    def set_tracking_mode(self):
        mode_map = {
            0: "all",
            1: "vehicles",
            2: "people"
        }
        self.track_mode = mode_map.get(self.track_combo.currentIndex(), "all")
    
    def start_tracking(self):
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "Warning", "Please select and load a video source first!")
            return
        
        if not self.load_model():
            return
        
        if len([z for z in self.zones if z[0] == "trigger"]) == 0:
            QMessageBox.warning(self, "Warning", "Please define at least one trigger zone!")
            return
        
        self.set_tracking_mode()
        self.define_zones_mode = False
        self.paused = False
        self.start_time = time.time()
        
        # Исправление ошибки с таймером
        timer_interval = max(1, int(1000 / self.fps))  # Гарантируем целое число
        self.timer.start(timer_interval)
        
        self.start_tracking_btn.setEnabled(False)
        self.zone_group.setEnabled(False)
    
    def toggle_play(self):
        if self.cap is None:
            return
            
        if self.paused:
            self.paused = False
            timer_interval = max(1, int(1000 / self.fps))
            self.timer.start(timer_interval)
            self.play_button.setIcon(QIcon.fromTheme("media-playback-pause"))
        else:
            self.paused = True
            self.timer.stop()
            self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
    
    def pause_video(self):
        self.paused = True
        self.timer.stop()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
    
    def stop_video(self):
        self.paused = True
        self.timer.stop()
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.update_frame_display()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
    
    def set_position(self, position):
        if self.cap is not None and isinstance(self.video_source, str):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.update_frame_display()
    
    def delete_zones(self, zone_type=None):
        if zone_type is None:
            self.zones = []
        else:
            self.zones = [zone for zone in self.zones if zone[0] != zone_type]
        self.update_frame_display()
    
    def cleanup_old_objects(self):
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
    
    def update_frame(self):
        if self.paused or self.define_zones_mode or self.cap is None:
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.video_source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        self.paused = True
                        return
                else:
                    self.paused = True
                    return
            
            self.current_frame = frame.copy()
            
            # Обработка кадра с YOLO
            if self.model is not None and not self.define_zones_mode:
                # Получаем фильтр классов
                classes = self.class_filters.get(self.track_mode, None)
                
                # Запускаем трекинг
                results = self.model.track(
                    frame, 
                    persist=True, 
                    tracker="botsort.yaml",
                    classes=classes,
                    verbose=False
                )
                
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []
                    classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []
                    
                    for box, obj_id, cls_id in zip(boxes, ids, classes):
                        x1, y1, x2, y2 = map(int, box)
                        center = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        if obj_id not in self.object_colors:
                            self.object_colors[obj_id] = (
                                random.randint(0, 255),
                                random.randint(0, 255), 
                                random.randint(0, 255)
                            )
                            self.object_in_trigger[obj_id] = False
                        
                        self.last_seen[obj_id] = time.time()
                        
                        # Проверка триггерных зон
                        in_trigger = any(
                            self.is_inside_zone(center, points) 
                            for zone_type, points in self.zones 
                            if zone_type == "trigger"
                        )
                        
                        if in_trigger and not self.object_in_trigger.get(obj_id, False):
                            self.object_in_trigger[obj_id] = True
                            self.trajectories[obj_id] = []
                            self.active_objects[obj_id] = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
                        
                        # Проверка зон удаления
                        in_delete = any(
                            self.is_inside_zone(center, points) 
                            for zone_type, points in self.zones 
                            if zone_type == "delete"
                        )
                        
                        if in_delete and obj_id in self.trajectories:
                            del self.trajectories[obj_id]
                            if obj_id in self.active_objects:
                                del self.active_objects[obj_id]
                            self.object_in_trigger[obj_id] = False
                        
                        if self.object_in_trigger.get(obj_id, False):
                            if obj_id not in self.trajectories:
                                self.trajectories[obj_id] = []
                            self.trajectories[obj_id].append(center)
            
            self.update_frame_display()
        
        except Exception as e:
            print(f"Error in frame processing: {e}")
            self.paused = True
            self.timer.stop()
    
    def is_inside_zone(self, point, points):
        if len(points) < 3:
            return False
        
        # Конвертируем точки в numpy массив
        polygon = np.array(points)
        
        # Используем OpenCV для проверки точки внутри полигона
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0
    
    def calculate_scale_and_offset(self):
        if self.original_frame_size.isEmpty():
            return 1.0, QPoint(0, 0)
        
        frame_width = self.original_frame_size.width()
        frame_height = self.original_frame_size.height()
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        
        width_ratio = label_width / frame_width
        height_ratio = label_height / frame_height
        self.scale_factor = min(width_ratio, height_ratio)
        
        scaled_width = int(frame_width * self.scale_factor)
        scaled_height = int(frame_height * self.scale_factor)
        
        x_offset = (label_width - scaled_width) // 2
        y_offset = (label_height - scaled_height) // 2
        self.offset = QPoint(x_offset, y_offset)
        
        return self.scale_factor, self.offset
    
    def map_point_to_frame(self, point):
        """Преобразует координаты из QLabel в координаты оригинального кадра"""
        x = (point.x() - self.offset.x()) / self.scale_factor
        y = (point.y() - self.offset.y()) / self.scale_factor
        
        # Ограничиваем координаты размерами кадра
        x = max(0, min(x, self.original_frame_size.width() - 1))
        y = max(0, min(y, self.original_frame_size.height() - 1))
        
        return (int(x), int(y))
    
    def update_frame_display(self):
        if self.current_frame is None:
            return
        
        # Рассчитываем масштаб и смещение
        self.calculate_scale_and_offset()
        
        # Конвертируем кадр в QImage
        frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Создаем QPixmap для рисования
        scaled_pixmap = pixmap.scaled(
            int(w * self.scale_factor), 
            int(h * self.scale_factor), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        final_pixmap = QPixmap(self.video_label.size())
        final_pixmap.fill(Qt.black)
        
        painter = QPainter(final_pixmap)
        painter.drawPixmap(self.offset, scaled_pixmap)
        
        # Рисуем зоны с учетом масштаба и смещения
        for zone_type, points in self.zones:
            color = self.zone_colors[zone_type]
            pen = QPen(color, 2)
            painter.setPen(pen)
            
            if len(points) > 1:
                scaled_points = []
                for point in points:
                    scaled_x = int(point[0] * self.scale_factor + self.offset.x())
                    scaled_y = int(point[1] * self.scale_factor + self.offset.y())
                    scaled_points.append(QPoint(scaled_x, scaled_y))
                
                for i in range(1, len(scaled_points)):
                    painter.drawLine(scaled_points[i-1], scaled_points[i])
                # Замыкаем полигон
                painter.drawLine(scaled_points[-1], scaled_points[0])
            
            for point in points:
                scaled_x = int(point[0] * self.scale_factor + self.offset.x())
                scaled_y = int(point[1] * self.scale_factor + self.offset.y())
                painter.setBrush(color)
                painter.drawEllipse(QPoint(scaled_x, scaled_y), 5, 5)
        
        # Рисуем текущую зону (если рисуем)
        if (self.define_zones_mode or self.paused) and self.current_zone:
            zone_type = self.current_zone_type
            color = self.zone_colors[zone_type]
            pen = QPen(color, 2)
            painter.setPen(pen)
            
            if len(self.current_zone) > 1:
                for i in range(1, len(self.current_zone)):
                    x1 = int(self.current_zone[i-1][0] * self.scale_factor + self.offset.x())
                    y1 = int(self.current_zone[i-1][1] * self.scale_factor + self.offset.y())
                    x2 = int(self.current_zone[i][0] * self.scale_factor + self.offset.x())
                    y2 = int(self.current_zone[i][1] * self.scale_factor + self.offset.y())
                    painter.drawLine(x1, y1, x2, y2)
            
            for point in self.current_zone:
                x = int(point[0] * self.scale_factor + self.offset.x())
                y = int(point[1] * self.scale_factor + self.offset.y())
                painter.setBrush(color)
                painter.drawEllipse(x-5, y-5, 10, 10)
        
        # Рисуем траектории
        for obj_id in self.trajectories:
            if len(self.trajectories[obj_id]) > 1:
                color = QColor(*self.object_colors.get(obj_id, (255, 0, 0)))
                pen = QPen(color, 2)
                painter.setPen(pen)
                
                for i in range(1, len(self.trajectories[obj_id])):
                    x1 = int(self.trajectories[obj_id][i-1][0] * self.scale_factor + self.offset.x())
                    y1 = int(self.trajectories[obj_id][i-1][1] * self.scale_factor + self.offset.y())
                    x2 = int(self.trajectories[obj_id][i][0] * self.scale_factor + self.offset.x())
                    y2 = int(self.trajectories[obj_id][i][1] * self.scale_factor + self.offset.y())
                    painter.drawLine(x1, y1, x2, y2)
        
        # Рисуем информацию о режиме
        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)
        
        mode = "PAUSED" if self.paused else "ZONE DEFINITION" if self.define_zones_mode else "TRACKING"
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(10, 30, f"MODE: {mode}")
        
        if self.define_zones_mode or self.paused:
            zone_type = "Trigger" if self.current_zone_type == "trigger" else "Delete"
            painter.drawText(10, 60, f"Zone type: {zone_type}")
        
        if not self.define_zones_mode:
            painter.drawText(final_pixmap.width() - 200, 30, f"Active objects: {len(self.active_objects)}")
            current_time = time.time() - self.start_time
            time_str = time.strftime("%H:%M:%S", time.gmtime(current_time))
            painter.drawText(final_pixmap.width() - 150, 60, time_str)
        
        painter.end()
        
        # Устанавливаем изображение в QLabel
        self.video_label.setPixmap(final_pixmap)
    
    def mousePressEvent(self, event):
        if (self.define_zones_mode or self.paused) and event.button() == Qt.LeftButton:
            # Получаем координаты относительно video_label
            pos = event.pos()
            label_pos = self.video_label.mapFromParent(pos)
            
            if not self.video_label.rect().contains(label_pos):
                return
                
            # Преобразуем координаты к оригинальному масштабу
            frame_x, frame_y = self.map_point_to_frame(label_pos)
            
            # Добавляем точку в текущую зону
            self.current_zone.append((frame_x, frame_y))
            self.drawing = True
            self.update_frame_display()
    
    def mouseReleaseEvent(self, event):
        if (self.define_zones_mode or self.paused) and event.button() == Qt.RightButton and self.drawing:
            if len(self.current_zone) > 2:
                self.zones.append((self.current_zone_type, self.current_zone.copy()))
            self.current_zone = []
            self.drawing = False
            self.update_frame_display()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_C:
            self.current_zone_type = "delete" if self.current_zone_type == "trigger" else "trigger"
            self.zone_type_combo.setCurrentIndex(1 if self.current_zone_type == "delete" else 0)
            self.update_frame_display()
        elif event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_Return:
            self.start_tracking()
        elif event.key() == Qt.Key_D:
            self.delete_zones()
        elif event.key() == Qt.Key_T:
            self.delete_zones("trigger")
        elif event.key() == Qt.Key_E:
            self.delete_zones("delete")
        elif event.key() == Qt.Key_Q:
            self.close()
    
    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        if self.timer.isActive():
            self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    player = VideoPlayer()
    player.show()
    app.exec_()