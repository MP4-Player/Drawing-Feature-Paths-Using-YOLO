import os
import cv2
import random
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QSlider, QLabel, QFrame, QComboBox, QGroupBox, 
                            QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QPainter, QColor, QPen
from ultralytics import YOLO

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
        self.source_layout.addWidget(self.source_combo)
        
        self.webcam_combo = QComboBox()
        self.webcam_combo.setVisible(False)
        self.source_layout.addWidget(self.webcam_combo)
        
        self.browse_video_btn = QPushButton("Select Video File")
        self.browse_video_btn.setVisible(False)
        self.browse_video_btn.clicked.connect(self.browse_video)
        self.source_layout.addWidget(self.browse_video_btn)
        
        self.source_combo.currentIndexChanged.connect(self.update_source_ui)
        self.update_source_ui(0)
        
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
        
        # Загрузка доступных камер
        self.load_available_cameras()
        
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
        
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.read()[0]:
                available_cameras.append(i)
                cap.release()
            else:
                cap.release()
        
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
            self.load_video()
    
    def load_video(self):
        if self.cap is not None:
            self.cap.release()
        
        if isinstance(self.video_source, int):
            self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"Error opening video source: {self.video_source}")
            return False
        
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.critical(self, "Error", "Error reading first frame!")
            self.cap.release()
            return False
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default FPS if not detected
        
        self.timeline_slider.setRange(0, self.frame_count)
        self.current_frame = frame.copy()
        self.paused = True
        self.timer.stop()
        
        self.update_frame_display()
        return True
    
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
            QMessageBox.warning(self, "Warning", "Please select a video source first!")
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
        self.timer.start(1000 / self.fps)
        
        self.start_tracking_btn.setEnabled(False)
        self.zone_group.setEnabled(False)
    
    def toggle_play(self):
        if self.cap is None:
            return
            
        if self.paused:
            self.paused = False
            self.timer.start(1000 / self.fps)
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
            self.current_frame = 0
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.update_frame_display()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
    
    def set_position(self, position):
        if self.cap is not None:
            self.current_frame = position
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
        
        ret, frame = self.cap.read()
        if not ret:
            if isinstance(self.video_source, str):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
                        self.is_inside_zone(center, ("trigger", zone)) 
                        for zone_type, zone in self.zones 
                        if zone_type == "trigger"
                    )
                    
                    if in_trigger and not self.object_in_trigger.get(obj_id, False):
                        self.object_in_trigger[obj_id] = True
                        self.trajectories[obj_id] = []
                        self.active_objects[obj_id] = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
                    
                    # Проверка зон удаления
                    in_delete = any(
                        self.is_inside_zone(center, ("delete", zone)) 
                        for zone_type, zone in self.zones 
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
    
    def is_inside_zone(self, point, zone):
        zone_type, points = zone
        if len(points) < 3:
            return False
        
        # Конвертируем точки в numpy массив
        polygon = np.array(points)
        
        # Используем OpenCV для проверки точки внутри полигона
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0
    
    def update_frame_display(self):
        if self.current_frame is None:
            return
        
        # Конвертируем кадр в QImage
        frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Создаем QPixmap для рисования
        painter = QPainter(pixmap)
        
        # Рисуем зоны
        for zone in self.zones:
            zone_type, points = zone
            color = self.zone_colors[zone_type]
            pen = QPen(color, 2)
            painter.setPen(pen)
            
            if len(points) > 1:
                for i in range(1, len(points)):
                    painter.drawLine(points[i-1][0], points[i-1][1], 
                                   points[i][0], points[i][1])
                # Замыкаем полигон
                painter.drawLine(points[-1][0], points[-1][1], 
                               points[0][0], points[0][1])
            
            for point in points:
                painter.setBrush(color)
                painter.drawEllipse(point[0]-5, point[1]-5, 10, 10)
        
        # Рисуем текущую зону (если рисуем)
        if (self.define_zones_mode or self.paused) and self.current_zone:
            zone_type = self.current_zone_type
            color = self.zone_colors[zone_type]
            pen = QPen(color, 2)
            painter.setPen(pen)
            
            if len(self.current_zone) > 1:
                for i in range(1, len(self.current_zone)):
                    painter.drawLine(self.current_zone[i-1][0], self.current_zone[i-1][1], 
                                   self.current_zone[i][0], self.current_zone[i][1])
            
            for point in self.current_zone:
                painter.setBrush(color)
                painter.drawEllipse(point[0]-5, point[1]-5, 10, 10)
        
        # Рисуем траектории
        for obj_id in self.trajectories:
            if len(self.trajectories[obj_id]) > 1:
                color = QColor(*self.object_colors.get(obj_id, (255, 0, 0)))
                pen = QPen(color, 2)
                painter.setPen(pen)
                
                for i in range(1, len(self.trajectories[obj_id])):
                    painter.drawLine(
                        self.trajectories[obj_id][i-1][0], self.trajectories[obj_id][i-1][1],
                        self.trajectories[obj_id][i][0], self.trajectories[obj_id][i][1]
                    )
        
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
            painter.drawText(w - 200, 30, f"Active objects: {len(self.active_objects)}")
            current_time = time.time() - self.start_time
            time_str = time.strftime("%H:%M:%S", time.gmtime(current_time))
            painter.drawText(w - 150, 60, time_str)
        
        painter.end()
        
        # Устанавливаем изображение в QLabel
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def mousePressEvent(self, event):
        if (self.define_zones_mode or self.paused) and event.button() == Qt.LeftButton:
            # Получаем координаты относительно video_label
            pos = self.video_label.mapFromParent(event.pos())
            if not self.video_label.rect().contains(pos):
                return
                
            # Масштабируем координаты к оригинальному размеру видео
            pixmap = self.video_label.pixmap()
            if pixmap is not None:
                x_scale = pixmap.width() / self.video_label.width()
                y_scale = pixmap.height() / self.video_label.height()
                
                x = int(pos.x() * x_scale)
                y = int(pos.y() * y_scale)
                
                self.current_zone.append((x, y))
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
            self.update_frame_display()
        elif event.key() == Qt.Key_Space:
            self.toggle_play()
    
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