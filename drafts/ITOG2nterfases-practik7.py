
import os  
import cv2  
import random  
import time  
import numpy as np  
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QSlider, QLabel, QFrame, QComboBox, QGroupBox, 
                            QFileDialog, QMessageBox, QRadioButton, QButtonGroup)  # Элементы GUI
from PyQt5.QtCore import Qt, QTimer, QSize, QPoint, QRect  # Базовые классы Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QPainter, QColor, QPen, QPolygon  # Графика Qt
from ultralytics import YOLO  #модель для детекции объектов

# Устранение конфликта OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Уменьшаем вывод ошибок OpenCV
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# Основной класс приложения
class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()  # Инициализация родительского класса
        self.setWindowTitle("YOLOv8 Trajectory Tracker")  # заголовка окна
        self.setMinimumSize(1000, 700) 
        
        # Инициализация переменных для трекинга
        self.drawing = False  # Флаг рисования зон
        self.current_zone = []  # Текущая рисуемая зона
        self.zones = []  # Все созданные зоны
        self.current_zone_type = "trigger"  # Тип зоны (trigger/delete)
        self.define_zones_mode = True  # Режим определения зон
        self.trajectories = {}  # Словарь траекторий объектов
        self.object_colors = {}  # Цвета для каждого объекта
        self.active_objects = {}  # Активные объекты (в зонах)
        self.object_in_trigger = {}  # Флаги нахождения в триггер-зоне
        self.paused = True  # Пауза воспроизведения
        self.current_frame = None  # Текущий кадр
        self.cap = None  # Объект захвата видео
        self.model = None  # Модель YOLO
        self.track_mode = "all"  # Режим трекинга (all/vehicles/people)
        self.video_source = None  # Источник видео (камера/файл)
        self.start_time = 0  # Время начала трекинга
        self.last_seen = {}  # Время последнего обнаружения объектов
        self.frame_count = 0  # Общее количество кадров (для видеофайлов)
        self.current_frame_pos = 0  # Текущая позиция в видео
        self.fps = 30  # Частота кадров
        self.scale_factor = 1.0  # Масштаб отображения
        self.offset = QPoint(0, 0)  # Смещение для центрирования
        self.original_frame_size = QSize(640, 480)  # Исходный размер кадра
        self.last_ids = set()  # ID объектов на предыдущем кадре
        self.tracking_active = False  # Флаг активного трекинга
        self.display_mode = "all"  # Режим отображения (all/zones_only)
        
        self.zone_colors = {
            "trigger": QColor(0, 255, 0, 100),  # Зеленый для триггер-зон
            "delete": QColor(255, 0, 0, 100)  # Красный для зон удаления
        }
        
        self.class_filters = {
            "vehicles": [2, 3, 5, 7],  # car, motorcycle, bus, truck
            "people": [0],  # person
            "all": None  # Все классы
        }
        
        # Инициализация пользовательского интерфейса
        self.init_ui()
        self.apply_styles()  # Применение стилей
        self.load_available_cameras()  # Загрузка доступных камер
    
    def init_ui(self):
        # Главный виджет и макет
        self.main_widget = QWidget()  # Создание главного виджета
        self.setCentralWidget(self.main_widget)  # Установка как центрального виджета
        
        self.main_layout = QHBoxLayout()  # Горизонтальный макет
        self.main_widget.setLayout(self.main_layout)  # Установка макета для главного виджета
        
        # Панель видео
        self.video_frame = QFrame() 
        self.video_frame.setFrameShape(QFrame.StyledPanel)  # Стиль фрейма
        self.video_layout = QVBoxLayout()  # Вертикальный макет для видео
        self.video_frame.setLayout(self.video_layout)  # Установка макета
        
        self.video_label = QLabel()  
        self.video_label.setAlignment(Qt.AlignCenter)  # Выравнивание по центру
        self.video_label.setMinimumSize(640, 480)  # Минимальный размер
        
        self.video_label.mousePressEvent = self.mousePressEvent
        self.video_label.mouseReleaseEvent = self.mouseReleaseEvent
        self.video_layout.addWidget(self.video_label)  # Добавление метки в макет
        
        
        self.timeline_slider = QSlider(Qt.Horizontal)  # Горизонтальный слайдер
        # Привязка событий слайдера
        self.timeline_slider.sliderMoved.connect(self.set_position)
        self.timeline_slider.sliderPressed.connect(self.pause_video)
        self.timeline_slider.sliderReleased.connect(self.toggle_play)
        self.video_layout.addWidget(self.timeline_slider)  # Добавление слайдера
        
        # Панель управления
        self.control_panel = QFrame()  
        self.control_panel.setFrameShape(QFrame.StyledPanel)  
        self.control_panel.setFixedWidth(300) 
        
        self.control_layout = QVBoxLayout()  # Вертикальный макет
        self.control_panel.setLayout(self.control_layout) 
        
        # Группа источника видео
        self.source_group = QGroupBox("Video Source")  
        self.source_layout = QVBoxLayout()  
        self.source_group.setLayout(self.source_layout) 
        
        self.source_combo = QComboBox()  # Выпадающий список
        self.source_combo.addItems(["Webcam", "Video File"]) 
        self.source_combo.currentIndexChanged.connect(self.update_source_ui)
        self.source_layout.addWidget(self.source_combo)  
        
        self.webcam_combo = QComboBox()  # Выпадающий список камер
        self.webcam_combo.setVisible(False) 
        self.source_layout.addWidget(self.webcam_combo)  # Добавление в макет
        
        self.browse_video_btn = QPushButton("Select Video File")  # Кнопка выбора файла
        self.browse_video_btn.setVisible(False)  
        self.browse_video_btn.clicked.connect(self.browse_video)  # Привязка события
        self.source_layout.addWidget(self.browse_video_btn)  
        
        self.load_source_btn = QPushButton("Load Source")  # Кнопка загрузки источника
        self.load_source_btn.clicked.connect(self.load_source) 
        self.source_layout.addWidget(self.load_source_btn) 
        
        # Группа модели YOLO
        self.model_group = QGroupBox("YOLO Model")  
        self.model_layout = QVBoxLayout()  
        self.model_group.setLayout(self.model_layout) 
        
        self.model_combo = QComboBox()  # Выпадающий список моделей
        self.model_combo.addItems(["yolov8n.pt (nano)", "yolov8s.pt (small)", 
                                 "yolov8m.pt (medium)", "yolov8l.pt (large)", 
                                 "yolov8x.pt (xlarge)"])  
        self.model_layout.addWidget(self.model_combo) 
        
        # Группа режима трекинга
        self.track_group = QGroupBox("Tracking Mode")  
        self.track_layout = QVBoxLayout()  
        self.track_group.setLayout(self.track_layout)  
        
        self.track_combo = QComboBox()  # Выпадающий список режимов
        self.track_combo.addItems(["All objects", "Vehicles only", "People only"]) 
        self.track_layout.addWidget(self.track_combo)  
        
        # Группа отображения объектов
        self.display_group = QGroupBox("Display Mode")  
        self.display_layout = QVBoxLayout() 
        self.display_group.setLayout(self.display_layout)  
        
        self.display_all_radio = QRadioButton("Show all objects")  # Зонирование трекинга
        self.display_zones_radio = QRadioButton("Show only objects in zones") 
        self.display_all_radio.setChecked(True)  
        
        self.display_button_group = QButtonGroup()  # Кнопочки
        self.display_button_group.addButton(self.display_all_radio, 1)  
        self.display_button_group.addButton(self.display_zones_radio, 2) 
        self.display_button_group.buttonClicked.connect(self.change_display_mode)
        #В макет
        self.display_layout.addWidget(self.display_all_radio)  
        self.display_layout.addWidget(self.display_zones_radio)  
        
        # Группа управления зонами
        self.zone_group = QGroupBox("Zone Management")  
        self.zone_layout = QVBoxLayout() 
        self.zone_group.setLayout(self.zone_layout)  
        
        self.zone_type_combo = QComboBox()  
        self.zone_type_combo.addItems(["Trigger Zone", "Delete Zone"])  
        # Привязка события изменения индекса
        self.zone_type_combo.currentIndexChanged.connect(self.change_zone_type)
        self.zone_layout.addWidget(self.zone_type_combo)  
        
        # Группа управления воспроизведением
        self.playback_group = QGroupBox("Playback Controls") 
        self.playback_layout = QHBoxLayout() 
        self.playback_group.setLayout(self.playback_layout)  
        
        self.play_button = QPushButton("Play")  
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))  #Иконка
        self.play_button.clicked.connect(self.toggle_play)  
        
        self.pause_button = QPushButton("Pause")  
        self.pause_button.setIcon(QIcon.fromTheme("media-playback-pause"))  
        self.pause_button.clicked.connect(self.pause_video) 
        
        self.stop_button = QPushButton("Stop")  
        self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop")) 
        self.stop_button.clicked.connect(self.stop_video)  
        
        self.playback_layout.addWidget(self.play_button)  
        self.playback_layout.addWidget(self.pause_button)  
        self.playback_layout.addWidget(self.stop_button)  
        
        # Группа управления зонами 
        self.clear_zones_btn = QPushButton("Clear All Zones")  # Кнопка 
        self.clear_zones_btn.clicked.connect(lambda: self.delete_zones())  # Привязка события
        self.zone_layout.addWidget(self.clear_zones_btn)  # Добавление в макет
        
        self.clear_trigger_btn = QPushButton("Clear Trigger Zones") 
        self.clear_trigger_btn.clicked.connect(lambda: self.delete_zones("trigger"))  
        self.zone_layout.addWidget(self.clear_trigger_btn) 
        
        self.clear_delete_btn = QPushButton("Clear Delete Zones")  
        self.clear_delete_btn.clicked.connect(lambda: self.delete_zones("delete"))  
        self.zone_layout.addWidget(self.clear_delete_btn)  
        
        # Кнопка запуска трекинга
        self.start_tracking_btn = QPushButton("Start Tracking")  # Кнопка старта трекинга
        self.start_tracking_btn.clicked.connect(self.start_tracking)  # Привязка события
        
        self.control_layout.addWidget(self.source_group)  # Добавление группы источника
        self.control_layout.addWidget(self.model_group)  # Добавление группы модели
        self.control_layout.addWidget(self.track_group)  # Добавление группы трекинга
        self.control_layout.addWidget(self.display_group)  # Добавление группы отображения
        self.control_layout.addWidget(self.zone_group)  # Добавление группы зон
        self.control_layout.addWidget(self.playback_group)  # Добавление группы воспроизведения
        self.control_layout.addWidget(self.start_tracking_btn)  # Добавление кнопки старта
        self.control_layout.addStretch()  # Добавление растягивающегося пространства
        self.main_layout.addWidget(self.video_frame, stretch=1)  # Добавление видео с растягиванием
        self.main_layout.addWidget(self.control_panel)  # Добавление панели управления
        
        # Таймер для обновления видео
        self.timer = QTimer(self)  
        self.timer.timeout.connect(self.update_frame) 
    
    def apply_styles(self):
        # Установка стилей для элементов интерфейса
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
                min-height: 30px;
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
            QRadioButton {
                color: #ffffff;
                padding: 5px;
            }
        """)
        
        # Установка размеров иконок
        self.play_button.setIconSize(QSize(16, 16))
        self.pause_button.setIconSize(QSize(16, 16))
        self.stop_button.setIconSize(QSize(16, 16))
    
    def load_available_cameras(self):
        self.webcam_combo.clear()  
        available_cameras = []  
        
        # Проверка доступных камер (до 3 камер)
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Попытка открыть камеру если она доступна 
                if cap is not None and cap.isOpened():  
                    available_cameras.append(i)  # Добавление в выпадающий список
                    cap.release()  # Закрытие камеры
            except:
                pass  
        
        if available_cameras: 
            for cam_idx in available_cameras:
                self.webcam_combo.addItem(f"Camera {cam_idx}", cam_idx)  # Добавление в список
        else:
            self.webcam_combo.addItem("No cameras found", -1)  # Сообщение, если камер нет
    
    def update_source_ui(self, index):
        # Обновление интерфейса в зависимости от выбранного источника
        if index == 0:  # Webcam
            self.webcam_combo.setVisible(True)  # Показать список камер
            self.browse_video_btn.setVisible(False)  # Скрыть кнопку выбора файла
        else:  # Video File
            self.webcam_combo.setVisible(False)  # Скрыть список камер
            self.browse_video_btn.setVisible(True)  # Показать кнопку выбора файла
    
    def browse_video(self):
        # Диалог выбора видеофайла
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file:  # Если файл выбран
            self.video_source = file  # Сохранение пути к файлу
    
    def load_source(self):
        # Загрузка выбранного источника видео
        if self.source_combo.currentIndex() == 0:  # Webcam
            cam_idx = self.webcam_combo.currentData()  # Получение ID камеры
            if cam_idx == -1:  # Если камеры не найдены
                QMessageBox.warning(self, "Warning", "No cameras available!")  # Показать предупреждение
                return
            self.video_source = cam_idx  # Установка источника
        else:  # Video File
            if not self.video_source:  # Если файл не выбран
                QMessageBox.warning(self, "Warning", "Please select a video file first!")  # Показать предупреждение
                return
        
        if self.cap is not None:  # Если источник уже загружен
            self.cap.release()  # Освобождение ресурсов
            self.cap = None
        
        try:
            if isinstance(self.video_source, int):  # Если источник - камера
                self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)  # Открытие камеры
                if not self.cap.isOpened():  # Если не удалось открыть
                    raise Exception(f"Could not open camera {self.video_source}")
                
                # Установка параметров камеры
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                ret, frame = self.cap.read()  # Чтение первого кадра
                if not ret:  # Если не удалось прочитать
                    raise Exception("Could not read frame from camera")
                
                # Сохранение параметров кадра
                self.original_frame_size = QSize(frame.shape[1], frame.shape[0])
                self.current_frame = frame.copy()  # Сохранение кадра
                self.paused = True  # Установка паузы
                self.timer.stop()  # Остановка таймера
                self.update_frame_display()  # Обновление отображения
                
            else:  # Если источник - файл
                self.cap = cv2.VideoCapture(self.video_source)  # Открытие файла
                if not self.cap.isOpened():  # Если не удалось открыть
                    raise Exception(f"Could not open video file {self.video_source}")
                
                # Получение параметров видео
                self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0:  # Если FPS не определен
                    self.fps = 30  # Установка значения по умолчанию
                
                ret, frame = self.cap.read()  # Чтение первого кадра
                if not ret:  # Если не удалось прочитать
                    raise Exception("Could not read first frame from video")
                
                # Сохранение параметров кадра
                self.original_frame_size = QSize(frame.shape[1], frame.shape[0])
                self.current_frame = frame.copy()  # Сохранение кадра
                self.paused = True  # Установка паузы
                self.timer.stop()  # Остановка таймера
                self.timeline_slider.setRange(0, self.frame_count)  # Установка диапазона слайдера
                self.current_frame_pos = 0  # Сброс позиции
                self.update_frame_display()  # Обновление отображения
                
        except Exception as e:  # Обработка ошибок
            QMessageBox.critical(self, "Error", f"Failed to load source: {str(e)}")  # Показать ошибку
            if self.cap is not None:  # Если источник был открыт
                self.cap.release()  # Освобождение ресурсов
                self.cap = None
            return
    
    def change_zone_type(self, index):
        # Изменение типа зоны
        self.current_zone_type = "trigger" if index == 0 else "delete"
    
    def change_display_mode(self, button):
        # Изменение режима отображения
        self.display_mode = "all" if button == self.display_all_radio else "zones_only"
        self.update_frame_display()  # Обновление отображения
    
    def load_model(self):
        # Загрузка модели YOLO
        model_map = {
            0: "yolov8n.pt",  
            1: "yolov8s.pt",  
            2: "yolov8m.pt",  
            3: "yolov8l.pt",  
            4: "yolov8x.pt"   
        }
        
        model_idx = self.model_combo.currentIndex()  # Получение выбранного индекса
        model_name = model_map.get(model_idx, "yolov8n.pt")  # Получение имени модели
        
        try:
            self.model = YOLO(model_name)  # Загрузка модели
            self.model.fuse() # Оптимизация модели
            self.class_names = self.model.names  # Получение имен классов
            return True  # Успешная загрузка
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")  # Показать ошибку
            return False  # Ошибка загрузки
    
    def set_tracking_mode(self):
        # Установка режима трекинга
        mode_map = {
            0: "all",  # Все объекты
            1: "vehicles",  # Только транспорт
            2: "people"  # Только люди
        }
        self.track_mode = mode_map.get(self.track_combo.currentIndex(), "all")  # Получение режима
    
    def start_tracking(self):
        # Запуск трекинга
        if self.cap is None or not self.cap.isOpened():  # Если источник не загружен
            QMessageBox.warning(self, "Warning", "Please select and load a video source first!")  # Показать предупреждение
            return
        
        if not self.load_model():  # Если не удалось загрузить модель
            return
        
        # Проверка наличия триггер-зон в режиме отображения только зон
        if len([z for z in self.zones if z[0] == "trigger"]) == 0 and self.display_mode == "zones_only":
            QMessageBox.warning(self, "Warning", "Please define at least one trigger zone!")  # Показать предупреждение
            return
        
        self.set_tracking_mode()  # Установка режима трекинга
        self.define_zones_mode = False  # Выход из режима определения зон
        self.paused = False  # Снятие паузы
        self.start_time = time.time()  # Запись времени начала
        self.tracking_active = True  # Активация трекинга
        
        # Сброс предыдущих данных трекинга
        self.trajectories = {}
        self.object_colors = {}
        self.active_objects = {}
        self.object_in_trigger = {}
        self.last_seen = {}
        self.last_ids = set()
        
        timer_interval = max(1, int(1000 / self.fps))  # Расчет интервала таймера
        self.timer.start(timer_interval)  # Запуск таймера
        
        self.start_tracking_btn.setEnabled(False)  # Отключение кнопки старта
        self.zone_group.setEnabled(False)  # Отключение группы зон
    
    def toggle_play(self):
        # Переключение воспроизведения/паузы
        if self.cap is None:  # Если источник не загружен
            return
            
        if self.paused:  # Если на паузе
            self.paused = False  # Снятие паузы
            timer_interval = max(1, int(1000 / self.fps))  # Расчет интервала
            self.timer.start(timer_interval)  # Запуск таймера
            self.play_button.setIcon(QIcon.fromTheme("media-playback-pause"))  # Смена иконки
            self.play_button.setText("Pause")  # Смена текста
        else:  # Если воспроизводится
            self.pause_video()  # Постановка на паузу
    
    def pause_video(self):
        # Постановка на паузу
        self.paused = True  # Установка флага паузы
        self.timer.stop()  # Остановка таймера
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))  # Смена иконки
        self.play_button.setText("Play")  # Смена текста
    
    def stop_video(self):
        # Остановка воспроизведения
        self.paused = True  # Установка паузы
        self.tracking_active = False  # Остановка трекинга
        self.timer.stop()  # Остановка таймера
        if self.cap is not None:  # Если источник загружен
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Перемотка в начало
            ret, frame = self.cap.read()  # Чтение первого кадра
            if ret:  # Если кадр прочитан
                self.current_frame = frame.copy()  # Сохранение кадра
                self.current_frame_pos = 0  # Сброс позиции
                self.timeline_slider.setValue(0)  # Сброс слайдера
                self.update_frame_display()  # Обновление отображения
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))  # Смена иконки
        self.play_button.setText("Play")  # Смена текста
    
    def set_position(self, position):
        # Установка позиции воспроизведения
        if self.cap is not None and isinstance(self.video_source, str):  # Если источник - файл
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)  # Установка позиции
            ret, frame = self.cap.read()  # Чтение кадра
            if ret:  # Если кадр прочитан
                self.current_frame = frame.copy()  # Сохранение кадра
                self.current_frame_pos = position  # Сохранение позиции
                self.update_frame_display()  # Обновление отображения
    
    def delete_zones(self, zone_type=None):
        # Удаление зон
        if zone_type is None:  # Если тип не указан
            self.zones = []  # Удаление всех зон
        else:  # Если тип указан
            self.zones = [zone for zone in self.zones if zone[0] != zone_type]  # Удаление зон определенного типа
        self.update_frame_display()  # Обновление отображения
    
    def cleanup_old_objects(self):
        # Очистка старых объектов
        current_time = time.time()  # Текущее время
        
        # Поиск объектов, которые не видели более 4 секунд
        to_delete = []
        for obj_id in list(self.trajectories.keys()):
            if current_time - self.last_seen.get(obj_id, 0) > 4.0:
                to_delete.append(obj_id)
        
        # Удаление старых объектов
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
        # Обновление кадра
        if self.paused or self.define_zones_mode or self.cap is None:  # Если пауза или режим зон или нет источника
            return
        
        try:
            ret, frame = self.cap.read()  # Чтение кадра
            if not ret:  # Если кадр не прочитан
                if isinstance(self.video_source, str):  # Если источник - файл
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Перемотка в начало
                    ret, frame = self.cap.read()  # Чтение кадра
                    if not ret:  # Если кадр не прочитан
                        self.paused = True  # Установка паузы
                        return
                    self.current_frame_pos = 0  # Сброс позиции
                else:  # Если источник - камера
                    self.paused = True  # Установка паузы
                    return
            
            self.current_frame = frame.copy()  # Сохранение кадра
            
            # Обновление позиции слайдера для видеофайлов
            if isinstance(self.video_source, str):
                self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.timeline_slider.setValue(self.current_frame_pos)
            
            if self.model is not None and not self.define_zones_mode:  # Если модель загружена и не режим зон
                classes = self.class_filters.get(self.track_mode, None)  # Получение классов для фильтрации
                
                # Трекинг объектов с помощью YOLO
                results = self.model.track(
                    frame,
                    persist=True,  # Сохранение ID между кадрами
                    tracker="botsort.yaml",  # Алгоритм трекинга
                    classes=self.class_filters[self.track_mode],  # Фильтр классов
                    conf=0.5,   # Порог уверенности
                    iou=0.5,    # Порог перекрытия
                    imgsz=640,  # Размер изображения
                    device='cpu',  # Устройство (CPU)
                    half=False,  # Полуточность (False для CPU)
                    verbose=False  # Отключение лишних логов
                )
                
                current_ids = set()  # ID текущих объектов
                
                # Обработка результатов трекинга
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Координаты bounding box
                    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []  # ID объектов
                    classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []  # Классы объектов
                    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []  # Уверенность
                    
                    # Обработка каждого обнаруженного объекта
                    for box, obj_id, cls_id, conf in zip(boxes, ids, classes, confs):
                        obj_id = int(obj_id)  # Преобразование ID в int
                        cls_id = int(cls_id)  # Преобразование класса в int
                        x1, y1, x2, y2 = map(int, box)  # Координаты bounding box
                        center = (x1 + x2) // 2, (y1 + y2) // 2  # Центр объекта
                        
                        if obj_id not in self.object_colors:  # Если новый объект
                            # Генерация случайного цвета
                            self.object_colors[obj_id] = (
                                random.randint(0, 255),
                                random.randint(0, 255), 
                                random.randint(0, 255)
                            )
                            self.object_in_trigger[obj_id] = False  # Инициализация флага
                        
                        current_ids.add(obj_id)  # Добавление ID в текущий набор
                        self.last_seen[obj_id] = time.time()  # Обновление времени последнего обнаружения
                        
                        # Проверка триггерных зон
                        in_trigger = any(
                            self.is_inside_zone(center, points) 
                            for zone_type, points in self.zones 
                            if zone_type == "trigger"
                        )
                        
                        # Если объект вошел в триггер-зону
                        if in_trigger and not self.object_in_trigger.get(obj_id, False):
                            self.object_in_trigger[obj_id] = True  # Установка флага
                            if obj_id not in self.trajectories:  # Если нет траектории
                                self.trajectories[obj_id] = []  # Создание траектории
                            if obj_id not in self.active_objects:  # Если нет информации
                                self.active_objects[obj_id] = {
                                    'start_time': time.time() - self.start_time,  # Время входа
                                    'class_id': cls_id  # Класс объекта
                                }
                        
                        # Проверка зон удаления
                        in_delete = any(
                            self.is_inside_zone(center, points) 
                            for zone_type, points in self.zones 
                            if zone_type == "delete"
                        )
                        
                        # Если объект в зоне удаления
                        if in_delete and obj_id in self.trajectories:
                            del self.trajectories[obj_id]  # Удаление траектории
                            if obj_id in self.active_objects:
                                del self.active_objects[obj_id]  # Удаление информации
                            self.object_in_trigger[obj_id] = False  # Сброс флага
                        
                        # Для режима "all" добавляем все объекты
                        if self.display_mode == "all" and obj_id not in self.trajectories:
                            self.trajectories[obj_id] = []  # Создание траектории
                            self.active_objects[obj_id] = {
                                'start_time': time.time() - self.start_time,  # Время обнаружения
                                'class_id': cls_id  # Класс объекта
                            }
                        
                        if obj_id in self.trajectories:  # Если есть траектория
                            self.trajectories[obj_id].append(center)  # Добавление точки
                
                # Очистка объектов, которые больше не видны
                to_delete = [obj_id for obj_id in self.last_ids if obj_id not in current_ids]
                for obj_id in to_delete:
                    if obj_id in self.last_seen and (time.time() - self.last_seen[obj_id]) > 2.0:
                        if obj_id in self.trajectories and self.display_mode == "zones_only":
                            del self.trajectories[obj_id]  # Удаление траектории
                        if obj_id in self.active_objects and self.display_mode == "zones_only":
                            del self.active_objects[obj_id]  # Удаление информации
                        if obj_id in self.object_in_trigger:
                            del self.object_in_trigger[obj_id]  # Сброс флага
                
                self.last_ids = current_ids  # Обновление последних ID
                self.cleanup_old_objects()  # Очистка старых объектов
            
            self.update_frame_display()  # Обновление отображения
        
        except Exception as e:  # Обработка ошибок
            print(f"Error in frame processing: {e}")  # Вывод ошибки
            self.paused = True  # Установка паузы
            self.timer.stop()  # Остановка таймера
    
    def is_inside_zone(self, point, points):
        # Проверка, находится ли точка внутри зоны
        if len(points) < 3:  # Если зона не замкнута
            return False
        
        polygon = np.array(points)  # Преобразование в массив numpy
        result = cv2.pointPolygonTest(polygon, point, False)  # Проверка точки
        return result >= 0  # Возврат результата
    
    def calculate_scale_and_offset(self):
        # Расчет масштаба и смещения для отображения
        if self.original_frame_size.isEmpty():  # Если размер не определен
            return 1.0, QPoint(0, 0)
        
        frame_width = self.original_frame_size.width()  # Ширина кадра
        frame_height = self.original_frame_size.height()  # Высота кадра
        label_width = self.video_label.width()  # Ширина метки
        label_height = self.video_label.height()  # Высота метки
        
        # Расчет соотношений
        width_ratio = label_width / frame_width
        height_ratio = label_height / frame_height
        self.scale_factor = min(width_ratio, height_ratio)  # Выбор минимального масштаба
        
        # Расчет размеров после масштабирования
        scaled_width = int(frame_width * self.scale_factor)
        scaled_height = int(frame_height * self.scale_factor)
        
        # Расчет смещения для центрирования
        x_offset = (label_width - scaled_width) // 2
        y_offset = (label_height - scaled_height) // 2
        self.offset = QPoint(x_offset, y_offset)
        
        return self.scale_factor, self.offset  # Возврат масштаба и смещения
    
    def map_point_to_frame(self, point):
        """Преобразует координаты из QLabel в координаты оригинального кадра"""
        # Корректировка координат с учетом точного позиционирования
        label_pos = self.video_label.mapFromParent(point)  # Позиция относительно метки
        x = (label_pos.x() - self.offset.x()) / self.scale_factor  # Расчет X
        y = (label_pos.y() - self.offset.y()) / self.scale_factor  # Расчет Y
        
        # Ограничиваем координаты размерами кадра
        x = max(0, min(x, self.original_frame_size.width() - 1))
        y = max(0, min(y, self.original_frame_size.height() - 1))
        
        return (int(x), int(y))  # Возврат координат
    
    def update_frame_display(self):
        # Обновление отображения кадра
        if self.current_frame is None:  # Если кадра нет
            return
        
        self.calculate_scale_and_offset()  # Расчет масштаба и смещения
        
        # Преобразование кадра в RGB
        frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape  # Получение размеров
        bytes_per_line = ch * w  # Расчет байт на строку
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)  # Создание QImage
        pixmap = QPixmap.fromImage(q_img)  # Создание QPixmap
        
        # Масштабирование изображения
        scaled_pixmap = pixmap.scaled(
            int(w * self.scale_factor), 
            int(h * self.scale_factor), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # Создание финального изображения
        final_pixmap = QPixmap(self.video_label.size())
        final_pixmap.fill(Qt.black)  # Заполнение черным
        
        painter = QPainter(final_pixmap)  # Создание рисовальщика
        try:
            painter.drawPixmap(self.offset, scaled_pixmap)  # Отрисовка кадра
            
            # Рисуем зоны с прозрачностью
            for zone_type, points in self.zones:
                color = self.zone_colors[zone_type]  # Получение цвета
                pen = QPen(color, 2)  # Создание пера
                painter.setPen(pen)  # Установка пера
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 50))  # Установка кисти
                
                if len(points) > 1:  # Если есть точки
                    scaled_points = []
                    for point in points:  # Масштабирование точек
                        scaled_x = int(point[0] * self.scale_factor + self.offset.x())
                        scaled_y = int(point[1] * self.scale_factor + self.offset.y())
                        scaled_points.append(QPoint(scaled_x, scaled_y))
                    
                    polygon = QPolygon(scaled_points)  # Создание полигона
                    painter.drawPolygon(polygon)  # Отрисовка полигона
                
                for point in points:  # Отрисовка точек
                    scaled_x = int(point[0] * self.scale_factor + self.offset.x())
                    scaled_y = int(point[1] * self.scale_factor + self.offset.y())
                    painter.setBrush(color)  # Установка кисти
                    painter.drawEllipse(QPoint(scaled_x, scaled_y), 5, 5)  # Отрисовка круга
            
            # Рисуем текущую зону (если рисуем)
            if (self.define_zones_mode or self.paused) and self.current_zone:
                zone_type = self.current_zone_type  # Тип зоны
                color = self.zone_colors[zone_type]  # Цвет
                pen = QPen(color, 2)  # Перо
                painter.setPen(pen)  # Установка пера
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 50))  # Кисть
                
                if len(self.current_zone) > 1:  # Если есть точки
                    scaled_points = []
                    for point in self.current_zone:  # Масштабирование точек
                        scaled_x = int(point[0] * self.scale_factor + self.offset.x())
                        scaled_y = int(point[1] * self.scale_factor + self.offset.y())
                        scaled_points.append(QPoint(scaled_x, scaled_y))
                    
                    polygon = QPolygon(scaled_points)  # Создание полигона
                    painter.drawPolygon(polygon)  # Отрисовка
                
                for point in self.current_zone:  # Отрисовка точек
                    x = int(point[0] * self.scale_factor + self.offset.x())
                    y = int(point[1] * self.scale_factor + self.offset.y())
                    painter.setBrush(color)  # Установка кисти
                    painter.drawEllipse(x-5, y-5, 10, 10)  # Отрисовка круга
            
            # Рисуем траектории и bounding boxes
            if not self.define_zones_mode and self.model is not None:  # Если трекинг активен
                for obj_id in self.trajectories:  # Для каждой траектории
                    if len(self.trajectories[obj_id]) > 1:  # Если есть точки
                        color = QColor(*self.object_colors.get(obj_id, (255, 0, 0)))  # Цвет объекта
                        pen = QPen(color, 2)  # Перо
                        painter.setPen(pen)  # Установка пера
                        
                        # Рисуем траекторию
                        for i in range(1, len(self.trajectories[obj_id])):
                            x1 = int(self.trajectories[obj_id][i-1][0] * self.scale_factor + self.offset.x())
                            y1 = int(self.trajectories[obj_id][i-1][1] * self.scale_factor + self.offset.y())
                            x2 = int(self.trajectories[obj_id][i][0] * self.scale_factor + self.offset.x())
                            y2 = int(self.trajectories[obj_id][i][1] * self.scale_factor + self.offset.y())
                            painter.drawLine(x1, y1, x2, y2)  # Отрисовка линии
                        
                        # Рисуем последний bounding box и ID
                        if len(self.trajectories[obj_id]) > 0:
                            last_point = self.trajectories[obj_id][-1]  # Последняя точка
                            box_size = 20  # Размер бокса
                            x = int(last_point[0] * self.scale_factor + self.offset.x())  # X координата
                            y = int(last_point[1] * self.scale_factor + self.offset.y())  # Y координата
                            
                            # Показываем информацию в зависимости от режима
                            if self.display_mode == "all" or self.object_in_trigger.get(obj_id, False):
                                # Рисуем прямоугольник
                                painter.setPen(QPen(color, 2))  # Перо
                                painter.setBrush(Qt.NoBrush)  # Без заливки
                                painter.drawRect(x-box_size//2, y-box_size//2, box_size, box_size)  # Отрисовка
                                
                                # Подписываем ID и класс объекта
                                if obj_id in self.active_objects:
                                    font = QFont()  # Шрифт
                                    font.setPointSize(10)  # Размер
                                    painter.setFont(font)  # Установка шрифта
                                    painter.setPen(QPen(QColor(255, 255, 255), 1))  # Белое перо
                                    
                                    obj_info = self.active_objects[obj_id]  # Информация об объекте
                                    class_name = self.class_names.get(obj_info['class_id'], "object")  # Имя класса
                                    time_str = time.strftime("%H:%M:%S", time.gmtime(obj_info['start_time']))  # Время
                                    
                                    # Отрисовка текста
                                    painter.drawText(x+box_size//2+5, y, f"ID: {int(obj_id)} {class_name}")
                                    painter.drawText(x+box_size//2+5, y+15, time_str)
            
            # Рисуем информацию о режиме
            font = QFont()  # Шрифт
            font.setPointSize(12)  # Размер
            painter.setFont(font)  # Установка шрифта
            
            # Определение текущего режима
            mode = "PAUSED" if self.paused else "ZONE DEFINITION" if self.define_zones_mode else "TRACKING"
            painter.setPen(QColor(255, 255, 255))  # Белое перо
            painter.drawText(10, 30, f"MODE: {mode}")  # Отрисовка текста
            
            if self.define_zones_mode or self.paused:  # Если режим зон или пауза
                zone_type = "Trigger" if self.current_zone_type == "trigger" else "Delete"  # Тип зоны
                painter.drawText(10, 60, f"Zone type: {zone_type}")  # Отрисовка текста
            
            if not self.define_zones_mode:  # Если трекинг активен
                # Отрисовка количества активных объектов
                painter.drawText(final_pixmap.width() - 200, 30, f"Active objects: {len(self.active_objects)}")
                current_time = time.time() - self.start_time  # Текущее время
                time_str = time.strftime("%H:%M:%S", time.gmtime(current_time))  # Форматирование времени
                painter.drawText(final_pixmap.width() - 150, 60, time_str)  # Отрисовка времени
        
        finally:
            painter.end()  # Завершение рисования
        
        self.video_label.setPixmap(final_pixmap)  # Установка изображения
    
    def mousePressEvent(self, event):
        # Обработка нажатия мыши
        if (self.define_zones_mode or self.paused) and event.button() == Qt.LeftButton:  # Если левая кнопка
            frame_x, frame_y = self.map_point_to_frame(event.pos())  # Преобразование координат
            self.current_zone.append((frame_x, frame_y))  # Добавление точки
            self.drawing = True  # Установка флага рисования
            self.update_frame_display()  # Обновление отображения
    
    def mouseReleaseEvent(self, event):
        # Обработка отпускания мыши
        if (self.define_zones_mode or self.paused) and event.button() == Qt.RightButton and self.drawing:  # Если правая кнопка
            if len(self.current_zone) > 2:  # Если точек достаточно
                self.zones.append((self.current_zone_type, self.current_zone.copy()))  # Добавление зоны
            self.current_zone = []  # Сброс текущей зоны
            self.drawing = False  # Сброс флага рисования
            self.update_frame_display()  # Обновление отображения
    
    def keyPressEvent(self, event):
        # Обработка нажатия клавиш
        if event.key() == Qt.Key_C:  # Клавиша C - переключение типа зоны
            self.current_zone_type = "delete" if self.current_zone_type == "trigger" else "trigger"
            self.zone_type_combo.setCurrentIndex(1 if self.current_zone_type == "delete" else 0)
            self.update_frame_display()
        elif event.key() == Qt.Key_Space:  # Пробел - переключение воспроизведения
            self.toggle_play()
        elif event.key() == Qt.Key_Return:  # Enter - старт трекинга
            self.start_tracking()
        elif event.key() == Qt.Key_D:  # D - удаление всех зон
            self.delete_zones()
        elif event.key() == Qt.Key_T:  # T - удаление триггер-зон
            self.delete_zones("trigger")
        elif event.key() == Qt.Key_E:  # E - удаление зон удаления
            self.delete_zones("delete")
        elif event.key() == Qt.Key_Q:  # Q - выход
            self.close()
    
    def closeEvent(self, event):
        # Обработка закрытия окна
        if self.cap is not None:  # Если источник открыт
            self.cap.release()  # Освобождение ресурсов
        if self.timer.isActive():  # Если таймер активен
            self.timer.stop()  # Остановка таймера
        if hasattr(self, 'model'):  # Если модель загружена
            del self.model  # Удаление модели
        event.accept()  # Подтверждение закрытия

if __name__ == "__main__":
    app = QApplication([])  # Создание приложения
    player = VideoPlayer()  # Создание плеера
    player.show()  # Показ окна
    app.exec_()  # Запуск цикла событий