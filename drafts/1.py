from ultralytics import YOLO
import cv2
import numpy as np
import lap

# Инициализация YOLOv8 с моделью для трекинга
model = YOLO('yolov8n.pt')  # Можно использовать 'yolov8n.pt', 'yolov8s.pt' и т.д.

# Глобальные переменные
trigger_zones = []  # Триггерная зона для отрисовки траектории
delete_zones = []   # Триггерная зона для удаления траектории
trajectories = {}   # Словарь для хранения траекторий объектов

# Функция для обработки кликов мыши
def mouse_callback(event, x, y, flags, param):
    global trigger_zones, delete_zones
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(trigger_zones) < 2:
            trigger_zones.append((x, y))
        elif len(delete_zones) < 2:
            delete_zones.append((x, y))

# Инициализация видео
video_path = 'video_2025-01-19_21-33-09.mp4'
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('YOLOv8 Tracking')
cv2.setMouseCallback('YOLOv8 Tracking', mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Отрисовка триггерных зон
    if len(trigger_zones) == 2:
        cv2.rectangle(frame, trigger_zones[0], trigger_zones[1], (0, 255, 0), 2)
    if len(delete_zones) == 2:
        cv2.rectangle(frame, delete_zones[0], delete_zones[1], (0, 0, 255), 2)

    # Запуск трекинга с помощью YOLOv8
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")  # Используем трекер ByteTrack

    # Визуализация результатов
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Координаты bounding box'ов
        ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []  # ID объектов
        clss = result.boxes.cls.cpu().numpy()  # Классы объектов

        for box, obj_id, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  # Центр bounding box'а

            # Отрисовка bounding box'а и ID
            color = (int(obj_id * 50) % 255, int(obj_id * 100) % 255, int(obj_id * 150) % 255)  # Уникальный цвет
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID: {int(obj_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Проверка попадания в триггерную зону
            if len(trigger_zones) == 2:
                if (trigger_zones[0][0] < center_x < trigger_zones[1][0] and
                    trigger_zones[0][1] < center_y < trigger_zones[1][1]):
                    if obj_id not in trajectories:
                        trajectories[obj_id] = []
                    trajectories[obj_id].append((center_x, center_y))

            # Отрисовка траектории
            if obj_id in trajectories:
                for i in range(1, len(trajectories[obj_id])):
                    cv2.line(frame, trajectories[obj_id][i - 1], trajectories[obj_id][i], color, 2)

            # Проверка попадания в зону удаления
            if len(delete_zones) == 2:
                if (delete_zones[0][0] < center_x < delete_zones[1][0] and
                    delete_zones[0][1] < center_y < delete_zones[1][1]):
                    if obj_id in trajectories:
                        del trajectories[obj_id]

    # Отображение кадра
    cv2.imshow('YOLOv8 Tracking', frame)

    # Обработка нажатия клавиши 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()