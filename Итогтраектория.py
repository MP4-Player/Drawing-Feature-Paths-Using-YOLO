from ultralytics import YOLO
import cv2
import random  

# К БОЛЬШОМУ СОЖЕЛЕНИЮ Я УБИЛ ЯДРО ЮПИТЕРА Я ХЗ ЧТО ДЕЛАТЬ ПОЭТОМУ Я ВСЁ ПЕРЕНЕС СЮДА по хорошему это всё должно быть в ячейках но и так запускаеться


model = YOLO('yolov8x.pt')
trajectories = {}
object_colors = {}

# Триггерные зоны
trigger_zones = [(100, 300, 300, 800)]  # (x1, y1, x2, y2)
delete_zones = [(500, 50, 600, 700)]   # (x1, y1, x2, y2)

# Функция для проверки пересечения двух прямоугольников
def is_intersecting(box1, box2):
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2
    if x2_box1 < x1_box2:
        return False  
    if x2_box2 < x1_box1:
        return False  
    if y2_box1 < y1_box2:
        return False 
    if y2_box2 < y1_box1:
        return False 
    return True




#video_path = 'video_2025-01-19_21-33-09.mp4'
cap = cv2.VideoCapture(0)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model.track(frame, persist=True, tracker="botsort.yaml")#"bytetrack.yaml")

    # Визуализация результатов
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # координаты мистера  бокса
        ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []  # ID объектов
        clss = result.boxes.cls.cpu().numpy() 

        for box, obj_id, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2 #рисуем от центра 

            # эдем , рисуй!
            if obj_id not in object_colors:
                red = random.randint(0, 255)  
                green = random.randint(0, 255) 
                blue = random.randint(0, 255) 
                color = (red, green, blue)
                object_colors[obj_id] = color
            else:
                color = object_colors[obj_id]


            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f'ID: {int(obj_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

          
            for zone in trigger_zones:
                if is_intersecting((x1, y1, x2, y2), zone):
                    if obj_id not in trajectories:
                        trajectories[obj_id] = []
                    trajectories[obj_id].append((center_x, center_y))

            
            if obj_id in trajectories:
                for i in range(1, len(trajectories[obj_id])):
                    cv2.line(frame, trajectories[obj_id][i - 1], trajectories[obj_id][i], color, 2)

            # Проверка попадания
            for zone in delete_zones:
                if is_intersecting((x1, y1, x2, y2), zone):
                    if obj_id in trajectories:
                        del trajectories[obj_id]

    # Отрисовка триггерав
    for zone in trigger_zones:
        cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 255, 0), 2)  # Зелёный прямоугольник
    for zone in delete_zones:
        cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 0, 255), 2)  # Красный прямоугольник

    cv2.imshow('YOLOv8 Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()