from ultralytics import YOLO
import cv2
import random

model = YOLO('yolov8x.pt')
trajectories = {}
object_colors = {}


trigger_zones = [(100, 300, 300, 800)]  # (x1, y1, x2, y2)
delete_zones = [(500, 50, 600, 700)]    # (x1, y1, x2, y2)


def is_intersecting(box1, box2):
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

   
    if x2_box1 < x1_box2 or x2_box2 < x1_box1:
        return False

    if y2_box1 < y1_box2 or y2_box2 < y1_box1:
        return False

    return True




#video_path = 'video_2025-01-19_21-33-09.mp4'
cap = cv2.VideoCapture(0)





while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break



    results = model.track(frame, persist=True, tracker="botsort.yaml")#"bytetrack.yaml")

 
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Координаты boxов
        ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []
        classes = result.boxes.cls.cpu().numpy()  #

        # Перебор всех объектов
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            obj_id = int(ids[i]) if i < len(ids) else -1  
            cls = classes[i] if i < len(classes) else -1  

            # Центр
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Генерация цвета для объекта
            if obj_id not in object_colors:
                red = random.randint(0, 255)
                green = random.randint(0, 255)
                blue = random.randint(0, 255)
                color = (red, green, blue)
                object_colors[obj_id] = color
            else:
                color = object_colors[obj_id]

            # Рисование box и ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            #попадание
            for zone in trigger_zones:
                if is_intersecting((x1, y1, x2, y2), zone):
                    if obj_id not in trajectories:
                        trajectories[obj_id] = []
                    trajectories[obj_id].append((center_x, center_y))

            # Рисование траектории
            if obj_id in trajectories:
                for j in range(1, len(trajectories[obj_id])):
                    cv2.line(frame, trajectories[obj_id][j - 1], trajectories[obj_id][j], color, 2)


            for zone in delete_zones:
                if is_intersecting((x1, y1, x2, y2), zone):
                    if obj_id in trajectories:
                        del trajectories[obj_id]

    # Отрисовка триггерных зон
    for zone in trigger_zones:
        cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 255, 0), 2)  # Зелёный 
    for zone in delete_zones:
        cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 0, 255), 2)  # Красный

    cv2.imshow('YOLOv8 Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()