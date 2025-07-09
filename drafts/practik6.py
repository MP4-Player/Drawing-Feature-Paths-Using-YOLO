import os
import cv2
import random
import time
from ultralytics import YOLO
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class TrajectoryTracker:
    def __init__(self):
        self.drawing = False
        self.current_zone = []
        self.zones = []
        self.current_zone_type = "trigger"
        self.define_zones_mode = True
        self.window_name = 'YOLOv8 Tracking and Zone Drawing'
        self.model = YOLO('yolov8m.pt')
        self.trajectories = {}
        self.object_colors = {}
        self.active_objects = {}
        self.object_in_trigger = {}
        self.paused = False
        self.current_frame = None
        self.zone_colors = {
            "trigger": (0, 255, 0),
            "delete": (0, 0, 255)
        }
    
    def draw_zones(self, frame):
        for zone in self.zones:
            zone_type, points = zone
            color = self.zone_colors[zone_type]
            if len(points) > 1:
                cv2.polylines(frame, [np.array(points)], True, color, 2)
            for point in points:
                cv2.circle(frame, point, 5, color, -1)
    
    def is_inside_zone(self, point, zone):
        zone_type, points = zone
        if len(points) < 3:
            return False
        return cv2.pointPolygonTest(np.array(points), point, False) >= 0
    
    def mouse_callback(self, event, x, y, flags, param):
        if not self.define_zones_mode:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_zone.append((x, y))
            self.drawing = True
        elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
            if len(self.current_zone) > 2:
                self.zones.append((self.current_zone_type, self.current_zone.copy()))
            self.current_zone = []
            self.drawing = False
    
    def select_source(self):
        print("Select video source:")
        print("1 - Webcam")
        print("2 - Video file")
        choice = input("Enter number (1/2): ").strip()
        
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
                print("No cameras found!")
                return None
            
            print(f"Available cameras: {available_cameras}")
            try:
                cam_idx = int(input(f"Select camera (default {available_cameras[0]}): ") or available_cameras[0])
                return cam_idx
            except:
                return available_cameras[0]
        
        elif choice == "2":
            video_files = [f for f in os.listdir() if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if not video_files:
                print("No video files in current directory!")
                return None
            
            print("Available video files:")
            for i, f in enumerate(video_files, 1):
                print(f"{i} - {f}")
            
            try:
                vid_idx = int(input("Select video file (number): ")) - 1
                return video_files[vid_idx]
            except:
                return video_files[0]
        
        return 0
    
    def delete_zones(self, zone_type=None):
        if zone_type is None:
            self.zones = []
        else:
            self.zones = [zone for zone in self.zones if zone[0] != zone_type]
    
    def run(self):
        video_source = self.select_source()
        if video_source is None:
            return
        
        cap = cv2.VideoCapture(video_source if isinstance(video_source, int) else video_source, 
                              cv2.CAP_DSHOW if isinstance(video_source, int) else 0)
        
        if not cap.isOpened():
            print("Error opening video source!")
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
        
        start_time = time.time()
        self.current_frame = None
        
        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                self.current_frame = frame.copy()
            
            if self.current_frame is None:
                continue
                
            display_frame = self.current_frame.copy()
            current_time = time.time() - start_time
            time_str = time.strftime("%H:%M:%S", time.gmtime(current_time))
            
            if self.define_zones_mode:
                if self.current_zone:
                    if len(self.current_zone) > 1:
                        cv2.polylines(display_frame, [np.array(self.current_zone)], False, 
                                     self.zone_colors[self.current_zone_type], 2)
                    for point in self.current_zone:
                        cv2.circle(display_frame, point, 5, self.zone_colors[self.current_zone_type], -1)
                
                self.draw_zones(display_frame)
                
                cv2.putText(display_frame, "ZONE DEFINITION MODE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Zone type: {self.current_zone_type}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "LMB: add point", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "RMB: finish zone (min 3 points)", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "SPACE: change zone type", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "ENTER: start tracking", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "D: delete all zones", (10, 210), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "T: delete trigger zones", (10, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "DEL: delete delete zones", (10, 270), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                if not self.paused:
                    results = self.model.track(display_frame, persist=True, tracker="botsort.yaml")
                    
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
                                self.object_in_trigger[obj_id] = False
                            
                            color = self.object_colors[obj_id]
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_frame, f'ID: {int(obj_id)}', (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            in_trigger = any(
                                self.is_inside_zone(center, ("trigger", zone)) 
                                for zone_type, zone in self.zones 
                                if zone_type == "trigger"
                            )
                            
                            if in_trigger and not self.object_in_trigger.get(obj_id, False):
                                self.object_in_trigger[obj_id] = True
                                self.trajectories[obj_id] = []
                                print(f"Object {int(obj_id)} entered trigger zone at {time_str}")
                                self.active_objects[obj_id] = time_str
                            
                            in_delete = any(
                                self.is_inside_zone(center, ("delete", zone)) 
                                for zone_type, zone in self.zones 
                                if zone_type == "delete"
                            )
                            
                            if in_delete and obj_id in self.trajectories:
                                del self.trajectories[obj_id]
                                if obj_id in self.active_objects:
                                    print(f"Object {int(obj_id)} entered delete zone at {time_str}")
                                    del self.active_objects[obj_id]
                                self.object_in_trigger[obj_id] = False
                            
                            if self.object_in_trigger.get(obj_id, False):
                                if obj_id not in self.trajectories:
                                    self.trajectories[obj_id] = []
                                self.trajectories[obj_id].append(center)
                            
                            if obj_id in self.trajectories and len(self.trajectories[obj_id]) > 1:
                                for i in range(1, len(self.trajectories[obj_id])):
                                    cv2.line(display_frame, self.trajectories[obj_id][i-1], 
                                            self.trajectories[obj_id][i], color, 2)
                
                self.draw_zones(display_frame)
                cv2.putText(display_frame, f"Active objects: {len(self.active_objects)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, time_str, (display_frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "PAUSED" if self.paused else "LIVE", (display_frame.shape[1] - 100, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if self.paused else (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if self.define_zones_mode:
                    self.current_zone_type = "delete" if self.current_zone_type == "trigger" else "trigger"
                else:
                    self.paused = not self.paused
            elif key == 13:  # Enter
                if self.define_zones_mode:
                    if len([z for z in self.zones if z[0] == "trigger"]) > 0:
                        self.define_zones_mode = False
                        print("Tracking started!")
                    else:
                        print("Error: No trigger zones defined!")
            elif key == ord('d'):
                self.delete_zones()
                print("All zones deleted")
            elif key == ord('t'):
                self.delete_zones("trigger")
                print("Trigger zones deleted")
            elif key == ord('e'):  # Using 'e' for delete zones (since 'del' key is hard to capture)
                self.delete_zones("delete")
                print("Delete zones deleted")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = TrajectoryTracker()
    tracker.run()