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
        self.paused = True  # Start in paused state
        self.current_frame = None
        self.cap = None
        self.zone_colors = {
            "trigger": (0, 255, 0),
            "delete": (0, 0, 255)
        }
    
    def init_video_capture(self, video_source):
        if isinstance(video_source, int):
            self.cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            print(f"Error opening video source: {video_source}")
            return False
        
        # Read first frame to initialize
        ret, frame = self.cap.read()
        if not ret:
            print("Error reading first frame!")
            self.cap.release()
            return False
        
        self.current_frame = frame.copy()
        return True
    
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
                if 0 <= vid_idx < len(video_files):
                    return video_files[vid_idx]
                return video_files[0]
            except:
                return video_files[0]
        
        return 0
    
    def delete_zones(self, zone_type=None):
        if zone_type is None:
            self.zones = []
        else:
            self.zones = [zone for zone in self.zones if zone[0] != zone_type]
    
    def show_instructions(self, frame):
        y = 30
        mode = "PAUSED" if self.paused else "ZONE DEFINITION" if self.define_zones_mode else "TRACKING"
        cv2.putText(frame, f"MODE: {mode}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 40
        
        if self.define_zones_mode or self.paused:
            cv2.putText(frame, f"Zone type: {self.current_zone_type}", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "LMB: add point", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "RMB: finish zone (min 3 points)", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "C: change zone type", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "SPACE: play/pause", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "ENTER: start tracking", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "D: delete all zones", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "T: delete trigger zones", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "E: delete delete zones", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 30
            
            cv2.putText(frame, "Q: quit", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_frame(self):
        if not self.paused and not self.define_zones_mode and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.video_source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return
                else:
                    self.paused = True
                    return
            self.current_frame = frame.copy()
    
    def update_display(self):
        if self.current_frame is None:
            return
            
        display_frame = self.current_frame.copy()
        current_time = time.time() - self.start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(current_time))
        
        # Draw trajectories
        for obj_id in self.trajectories:
            if len(self.trajectories[obj_id]) > 1:
                color = self.object_colors.get(obj_id, (0, 0, 255))
                for i in range(1, len(self.trajectories[obj_id])):
                    cv2.line(display_frame, self.trajectories[obj_id][i-1], 
                            self.trajectories[obj_id][i], color, 2)
        
        # Draw current zone being edited
        if (self.define_zones_mode or self.paused) and self.current_zone:
            if len(self.current_zone) > 1:
                cv2.polylines(display_frame, [np.array(self.current_zone)], False, 
                             self.zone_colors[self.current_zone_type], 2)
            for point in self.current_zone:
                cv2.circle(display_frame, point, 5, self.zone_colors[self.current_zone_type], -1)
        
        # Draw all zones
        self.draw_zones(display_frame)
        
        # Show instructions
        self.show_instructions(display_frame)
        
        # Tracking info
        if not self.define_zones_mode:
            cv2.putText(display_frame, f"Active objects: {len(self.active_objects)}", 
                       (display_frame.shape[1] - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, time_str, 
                       (display_frame.shape[1] - 150, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Object detection when not paused
        if not self.paused and not self.define_zones_mode:
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
                    
                    # Check trigger zones
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
                    
                    # Check delete zones
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
        
        cv2.imshow(self.window_name, display_frame)
    
    def run(self):
        self.video_source = self.select_source()
        if self.video_source is None:
            return
        
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
            if key == ord(' '):  # Space to pause/resume
                self.paused = not self.paused
            elif key == ord('c'):  # Change zone type
                self.current_zone_type = "delete" if self.current_zone_type == "trigger" else "trigger"
            elif key == 13:  # Enter to start tracking
                if self.define_zones_mode:
                    if len([z for z in self.zones if z[0] == "trigger"]) > 0:
                        self.define_zones_mode = False
                        self.paused = False
                        print("Tracking started!")
                    else:
                        print("Error: No trigger zones defined!")
            elif key == ord('d'):  # Delete all zones
                self.delete_zones()
                print("All zones deleted")
            elif key == ord('t'):  # Delete trigger zones
                self.delete_zones("trigger")
                print("Trigger zones deleted")
            elif key == ord('e'):  # Delete delete zones
                self.delete_zones("delete")
                print("Delete zones deleted")
            elif key == ord('q'):  # Quit
                break
        
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = TrajectoryTracker()
    tracker.run()