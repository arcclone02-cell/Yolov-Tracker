import sys
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone 
from tkinter import Tk, Label, Button, Text, filedialog, messagebox, Frame
from PIL import Image, ImageTk
import math
import threading
import time
from sort import Sort
from collections import deque

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl","banana", "apple",
              "sandwich", "orange", "broccoli", "carrot","hot dog", "pizza", "donut", "cake","chair",
              "sofa", "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote",
              "keyboard","cell phone", "microwave","oven","toaster","sink","refrigerator","book","clock",
              "vase","scissors","teddy bear","hair drier","toothbrush"]

class SmoothTracker:
    """Lưu trữ lịch sử track để smooth hóa"""
    def __init__(self, history_size=3):
        self.history_size = history_size
        self.positions = deque(maxlen=history_size)
    
    def add_position(self, x1, y1, x2, y2):
        self.positions.append([x1, y1, x2, y2])
    
    def get_smooth_position(self):
        """Tính trung bình vị trí để smooth"""
        if not self.positions:
            return None
        positions_array = np.array(list(self.positions))
        smooth_pos = np.mean(positions_array, axis=0)
        return tuple(map(int, smooth_pos))

def process_video(video_path, mask_path, video_label, result_text):
    def run():
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("Lỗi", "Không mở được video!")
                return

            # Lấy thông tin video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            model = YOLO("yolo11n.pt")
            tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)  # ✅ Giảm min_hits từ 3 -> 1

            # ✅ Đọc và resize mask một lần
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                messagebox.showerror("Lỗi", "Không đọc được mask!")
                return
            
            mask_rs = cv2.resize(mask, (frame_width, frame_height))
            mask_rs = cv2.cvtColor(mask_rs, cv2.COLOR_GRAY2BGR)
            
            # ✅ Tạo smooth tracker cho mỗi object
            smooth_trackers = {}

            vehicle_count = {"car": 0, "motorbike": 0, "bicycle": 0, "truck": 0}
            counted_ids = {k: set() for k in vehicle_count}
            counted_at_line_1 = set()
            counted_at_line_2 = set()

            # ✅ Điều chỉnh lines theo video của bạn (thường ở giữa frame)
            # Format: [x1, y1, x2, y2]
            limits = [0, int(frame_height * 0.5), frame_width, int(frame_height * 0.5)]  # Line 1 ở 50% height
            limits_2 = [0, int(frame_height * 0.55), frame_width, int(frame_height * 0.55)]  # Line 2 ở 55% height

            frame_num = 0
            debug_mode = True  # ✅ Bật debug để in chi tiết

            while True:
                success, img = cap.read()
                if not success:
                    print(f"Video ended or failed at frame: {frame_num}")
                    break

                frame_num += 1
                print(f"Processing frame: {frame_num}")

                # ✅ Áp dụng mask chỉ một lần (đã resize)
                img_region = cv2.bitwise_and(img, mask_rs)
                results = model(img_region, stream=False)

                detections = np.empty((0, 5))
                boxes_info = []  

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = classNames[cls_id]

                        if cls_name in vehicle_count and conf > 0.3:
                            detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
                            boxes_info.append([x1, y1, x2, y2, cls_name])

                            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))
                            cv2.putText(img, cls_name, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ✅ Update tracker với detections
                track_results = tracker.update(detections)

                # ✅ Tạo mapping từ track_id -> class_name từ boxes_info
                track_to_class = {}
                for b in boxes_info:
                    b_x1, b_y1, b_x2, b_y2, b_class = b
                    b_cx = (b_x1 + b_x2) // 2
                    b_cy = (b_y1 + b_y2) // 2
                    
                    # Tìm track nào match với detection này
                    for track in track_results:
                        t_x1, t_y1, t_x2, t_y2, t_id = map(int, track)
                        t_cx = (t_x1 + t_x2) // 2
                        t_cy = (t_y1 + t_y2) // 2
                        
                        # Match dựa trên IoU hoặc center distance
                        if abs(b_cx - t_cx) < 100 and abs(b_cy - t_cy) < 100:
                            track_to_class[int(t_id)] = b_class
                            break

                # ✅ Xử lý tracked objects
                for track in track_results:
                    x1, y1, x2, y2, track_id = map(int, track)
                    track_id = int(track_id)
                    
                    # Thêm vào smooth tracker (chỉ cho visualization)
                    if track_id not in smooth_trackers:
                        smooth_trackers[track_id] = SmoothTracker(history_size=3)
                    
                    smooth_trackers[track_id].add_position(x1, y1, x2, y2)
                    
                    # ✅ Lấy vị trí smooth cho vẽ hình
                    smooth_pos = smooth_trackers[track_id].get_smooth_position()
                    draw_x1, draw_y1, draw_x2, draw_y2 = smooth_pos if smooth_pos else (x1, y1, x2, y2)
                    
                    # ✅ Dùng vị trí Kalman gốc cho đếm (không smooth)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    matched_class = track_to_class.get(track_id)

                    # Vẽ tracking info (dùng smooth position)
                    cv2.rectangle(img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                    
                    class_label = f"{matched_class}" if matched_class else "unknown"
                    cv2.putText(img, f"ID {track_id} {class_label}", (draw_x1, draw_y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # ✅ Counting logic - chỉ khi có class
                    if matched_class:
                        # Debug: In vị trí để kiểm tra
                        if debug_mode and frame_num % 10 == 0:  # In mỗi 10 frame
                            print(f"Frame {frame_num}: ID {track_id} ({matched_class}), cx={cx}, cy={cy}")
                            print(f"  Line 1 range: y={limits[1]-30}~{limits[1]+30}, x={limits[0]}~{limits[2]}")
                        
                        # Line 1: Từ trái sang phải (mở rộng tolerance)
                        if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 30:
                            if track_id not in counted_at_line_1:
                                counted_at_line_1.add(track_id)
                                vehicle_count[matched_class] += 1
                                print(f"✅ Đếm {matched_class} (ID {track_id}) tại line 1, cx={cx}, cy={cy}")

                        # Line 2: Từ trái sang phải (đặc biệt cho các lane khác)
                        if limits_2[0] < cx < limits_2[2] and limits_2[1] - 30 < cy < limits_2[1] + 30:
                            if track_id not in counted_at_line_2:
                                counted_at_line_2.add(track_id)
                                vehicle_count[matched_class] += 1
                                print(f"✅ Đếm {matched_class} (ID {track_id}) tại line 2, cx={cx}, cy={cy}")
                    else:
                        if debug_mode and frame_num % 20 == 0:
                            print(f"⚠️ Frame {frame_num}: Track {track_id} không match class nào, cx={cx}, cy={cy}")

                # ✅ Xóa smooth trackers của các object đã mất
                active_ids = set(int(t[4]) for t in track_results) if len(track_results) > 0 else set()
                dead_ids = [tid for tid in smooth_trackers.keys() if tid not in active_ids]
                for tid in dead_ids:
                    del smooth_trackers[tid]

                # Vẽ lines
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 4)
                cv2.line(img, (limits_2[0], limits_2[1]), (limits_2[2], limits_2[3]), (0, 255, 255), 4)

                # Update UI
                result_text.delete("1.0", "end")
                for k, v in vehicle_count.items():
                    result_text.insert("end", f"{k}: {v}\n")

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
                video_label.imgtk = imgtk
                video_label.config(image=imgtk)

                video_label.update_idletasks()
                time.sleep(0.05)  # ✅ Tăng từ 0.01s -> 0.05s để giảm nháy

            cap.release()
            print("Released video capture")

        except Exception as e:
            print(f"Error in video processing thread: {e}")
            messagebox.showerror("Lỗi", f"Error: {e}")

    threading.Thread(target=run, daemon=True).start()

def start_interface():
    root = Tk()
    root.title("Nhận diện phương tiện giao thông")
    root.geometry("1920x1080")

    Label(root, text="Đường dẫn video:").grid(row=0, column=0, sticky="e")
    video_path_entry = Text(root, width=50, height=1)
    video_path_entry.grid(row=0, column=1)

    Label(root, text="Đường dẫn mask:").grid(row=1, column=0, sticky="e")
    mask_path_entry = Text(root, width=50, height=1)
    mask_path_entry.grid(row=1, column=1)

    def browse_video():
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi")])
        video_path_entry.delete("1.0", "end")
        video_path_entry.insert("1.0", p)

    def browse_mask():
        p = filedialog.askopenfilename(filetypes=[("Image", "*.png *.jpg")])
        mask_path_entry.delete("1.0", "end")
        mask_path_entry.insert("1.0", p)

    Button(root, text="Chọn video", command=browse_video).grid(row=0, column=2)
    Button(root, text="Chọn mask", command=browse_mask).grid(row=1, column=2)

    video_frame = Frame(root, width=1200, height=700, bg="black")
    video_frame.grid(row=2, column=0, columnspan=2, pady=20)
    video_frame.grid_propagate(False)
    video_label = Label(video_frame)
    video_label.pack(fill="both", expand=True)

    result_text = Text(root, width=40, height=20)
    result_text.grid(row=2, column=2, padx=20)

    def start_processing():
        video = video_path_entry.get("1.0", "end").strip()
        mask = mask_path_entry.get("1.0", "end").strip()

        if not video or not mask:
            messagebox.showwarning("Thiếu dữ liệu", "Chưa chọn video hoặc mask")
            return

        process_video(video, mask, video_label, result_text)

    Button(root, text="Bắt đầu xử lý", command=start_processing).grid(row=3, column=1, pady=20)

    root.mainloop()

start_interface()