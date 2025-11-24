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

def process_video(video_path, mask_path, video_label, result_text, control_vars, root=None):
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
            # ✅ Giảm input size để tăng tốc độ nếu cần
            tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)

            # ✅ Không dùng mask nữa - detection trực tiếp từ video để tọa độ chính xác
            # Mask chỉ làm detection sai sót vì tọa độ không khớp
            
            # ✅ Tạo smooth tracker cho mỗi object
            smooth_trackers = {}

            vehicle_count = {"car": 0, "motorbike": 0,  "truck": 0, "bus": 0}
            counted_ids = {k: set() for k in vehicle_count}
            counted_objects = set()  # ✅ Chỉ 1 set duy nhất - mỗi object đếm 1 lần
            
            # ✅ Lưu vị trí trước của mỗi object để detect direction
            track_prev_positions = {}

            # ✅ Điều chỉnh line dọc theo video (object chạy trái-phải, không từ trên-xuống)
            # Format: x_line dọc ở vị trí X
            line_x = int(frame_width * 0.25)  # ✅ Line dọc ở 25% width (bên trái-giữa)

            frame_num = 0
            debug_mode = False  # ✅ Tắt debug mode
            
            # ✅ Biến điều khiển phát video
            is_paused = control_vars["paused"]
            playback_speed = control_vars["speed"]
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # ✅ Bộ đệm frame để giảm flickering
            frame_buffer = None
            buffer_time = time.time()

            while True:
                success, img = cap.read()
                if not success:
                    print(f"Video ended or failed at frame: {frame_num}")
                    break

                frame_num += 1
                print(f"Processing frame: {frame_num}")
                
                # ✅ Pause handling
                while control_vars["paused"]:
                    time.sleep(0.1)  # Chờ khi pause
                
                # ✅ Lấy tốc độ hiện tại
                playback_speed = control_vars["speed"]

                # ✅ Detection trực tiếp từ frame gốc (không mask)
                results = model(img, stream=False)
                
                # ✅ Frame để hiển thị
                img_display = img.copy()

                detections = np.empty((0, 5))
                boxes_info = []  

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = classNames[cls_id]

                        if cls_name in vehicle_count and conf > 0.15:  # ✅ Giảm xuống 0.15 để bắt object sót
                            detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
                            boxes_info.append([x1, y1, x2, y2, cls_name])

                            cvzone.cornerRect(img_display, (x1, y1, x2 - x1, y2 - y1))
                            cv2.putText(img_display, cls_name, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ✅ Update tracker với detections
                track_results = tracker.update(detections)

                # ✅ Không skip frame - luôn render để tránh chớp nháy
                # Nếu không có tracking, vẫn cập nhật UI với frame hiện tại

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
                    cv2.rectangle(img_display, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
                    cv2.circle(img_display, (cx, cy), 5, (0, 0, 255), -1)
                    
                    class_label = f"{matched_class}" if matched_class else "unknown"
                    cv2.putText(img_display, f"ID {track_id} {class_label}", (draw_x1, draw_y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # ✅ Counting logic - chỉ khi có class
                    if matched_class:
                        # Debug: In vị trí để kiểm tra
                        if debug_mode and frame_num % 5 == 0:
                            print(f"Frame {frame_num}: ID {track_id} ({matched_class}), cx={cx}, cy={cy}")
                            if track_id in track_prev_positions:
                                prev_cx = track_prev_positions[track_id]["cx"]
                                print(f"  prev_cx={prev_cx}, Line X: x={line_x}")
                            else:
                                print(f"  FIRST FRAME - No prev position yet")
                        
                        # ✅ Chỉ đếm 1 lần khi vượt qua line dọc (X-axis crossing)
                        if track_id in track_prev_positions:
                            prev_cx = track_prev_positions[track_id]["cx"]
                            
                            # ✅ Vượt qua line dọc (từ trái sang phải hoặc phải sang trái)
                            crossed_line = (prev_cx < line_x and cx >= line_x) or (prev_cx > line_x and cx <= line_x)
                            
                            if debug_mode and frame_num % 5 == 0:
                                print(f"  Crossed line: {crossed_line}, prev_cx={prev_cx}, cx={cx}")
                            
                            if crossed_line and track_id not in counted_objects:
                                counted_objects.add(track_id)
                                vehicle_count[matched_class] += 1
                                direction = "→" if cx >= line_x else "←"
                                print(f"✅ Đếm {matched_class} (ID {track_id}) vượt line {direction}, prev_cx={prev_cx}, cx={cx}")
                        else:
                            # ✅ Lần đầu tiên thấy object này - lưu vị trí
                            if debug_mode and frame_num % 5 == 0:
                                print(f"  SAVING INITIAL POSITION: cx={cx}")
                        
                        # ✅ LUÔN lưu vị trí hiện tại cho lần check tiếp theo
                        track_prev_positions[track_id] = {"cx": cx, "cy": cy}
                    else:
                        if debug_mode and frame_num % 20 == 0:
                            print(f"⚠️ Frame {frame_num}: Track {track_id} không match class nào, cx={cx}, cy={cy}")                # ✅ Xóa smooth trackers của các object đã mất
                active_ids = set(int(t[4]) for t in track_results) if len(track_results) > 0 else set()
                dead_ids = [tid for tid in smooth_trackers.keys() if tid not in active_ids]
                for tid in dead_ids:
                    del smooth_trackers[tid]
                    # ✅ Xóa vị trí trước của object đã mất
                    if tid in track_prev_positions:
                        del track_prev_positions[tid]

                # ✅ Không vẽ lines trên màn hình (chỉ dùng cho logic đếm)
                # cv2.line(img_display, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 4)
                # cv2.line(img_display, (limits_2[0], limits_2[1]), (limits_2[2], limits_2[3]), (0, 255, 255), 4)

                # ✅ Hiển thị line dọc trên màn hình để kiểm tra
                cv2.line(img_display, (line_x, 0), (line_x, frame_height), (0, 0, 255), 3)

                # Update UI - ✅ Resize trước để giảm lag
                display_width = 1280
                display_height = 720
                img_resized = cv2.resize(img_display, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
                
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
                
                # ✅ Dùng root.after() để update UI thread-safe
                if root:
                    def update_ui(photo, counts):
                        try:
                            video_label.imgtk = photo
                            video_label.config(image=photo)
                            result_text.delete("1.0", "end")
                            for k, v in counts.items():
                                result_text.insert("end", f"{k}: {v}\n")
                        except:
                            pass  # Ignore errors nếu window đã đóng
                    
                    root.after(0, update_ui, imgtk, vehicle_count.copy())
                
                # ✅ Fixed timing - hạ xuống 60ms = ~16 FPS ở tốc độ bình thường
                time.sleep(0.06 / playback_speed)
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

    # ✅ Biến điều khiển video (shared giữa các function)
    control_vars = {"paused": False, "speed": 1.0}

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
    Button(root, text="Chọn mask (optional)", command=browse_mask).grid(row=1, column=2)

    video_frame = Frame(root, width=800, height=600, bg="black")
    video_frame.grid(row=2, column=0, columnspan=2, pady=20)
    video_frame.grid_propagate(False)
    video_label = Label(video_frame)
    video_label.pack(fill="both", expand=True)

    result_text = Text(root, width=40, height=20)
    result_text.grid(row=2, column=2, padx=20)

    def start_processing():
        video = video_path_entry.get("1.0", "end").strip()
        mask = mask_path_entry.get("1.0", "end").strip()

        if not video:
            messagebox.showwarning("Thiếu dữ liệu", "Chưa chọn video")
            return

        # ✅ Mask là optional - có thể để trống
        process_video(video, mask if mask else None, video_label, result_text, control_vars, root)

    # ✅ Nút điều khiển video
    control_frame = Frame(root)
    control_frame.grid(row=3, column=0, columnspan=3, pady=10)
    
    pause_button = Button(control_frame, text="⏸ Pause", width=10)
    pause_button.pack(side="left", padx=5)
    
    play_button = Button(control_frame, text="▶ Play", width=10)
    play_button.pack(side="left", padx=5)
    
    rewind_button = Button(control_frame, text="⏮ Tua Ngược", width=10)
    rewind_button.pack(side="left", padx=5)
    
    forward_button = Button(control_frame, text="⏭ Tua Tới", width=10)
    forward_button.pack(side="left", padx=5)
    
    speed_half = Button(control_frame, text="0.5x", width=8)
    speed_half.pack(side="left", padx=5)
    
    speed_normal = Button(control_frame, text="1.0x", width=8)
    speed_normal.pack(side="left", padx=5)
    
    speed_double = Button(control_frame, text="2.0x", width=8)
    speed_double.pack(side="left", padx=5)
    
    def on_pause():
        control_vars["paused"] = True
        pause_button.config(state="disabled", bg="gray")
        play_button.config(state="normal", bg="SystemButtonFace")
    
    def on_play():
        control_vars["paused"] = False
        play_button.config(state="disabled", bg="green")
        pause_button.config(state="normal", bg="SystemButtonFace")
    
    def set_speed(speed):
        control_vars["speed"] = speed
        speed_half.config(state="normal", bg="SystemButtonFace")
        speed_normal.config(state="normal", bg="SystemButtonFace")
        speed_double.config(state="normal", bg="SystemButtonFace")
        
        if speed == 0.5:
            speed_half.config(state="disabled", bg="lightblue")
        elif speed == 1.0:
            speed_normal.config(state="disabled", bg="lightblue")
        elif speed == 2.0:
            speed_double.config(state="disabled", bg="lightblue")
    
    pause_button.config(command=on_pause)
    play_button.config(command=on_play, state="disabled", bg="green")
    speed_half.config(command=lambda: set_speed(0.5))
    speed_normal.config(command=lambda: set_speed(1.0), state="disabled", bg="lightblue")
    speed_double.config(command=lambda: set_speed(2.0))
    
    # Tua tới/tua ngược (placeholder - cần refactor video capture để hoạt động)
    rewind_button.config(state="disabled")
    forward_button.config(state="disabled")

    start_button = Button(root, text="Bắt đầu xử lý", command=start_processing)
    start_button.grid(row=4, column=1, pady=20)

    root.mainloop()

start_interface()