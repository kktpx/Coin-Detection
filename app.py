import os
import sys
import time
import datetime
import cv2
import numpy as np
import torch
import flask
from flask import Flask, render_template, Response

# Add yolov5 to the python path so it can import its internal modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov5'))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, plot_one_box, set_logging
from utils.torch_utils import select_device, time_synchronized

app = Flask(__name__)

# --- YOLOv5 Configuration ---
imgsz = 640
my_confidence = 0.80
my_threshold = 0.45
my_weight = os.path.join(os.path.dirname(__file__), 'yolov5', 'weights', 'coin_v1-9_last.pt')

set_logging()
device = select_device('')
half = device.type != 'cpu'

print('>> Loading model on device:', device.type)
model = attempt_load(my_weight, map_location=device)
imgsz = check_img_size(imgsz, s=model.stride.max())
if half:
    model.half()

names = model.module.names if hasattr(model, 'module') else model.names
colors = [
    (232, 182, 0),  # 5Baht
    (0, 204, 255),  # 1Baht
    (69, 77, 246),  # 10Baht
    (51, 136, 222), # 2Baht
    (222, 51, 188), # .50Baht
]

def main_process(input_img):
    img0 = input_img.copy()
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_synchronized()
    pred = model(img, augment=True)[0]
    pred = non_max_suppression(pred, my_confidence, my_threshold, classes=None, agnostic=None)
    t2 = time_synchronized()

    total = 0.0
    class_count = [0 for _ in range(len(names))]
    for i, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                cls_id = int(cls)
                class_count[cls_id] += 1
                
                # Fixed bug: convert to float instead of int to handle .50 Baht
                coin_value = float(names[cls_id])
                total += coin_value
                
                label = '%s baht (%.1f%%)' % (names[cls_id], conf * 100)
                plot_one_box(xyxy, img0, label=label, color=colors[cls_id], line_thickness=3)
    
    # Overlay info
    y_offset = 45
    for i, (name, cnt) in enumerate(zip(names, class_count)):
        if cnt > 0:
            cv2.putText(img0, f"{name} Baht: {cnt} coin(s)", (10, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 0), 2)
            y_offset += 25
            
    cv2.putText(img0, f"Total: {total} Baht", (10, y_offset + 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0), 2)
    return img0

class VideoCamera(object):
    def __init__(self):
        # 0 for webcam. Update to URL if using IP Camera
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        fps_start = datetime.datetime.now()
        success, img = self.video.read()
        if not success or img is None:
            # Return a blank image if camera fails
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Camera not found", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img
            
        img = main_process(img)
        fps_end = datetime.datetime.now()
        fps_interval = 1 / (fps_end - fps_start).total_seconds()
        cv2.putText(img, f"FPS: {round(fps_interval, 2)}", (img.shape[1] - 150, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 255), 2)
        return img

def image_generator(video_feeder):
    while True:
        image = video_feeder.get_frame()
        _, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(image_generator(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
