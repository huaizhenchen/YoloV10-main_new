from ultralytics import YOLOv10
import glob
import os
import time
import cv2


classes = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class Colors:
    """Ultralytics color palette https://ultralytics.com/."""
 
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        # print(self.palette)
        self.n = len(self.palette)
 
    def __call__(self, i, bgr=False):
        """Converts hex color codes to rgb values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c
 
    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


if __name__ == "__main__":

    # 设置颜色
    colors = Colors()  # create instance for 'from utils.plots import colors'

    # 设置模式
    mode = "video"    # mode: "image", "video"

    # 设置AI模型
    model_path = '/projectnb/cislbu/huaizhen/TRY3/COIL/YoloV10-main/runs/detect/train3/weights/best.pt'
    # model_path = r'.\model\yolov10x.onnx'
    # model_path = r'.\model\yolov10x.engine'
    # model_path = r'.\model\yolov10x_openvino_model'
    model = YOLOv10(model_path)

    # 设置保存路径
    save_dir  = r'.\runs\predict'
    os.makedirs(save_dir,exist_ok=True)

    if mode == "video":
        # 设置视频文件
        video_path = r'.\ultralytics\assets\test.mp4'
        video_name = video_path.split('\\')[-1]
        cap = cv2.VideoCapture(video_path)       # 创建VideoCapture对象
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # 获取视频的编码格式
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(os.path.join(save_dir, video_name), fourcc, video_fps, video_size) # 创建VideoWriter对象，用于保存视频

        while True:
            # 读取视频帧
            ret, frame = cap.read()
            if not ret:
                break

            # 计时
            start = time.time()

            # 推理
            results = model.predict(frame)[0]

            # 后处理
            for box in results.boxes:
                # print(box)
                xyxy = box.xyxy.squeeze().tolist()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                c, conf = int(box.cls), float(box.conf)
                name = classes[c]
                color = colors(c, True)
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(frame, f"{name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            
            # 计时
            end = time.time()

            # 计算FPS
            if start != end:
                FPS = int(1 / (end - start))

            cv2.putText(frame, f"FPS: {FPS}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 写入帧到输出视频文件
            out.write(frame)

        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "image":
        # 设置图像文件
        img_path = r'.\ultralytics\assets'
        imgs = glob.glob(os.path.join(img_path,'*.jpg'))

        for img in imgs:
            imgname = img.split('\\')[-1]

            # 读取图像数据
            frame = cv2.imread(img)

            # 推理
            # results = model.predict(frame)[0]
            results = model(frame)[0]
        
            # 后处理
            for box in results.boxes:
                xyxy = box.xyxy.squeeze().tolist()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                c, conf = int(box.cls), float(box.conf)
                name = classes[c]
                color = colors(c, True)
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(frame, f"{name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # 保存
            cv2.imwrite(os.path.join(save_dir,imgname),frame)
