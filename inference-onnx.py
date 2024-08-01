import glob
import os
import time
import cv2
import numpy as np
import onnxruntime as ort


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


def letterbox(
        im,
        new_shape,
        color=(114, 114, 114),
        auto=False,
        scaleFill=False,
        scaleup=True,
        stride=32,
):
    """
    Resize and pad image while meeting stride-multiple constraints
    Returns:
        im (array): (height, width, 3)
        ratio (array): [w_ratio, h_ratio]
        (dw, dh) (array): [w_padding h_padding]
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):  # [h_rect, w_rect]
        new_shape = (new_shape, new_shape)
 
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
 
    # Compute padding
    ratio = r, r  # wh ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w h
    dw, dh = (
        new_shape[1] - new_unpad[0],
        new_shape[0] - new_unpad[1],
    )  # wh padding
 
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])  # [w h]
        ratio = (
            new_shape[1] / shape[1],
            new_shape[0] / shape[0],
        )  # [w_ratio, h_ratio]
 
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)


def rescale_coords(boxes, image_shape, input_shape):
    image_height, image_width = image_shape
    input_height, input_width = input_shape
 
    scale = min(input_width / image_width, input_height / image_height)
 
    pad_w = (input_width - image_width * scale) / 2
    pad_h = (input_height - image_height * scale) / 2
 
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
 
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height)
 
    return boxes.astype(int)


def preprocess(image, input_shape):
    # Resize
    input_img = letterbox(image, input_shape)[0]
    # Transpose
    input_img = input_img[..., ::-1].transpose(2, 0, 1)
    # Expand
    input_img = input_img[np.newaxis, :, :, :].astype(np.float32)
    # Contiguous
    input_img = np.ascontiguousarray(input_img)
    # Norm
    blob = input_img / 255.0
    return blob
 
 
def postprocess(outs, conf_thres, image_shape, input_shape):
    # Filtered by conf
    outs = outs[outs[:, 4] >= conf_thres]
 
    # Extract
    boxes = outs[:, :4]
    scores = outs[:, -2]
    labels = outs[:, -1].astype(int)
 
    # Rescale
    boxes = rescale_coords(boxes, image_shape, input_shape)
 
    return boxes, scores, labels


if __name__ == "__main__":

    # 设置置信度阈值和输入尺寸
    conf_thres = 0.3
    input_shape = (640, 640)

    # 设置颜色
    colors = Colors()  # create instance for 'from utils.plots import colors'

    # 设置模式
    mode = "video"    # mode: "image", "video"

    # 设置AI模型
    model_path = r'.\model\yolov10x.onnx'
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_model = ort.InferenceSession(model_path, providers=providers)

    # 设置保存路径
    save_dir  = r'.\runs\predict'
    os.makedirs(save_dir,exist_ok=True)

    if mode == "video":
        # 设置视频文件
        video_path = r'.\ultralytics\assets\sample-2.mp4'
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

            # 前处理
            image_shape = frame.shape[:2]
            blob = preprocess(frame, input_shape)

            # 推理
            results = ort_model.run(None, {'images': blob})[0][0]

            # 后处理
            boxes, scores, labels = postprocess(results, conf_thres, image_shape, input_shape)

            # 绘图
            for label, score, box in zip(labels, scores, boxes):
                color = colors(label,True)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(frame, f'{classes[label]}: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

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

            # 前处理
            image_shape = frame.shape[:2]
            blob = preprocess(frame, input_shape)

            # 推理
            results = ort_model.run(None, {'images': blob})[0][0]
        
            # 后处理
            boxes, scores, labels = postprocess(results, conf_thres, image_shape, input_shape)

            # 绘图
            for label, score, box in zip(labels, scores, boxes):
                color = colors(label,True)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(frame, f'{classes[label]}: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            # 保存
            cv2.imwrite(os.path.join(save_dir,imgname),frame)