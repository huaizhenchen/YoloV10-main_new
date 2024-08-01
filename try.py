import cv2
import torch
from ultralytics import YOLOv10

# 加载模型
model_path = "/projectnb/cislbu/huaizhen/TRY3/COIL/YoloV10-main/runs/detect/train3/weights/best.pt"
model = YOLOv10(model_path)

# 将模型加载到 CPU 上
device = torch.device("cpu")
model.to(device)
model.eval()  # 设置为评估模式

# 打开视频文件
video_path = "/projectnb/cislbu/huaizhen/TRY3/COIL/YoloV10-main/ultralytics/assets/test.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频帧率和帧大小
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义视频写入器
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 逐帧处理视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 将帧转换为模型输入格式
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)  # 增加批次维度并移动到 CPU 上

    # 进行推理
    with torch.no_grad():
        pred = model(img)
    
    # 假设模型输出为 (boxes, scores, labels)，根据你的模型调整
    # 这里假设模型输出为每个检测框的坐标和置信度
    boxes = pred[0][:, :4]  # 检测框坐标
    scores = pred[0][:, 4]  # 置信度分数
    labels = pred[0][:, 5]  # 类别标签

    # 处理推理结果
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:  # 设定置信度阈值
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 写入帧到输出视频
    out.write(frame)

    # 显示当前帧（可选）
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

