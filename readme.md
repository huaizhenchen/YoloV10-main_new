# Yolo V10：训练、验证、推理

## 0. 环境配置

### 测试硬件环境

- **i7 13700KF**
- **16 GB * 2，DDR5，6400 MHz**
- **RTX 4080**

### 测试软件环境

- **Windows 11 专业版 23H2**
- **显卡驱动：555.85**
- **CUDA：11.8**
- **cuDNN：8.9.5**
- **TensorRT：8.6.1.6（可选）**
- **conda：23.3.1**

### 环境安装命令

```shell
conda create -n yolov10 python=3.9
conda activate yolov10
pip install -r requirements.txt
pip install -e .
```

- **注意**

1. 默认安装的torch是CPU的版本，如需要用GPU的版本，请**卸载原来的torch和torchvision，安装GPU版本的torch和torchvision**

   ```shell
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. 建议**修改conda的安装源和pip的安装源**，以提高安装包的下载速度

## 1. 训练

### 方法1

```shell
yolo detect train data=COCO.yaml model=model/yolov10b.pt epochs=40 batch=4 device=0
# yolo detect train data=COCO.yaml model=model/yolov10x.pt epochs=500 batch=128 device=0,1,2,3,4,5,6,7 
```

### 方法2

```python
from ultralytics import YOLOv10

model = YOLOv10()                                 # 1. Train the model from scratch
# model = YOLOv10.from_pretrained('yolov10x.pt')  # 2. Finetune the model
# model = YOLOv10('yolov10x.pt')                  # 3. Finetune the model

model.train(data='VOC.yaml', epochs=500, batch=128, imgsz=640)
```

### 数据集

- 对应**ultralytics/cfg/dataset**路径下指定的**yaml**文件，在文件中记有**train、val、test的数据集路径位置**和**检测目标类别**。

### 结果输出

- 验证结果输出至当前目录的 **runs/detect/train** 文件夹。

## 2. 精度验证

### 方法1

```shell
yolo val model=model/yolov10x.pt data=COCO.yaml batch=128
```

### 方法2

```python
from ultralytics import YOLOv10

model = YOLOv10('yolov10x.pt')

model.val(data='COCO.yaml', batch=128)
```

### 数据输入

- 当前目录下的 **datasets** 文件夹。

### 结果输出

- 验证结果输出至当前目录的 **runs/detect/val** 文件夹。

## 3. 推理

### pt模型

#### 方法1

```shell
yolo predict model=model/yolov10x.pt source=ultralytics/assets
```

- **predict**：调用预测方法

- **model**：选择模型——**yolov10{n/s/m/b/l/x}**

- **source**：测试数据
  - **目录**：读取目录下的所有文件
    - `source=ultralytics/assets`
  - **文件**：读取指定文件
    - `source=ultralytics/assets/*.jpg`
    - `source=ultralytics/assets/*.mp4`

- **推理结果**保存在 **./runs/detect/predict** 中

#### 方法2

```shell
python inference.py
```

- 在**源代码**中设置**mode**为**image或video**，用于**测试图像或者测试视频**
- **推理结果**保存在 **./runs/predict** 中

## 4. 推理加速

### ONNX模型

#### 模型转换：pt→onnx

```bash
yolo export model=model/yolov10x.pt format=onnx opset=13 simplify
```

#### 方法1

```shell
yolo predict model=model/yolov10x.onnx source=ultralytics/assets
```

- **predict**：调用预测方法

- **model**：选择模型——**yolov10{n/s/m/b/l/x}**

- **source**：测试数据
  - **目录**：读取目录下的所有文件
    - `source=ultralytics/assets`
  - **文件**：读取指定文件
    - `source=ultralytics/assets/*.jpg`
    - `source=ultralytics/assets/*.mp4`

- **推理结果**保存在 **./runs/detect/predict** 中

#### 方法2

```shell
python inference.py
```

- 将**源代码**中的模型，由**pt模型**改为**onnx模型**
- 在**源代码**中设置**mode**为**image或video**，用于**测试图像或者测试视频**
- **推理结果**保存在 **./runs/predict** 中

#### 方法3

```shell
python inference-onnx.py
```

- 在**源代码**中设置**mode**为**image或video**，用于**测试图像或者测试视频**
- **推理结果**保存在 **./runs/predict** 中
- 该方法可**脱离ultralytics框架**运行

#### 参数配置

##### 推理设备

- `providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']`：使用**GPU**推理
- `providers = ['CPUExecutionProvider']`：使用**CPU**推理

##### 版本对应关系

|     ONNX  Runtime     | CUDA |               cuDNN               |
| :-------------------: | :--: | :-------------------------------: |
|         1.17          | 12.2 | 8.9.2 (Linux), 8.9.2.26 (Windows) |
|   1.15, 1.16, 1.17    | 11.8 | 8.2.4 (Linux), 8.5.0.96 (Windows) |
|      1.14, 1.13       | 11.6 | 8.2.4 (Linux), 8.5.0.96 (Windows) |
| 1.12, 1.11, 1.10, 1.9 | 11.4 | 8.2.4 (Linux), 8.2.2.26 (Windows) |

### OpenVINO模型

#### 安装OpenVINO的开发包

```shell
pip install openvino
```

#### 参数配置修改

- 在 **.\ultralytics\cfg\default.yaml** 中修改**device**字段，**强制修改为CPU**。

#### 模型转换：pt→bin/xml

```bash
yolo export model=model/yolov10x.pt format=openvino half=True
```

- 会生成 **yolov10x_openvino_model** 文件夹
- 将**half**设置为True，可以**启用FP16模式**

#### 方法1

```shell
yolo predict model=model/yolov10x_openvino_model source=ultralytics/assets
```

- **predict**：调用预测方法

- **model**：选择模型——**yolov10{n/s/m/b/l/x}_openvino_model**

- **source**：测试数据
  - **目录**：读取目录下的所有文件
    - `source=ultralytics/assets`
  - **文件**：读取指定文件
    - `source=ultralytics/assets/*.jpg`
    - `source=ultralytics/assets/*.mp4`

- **推理结果**保存在 **./runs/detect/predict** 中

#### 方法2

```shell
python inference.py
```

- 将**源代码**中的模型，由**pt模型**改为**openvino模型的路径**
- 在**源代码**中设置**mode**为**image或video**，用于**测试图像或者测试视频**
- **推理结果**保存在 **./runs/predict** 中

### TensorRT模型

#### 安装TensorRT的开发包

- 安装[**编译工具**](https://download.microsoft.com/download/E/E/D/EEDF18A8-4AED-4CE0-BEBE-70A83094FC5A/BuildTools_Full.exe)：**visual_cpp_build_tools_2015_update_3_x64.iso**

- 安装**pyCUDA(2024.1)、TensorRT(8.6.1)**

  ```shell
  pip install pycuda
  pip install tensorrt
  ```

#### 参数配置修改

- 在 **.\ultralytics\cfg\default.yaml** 中修改**device**字段，**强制修改为GPU**，一般默认设置为**0**。

#### 模型转换：pt→onnx→trt

```bash
yolo export model=model/yolov10x.pt format=engine half=True simplify opset=13 workspace=16
```

#### 方法1

```shell
yolo predict model=model/yolov10x.engine source=ultralytics/assets
```

- **predict**：调用预测方法

- **model**：选择模型——**yolov10{n/s/m/b/l/x}**

- **source**：测试数据
  - **目录**：读取目录下的所有文件
    - `source=ultralytics/assets`
  - **文件**：读取指定文件
    - `source=ultralytics/assets/*.jpg`
    - `source=ultralytics/assets/*.mp4`

- **推理结果**保存在 **./runs/detect/predict** 中

#### 方法2

```shell
python inference.py
```

- 将**源代码**中的模型，由**pt模型**改为**engine模型**
- 在**源代码**中设置**mode**为**image或video**，用于**测试图像或者测试视频**
- **推理结果**保存在 **./runs/predict** 中

## 5. 参数配置

### 推理设备

​		在 **.\ultralytics\cfg\default.yaml** 中修改**device**字段，**device为0 - 7**，表示使用**GPU**，**device为cpu**，表示使用**CPU**。

### 打印LOG

​		在 **.\ultralytics\cfg\default.yaml** 中修改**verbose**字段，**verbose为True**表示**打印LOG**，**verbose为False**表示**不打印LOG**。

## 6. 参考

[GitHub - THU-MIG/yolov10: YOLOv10: Real-Time End-to-End Object Detection](https://github.com/THU-MIG/yolov10)