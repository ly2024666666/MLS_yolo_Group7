# GradCAM 工具集

这个目录包含了用于YOLOv5模型的GradCAM可视化工具。

## 文件说明

- `main_gradcam.py` - 主要的GradCAM生成脚本
- `test_gradcam.py` - 测试GradCAM实现的脚本
- `demo_gradcam.py` - GradCAM演示脚本
- `quick_test.py` - 快速测试脚本，验证GradCAM修复
- `display_gradcam_images.py` - 显示GradCAM结果图片的脚本

## 使用方法

### 1. 运行GradCAM分析

```bash
# 从项目根目录运行
cd /path/to/MLS_yolo_Group7

# 对单张图片运行GradCAM
python utils/gradcam/main_gradcam.py --model-path yolov5s.pt --img-path data/images/bus.jpg --device cpu

# 对整个文件夹的图片运行GradCAM
python utils/gradcam/main_gradcam.py --model-path yolov5s.pt --img-path data/images/ --device cpu

# 使用GradCAM++方法
python utils/gradcam/main_gradcam.py --model-path yolov5s.pt --img-path data/images/bus.jpg --method gradcampp
```

### 2. 测试GradCAM功能

```bash
# 快速测试
python utils/gradcam/quick_test.py

# 完整测试
python utils/gradcam/test_gradcam.py
```

### 3. 运行演示

```bash
python utils/gradcam/demo_gradcam.py
```

### 4. 显示GradCAM结果

```bash
# 显示所有生成的GradCAM图片
python utils/gradcam/display_gradcam_images.py
```

## 参数说明

### main_gradcam.py 参数

- `--model-path`: 模型文件路径 (默认: "weights/yolov5s.pt")
- `--img-path`: 输入图片路径或文件夹 (默认: 'data/images')
- `--output-dir`: 输出目录 (默认: 'outputs/')
- `--img-size`: 输入图片大小 (默认: 640)
- `--target-layer`: 目标层名称 (默认: 'model_17_cv3_act')
- `--method`: GradCAM方法 ('gradcam' 或 'gradcampp', 默认: 'gradcam')
- `--device`: 设备 ('cpu' 或 'cuda', 默认: 'cpu')
- `--no_text_box`: 不显示标签和边界框

## 输出结果

GradCAM结果将保存在 `outputs/[图片名]/[方法名]/` 目录下，文件命名格式为 `[层编号]_[目标编号].jpg`。

例如：
- `outputs/bus/gradcam/17_0.jpg` - 第17层检测到的第0个目标的GradCAM结果
- `outputs/bus/gradcam/20_1.jpg` - 第20层检测到的第1个目标的GradCAM结果

## 注意事项

1. 确保从项目根目录运行脚本
2. 确保模型文件存在
3. 确保输入图片路径正确
4. 如果使用GPU，确保CUDA可用 