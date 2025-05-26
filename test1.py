# import torch
# from models.yolo import Model
#
# # 加载 YOLOv5 模型（选项：yolov5n, yolov5s, yolov5m, yolov5l, yolov5x）
# model = torch.hub.load("ultralytics/yolov5", "custom",
#                        r"E:\机器学习PJ\yolov5\runs\train\exp3\weights\best.pt",
#                        force_reload=True)  # 默认：yolov5s
#
# # 定义输入图像源（URL、本地文件、PIL 图像、OpenCV 帧、numpy 数组或列表）
# img = "https://ultralytics.com/images/zidane.jpg"  # 示例图像
#
# # 执行推理（自动处理批处理、调整大小、归一化）
# results = model(img)
#
# # 处理结果（选项：.print(), .show(), .save(), .crop(), .pandas()）
# results.print()  # 将结果打印到控制台
# results.show()  # 在窗口中显示结果
# results.save()  # 将结果保存到 runs/detect/exp
import torch

from models.yolo import Model  # 导入本地修改后的 Model 类

# 加载自定义模型（使用本地代码）
device = torch.device("cpu")  # 或 "cuda"
ckpt_path = r"E:\机器学习PJ\yolov5\runs\train\exp3\weights\best.pt"

# 手动构建模型（需匹配训练时的 cfg 文件）
model = Model(cfg=r"E:\机器学习PJ\yolov5\models\yolov5s.yaml", ch=3, nc=80)  # 替换为你的 YAML 路径
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"].state_dict())
model = model.to(device).eval()

# 执行推理
img = "https://ultralytics.com/images/zidane.jpg"
results = model(img)
results.print()
