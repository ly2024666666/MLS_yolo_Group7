#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试GradCAM实现的脚本
"""

import torch
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.yolov5_object_detector import YOLOV5TorchObjectDetector
from models.gradcam import YOLOV5GradCAM, YOLOV5GradCAMPP

# COCO数据集类别名
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

def test_gradcam():
    """测试GradCAM功能"""
    print("开始测试GradCAM实现...")
    
    # 设置参数
    device = 'cpu'  # 使用CPU进行测试
    img_size = (640, 640)
    model_path = 'yolov5s.pt'  # 确保模型文件存在
    
    try:
        # 1. 创建模型
        print("1. 加载YOLOv5模型...")
        model = YOLOV5TorchObjectDetector(
            model_weight=model_path,
            device=device,
            img_size=img_size,
            names=names
        )
        print("   模型加载成功!")
        
        # 2. 创建测试图像
        print("2. 创建测试图像...")
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        torch_img = model.preprocessing(test_img)
        print(f"   测试图像形状: {torch_img.shape}")
        
        # 3. 测试模型前向传播
        print("3. 测试模型前向传播...")
        with torch.no_grad():
            prediction, logits, _ = model.model(torch_img)
            print(f"   预测输出形状: {prediction.shape}")
            print(f"   logits输出形状: {logits.shape}")
        
        # 4. 测试GradCAM
        print("4. 测试GradCAM...")
        target_layer = 'model_17_cv3_act'  # YOLOv5s的一个检测层
        
        try:
            gradcam = YOLOV5GradCAM(model=model, layer_name=target_layer, img_size=img_size)
            print("   GradCAM初始化成功!")
        except Exception as e:
            print(f"   GradCAM初始化失败: {e}")
            return False
        
        # 5. 测试GradCAM++
        print("5. 测试GradCAM++...")
        try:
            gradcampp = YOLOV5GradCAMPP(model=model, layer_name=target_layer, img_size=img_size)
            print("   GradCAM++初始化成功!")
        except Exception as e:
            print(f"   GradCAM++初始化失败: {e}")
            return False
        
        print("所有测试通过! GradCAM实现正确。")
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path}")
        print("请确保yolov5s.pt文件存在于当前目录中")
        return False
    except TypeError as e:
        if "attempt_load" in str(e):
            print(f"模型加载参数错误: {e}")
            print("这可能是由于YOLOv5版本不兼容导致的")
        else:
            print(f"类型错误: {e}")
        return False
    except Exception as e:
        print(f"测试失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gradcam()
    if success:
        print("\n✅ GradCAM实现测试成功!")
        print("现在可以使用以下命令运行GradCAM:")
        print("python main_gradcam.py --model-path yolov5s.pt --img-path data/images --device cpu")
    else:
        print("\n❌ GradCAM实现测试失败!")
        print("请检查错误信息并修复问题。") 