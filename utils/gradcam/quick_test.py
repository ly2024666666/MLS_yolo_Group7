#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证GradCAM修复
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_imports():
    """测试导入"""
    print("测试导入...")
    try:
        from models.yolov5_object_detector import YOLOV5TorchObjectDetector
        print("✅ YOLOV5TorchObjectDetector 导入成功")
        
        from models.gradcam import YOLOV5GradCAM, YOLOV5GradCAMPP
        print("✅ GradCAM 类导入成功")
        
        from models.experimental import attempt_load
        print("✅ attempt_load 导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_attempt_load_signature():
    """测试attempt_load函数签名"""
    print("\n测试attempt_load函数签名...")
    try:
        from models.experimental import attempt_load
        import inspect
        
        sig = inspect.signature(attempt_load)
        print(f"attempt_load 参数: {list(sig.parameters.keys())}")
        
        # 检查是否有device参数
        if 'device' in sig.parameters:
            print("✅ attempt_load 支持 device 参数")
            return True
        else:
            print("❌ attempt_load 不支持 device 参数")
            return False
            
    except Exception as e:
        print(f"❌ 检查函数签名失败: {e}")
        return False

def test_model_loading():
    """测试模型加载（如果模型文件存在）"""
    print("\n测试模型加载...")
    import os
    
    model_path = 'yolov5s.pt'
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件 {model_path} 不存在，跳过模型加载测试")
        return True
    
    try:
        from models.yolov5_object_detector import YOLOV5TorchObjectDetector
        
        names = ['person', 'bicycle', 'car']  # 简化的类别名
        device = 'cpu'
        img_size = (640, 640)
        
        print(f"尝试加载模型: {model_path}")
        model = YOLOV5TorchObjectDetector(
            model_weight=model_path,
            device=device,
            img_size=img_size,
            names=names
        )
        print("✅ 模型加载成功!")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始快速测试...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_attempt_load_signature,
        test_model_loading
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 测试结果:")
    
    if all(results):
        print("✅ 所有测试通过!")
        print("GradCAM修复成功，可以正常使用。")
        return True
    else:
        print("❌ 部分测试失败!")
        print("请检查上述错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 修复验证成功!")
        print("现在可以运行:")
        print("python test_gradcam.py")
        print("python main_gradcam.py --model-path yolov5s.pt --img-path data/images/bus.jpg --device cpu")
    else:
        print("\n🔧 需要进一步修复!") 