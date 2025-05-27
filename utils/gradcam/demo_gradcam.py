#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GradCAM演示脚本
使用data/images/bus.jpg作为示例图片
"""

import os
import sys
import subprocess

def run_gradcam_demo():
    """运行GradCAM演示"""
    print("🚀 开始运行YOLOv5 GradCAM演示...")
    print("=" * 50)
    
    # 检查必要文件
    model_path = "yolov5s.pt"
    img_path = "data/images/bus.jpg"
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        print("请确保yolov5s.pt文件存在于当前目录中")
        return False
    
    if not os.path.exists(img_path):
        print(f"❌ 错误: 找不到测试图片 {img_path}")
        return False
    
    print(f"✅ 模型文件: {model_path}")
    print(f"✅ 测试图片: {img_path}")
    print()
    
    # 创建输出目录
    output_dir = "outputs/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行GradCAM
    print("🔥 运行GradCAM...")
    cmd_gradcam = [
        "python", "main_gradcam.py",
        "--model-path", model_path,
        "--img-path", img_path,
        "--output-dir", output_dir,
        "--method", "gradcam",
        "--device", "cpu"
    ]
    
    try:
        result = subprocess.run(cmd_gradcam, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ GradCAM运行成功!")
        else:
            print("❌ GradCAM运行失败:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ GradCAM运行超时")
        return False
    except Exception as e:
        print(f"❌ 运行GradCAM时出错: {e}")
        return False
    
    print()
    
    # 运行GradCAM++
    print("🔥 运行GradCAM++...")
    cmd_gradcampp = [
        "python", "main_gradcam.py",
        "--model-path", model_path,
        "--img-path", img_path,
        "--output-dir", output_dir,
        "--method", "gradcampp",
        "--device", "cpu"
    ]
    
    try:
        result = subprocess.run(cmd_gradcampp, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ GradCAM++运行成功!")
        else:
            print("❌ GradCAM++运行失败:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ GradCAM++运行超时")
        return False
    except Exception as e:
        print(f"❌ 运行GradCAM++时出错: {e}")
        return False
    
    print()
    print("🎉 演示完成!")
    print(f"📁 结果保存在: {output_dir}")
    print("📸 查看生成的热力图可视化结果")
    
    return True

def show_usage():
    """显示使用说明"""
    print("📖 YOLOv5 GradCAM 使用说明")
    print("=" * 50)
    print()
    print("1. 基本使用:")
    print("   python main_gradcam.py --model-path yolov5s.pt --img-path data/images/bus.jpg")
    print()
    print("2. 使用GradCAM++:")
    print("   python main_gradcam.py --model-path yolov5s.pt --img-path data/images/bus.jpg --method gradcampp")
    print()
    print("3. 批量处理:")
    print("   python main_gradcam.py --model-path yolov5s.pt --img-path data/images/")
    print()
    print("4. 使用GPU:")
    print("   python main_gradcam.py --model-path yolov5s.pt --img-path data/images/bus.jpg --device cuda")
    print()
    print("5. 测试实现:")
    print("   python test_gradcam.py")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_usage()
    else:
        success = run_gradcam_demo()
        if success:
            print("\n" + "=" * 50)
            print("✨ 演示成功完成!")
            print("现在您可以:")
            print("1. 查看outputs/目录中的结果")
            print("2. 尝试其他图片")
            print("3. 使用不同的参数")
            print("4. 运行 python demo_gradcam.py --help 查看更多用法")
        else:
            print("\n" + "=" * 50)
            print("❌ 演示失败!")
            print("请检查错误信息并确保:")
            print("1. yolov5s.pt模型文件存在")
            print("2. 所有依赖包已安装")
            print("3. Python环境配置正确") 