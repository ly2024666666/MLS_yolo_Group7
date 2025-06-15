#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vanilla模型GradCAM测试脚本
"""

import os
import subprocess

def test_vanilla_gradcam():
    """测试vanilla.pt模型的GradCAM功能"""
    print("🚀 开始测试Vanilla模型GradCAM...")
    print("=" * 50)
    
    # 检查必要文件
    model_path = "vanilla.pt"
    img_path = "data/images/girl.jpg"
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        return False
    
    if not os.path.exists(img_path):
        print(f"❌ 错误: 找不到测试图片 {img_path}")
        return False
    
    print(f"✅ 模型文件: {model_path}")
    print(f"✅ 测试图片: {img_path}")
    print()
    
    # 创建输出目录
    output_dir = "outputs_vanilla/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行修复版本的GradCAM
    print("🔥 运行修复版本的GradCAM...")
    cmd = [
        "python", "utils/gradcam/main_gradcam_fixed.py",
        "--model-path", model_path,
        "--img-path", img_path,
        "--output-dir", output_dir,
        "--method", "gradcam",
        "--device", "cpu"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Vanilla模型GradCAM运行成功!")
            print(result.stdout)
        else:
            print("❌ 运行失败:")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            return False
    except subprocess.TimeoutExpired:
        print("❌ 运行超时")
        return False
    except Exception as e:
        print(f"❌ 运行时出错: {e}")
        return False
    
    print()
    print("🎉 测试完成!")
    print(f"📁 结果保存在: {output_dir}")
    
    return True

if __name__ == "__main__":
    success = test_vanilla_gradcam()
    if success:
        print("\n" + "=" * 50)
        print("✨ 测试成功!")
        print("现在您可以:")
        print("1. 查看outputs_vanilla/目录中的结果")
        print("2. 尝试其他图片")
    else:
        print("\n" + "=" * 50)
        print("❌ 测试失败!")
        print("请检查错误信息")