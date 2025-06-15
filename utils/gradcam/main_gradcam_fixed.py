#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import time
import argparse
import numpy as np
import sys
import torch
import logging
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1].parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.gradcam import YOLOV5GradCAM, YOLOV5GradCAMPP
from models.yolov5_object_detector import YOLOV5TorchObjectDetector
from utils.general import LOGGER, colorstr
import cv2

LOGGER.setLevel(logging.INFO)

def detect_model_type_and_layers(model):
    """智能检测模型类型并返回合适的目标层配置"""
    try:
        model_type = "unknown"
        target_layers = []
        
        # 获取模型结构
        if hasattr(model, 'model'):
            model_blocks = model.model.model
            
            # 检查第一个卷积层的输出通道数
            if hasattr(model_blocks[0], 'conv'):
                first_conv = model_blocks[0].conv
                out_channels = first_conv.out_channels
                LOGGER.info(f'检测到首层输出通道数: {out_channels}')
                
                # 根据通道数判断模型类型
                if out_channels == 16:
                    model_type = "vanilla"
                    # 对于vanilla模型，尝试使用较早的层
                    target_layers = ['model_10_cv3_act', 'model_13_cv3_act', 'model_16_cv3_act']
                elif out_channels == 32:
                    model_type = "yolov5s"
                    target_layers = ['model_17_cv3_act', 'model_20_cv3_act', 'model_23_cv3_act']
                elif out_channels == 64:
                    model_type = "yolov5m"
                    target_layers = ['model_17_cv3_act', 'model_20_cv3_act', 'model_23_cv3_act']
            
            # 如果没有找到合适的配置，尝试自动发现可用层
            if not target_layers:
                LOGGER.info('尝试自动发现可用层...')
                available_layers = []
                for i, block in enumerate(model_blocks):
                    if isinstance(block, torch.nn.Module):
                        if hasattr(block, 'cv3'):
                            layer_name = f'model_{i}_cv3_act'
                            available_layers.append((i, layer_name))
                        elif hasattr(block, 'cv2'):
                            layer_name = f'model_{i}_cv2_act'
                            available_layers.append((i, layer_name))
                        elif hasattr(block, 'cv1'):
                            layer_name = f'model_{i}_cv1_act'
                            available_layers.append((i, layer_name))
                
                # 选择合适的层（通常选择后面的几层）
                if available_layers:
                    # 选择最后几个可用层
                    selected_layers = available_layers[-3:] if len(available_layers) >= 3 else available_layers
                    target_layers = [layer[1] for layer in selected_layers]
                    LOGGER.info(f'自动发现的层: {target_layers}')
        
        if target_layers:
            LOGGER.info(f'模型类型: {model_type}, 使用目标层: {target_layers}')
            return target_layers
        
        # 如果所有检测都失败，返回默认配置
        LOGGER.warning('无法确定模型类型，使用默认配置')
        return ['model_10_cv3_act']
        
    except Exception as e:
        LOGGER.warning(f'模型检测失败: {str(e)}，使用默认配置')
        return ['model_10_cv3_act']

def get_res_img(bbox, mask, res_img):
    """生成热力图可视化结果"""
    try:
        # 确保mask是正确的格式
        if isinstance(mask, torch.Tensor):
            mask = mask.detach()
            
        # 处理多维度的mask
        while len(mask.shape) > 2:
            mask = mask.squeeze(0)
        
        # 确保mask是2D张量
        if len(mask.shape) == 1:
            # 如果是1D，尝试重塑为合理的2D形状
            size = int(np.sqrt(mask.shape[0]))
            if size * size == mask.shape[0]:
                mask = mask.reshape(size, size)
            else:
                LOGGER.warning(f'无法重塑mask形状: {mask.shape}')
                return res_img, None
        
        # 转换为numpy
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # 归一化mask到0-255
        mask = ((mask - mask.min()) / (mask.max() - mask.min() + 1e-8) * 255).astype(np.uint8)
        
        # 调整mask大小以匹配输入图像
        target_height, target_width = res_img.shape[:2]
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        
        # 归一化热力图
        n_heatmat = (heatmap / 255.0).astype(np.float32)
        
        # 归一化原始图像
        if res_img.max() > 1:
            res_img = res_img / 255.0
        
        # 叠加热力图
        alpha = 0.4  # 热力图透明度
        res_img = (1 - alpha) * res_img + alpha * n_heatmat
        
        # 归一化最终结果
        res_img = (res_img / res_img.max()).astype(np.float32)
        
        return res_img, n_heatmat
        
    except Exception as e:
        LOGGER.error(f'生成热力图失败: {str(e)}')
        return res_img, None

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """绘制单个边界框"""
    try:
        # 确保图像格式正确
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # 复制图像以避免修改原始图像
        img = img.copy()
        
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        color = color or [random.randint(0, 255) for _ in range(3)]
        
        # 确保坐标是整数
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        
        # 绘制边界框
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2_text = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2_text, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
        return img
        
    except Exception as e:
        LOGGER.error(f'绘制边界框失败: {str(e)}')
        return img

def process_image(img_path, model, device, input_size, method, output_dir, target_layer):
    """处理单张图片"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            LOGGER.error(f'无法读取图片: {img_path}')
            return False
            
        LOGGER.info(f'原始图像尺寸: {img.shape}')
            
        # 图像预处理
        torch_img = model.preprocessing(img[..., ::-1])
        LOGGER.info(f'预处理后张量尺寸: {torch_img.shape}')
        
        # 选择GradCAM方法
        try:
            saliency_method = (YOLOV5GradCAMPP if method == 'gradcampp' else YOLOV5GradCAM)(
                model=model, 
                layer_name=target_layer, 
                img_size=input_size
            )
        except Exception as e:
            LOGGER.error(f'创建GradCAM方法失败: {str(e)}')
            return False
        
        # 获取预测结果
        try:
            masks, logits, [boxes, _, class_names, conf] = saliency_method(torch_img)
        except Exception as e:
            LOGGER.error(f'GradCAM计算失败: {str(e)}')
            return False
        
        # 检查是否有检测结果
        if not boxes or len(boxes[0]) == 0:
            LOGGER.warning(f'图片 {img_path} 未检测到目标')
            return True
        
        # 准备结果图像
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]  # RGB to BGR
        
        # 创建保存目录
        img_name = Path(img_path).stem
        save_path = Path(output_dir) / img_name / method
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 处理每个检测到的目标
        success_count = 0
        for i, (mask, box, cls_name, conf_val) in enumerate(zip(masks, boxes[0], class_names[0], conf[0])):
            res_img = result.copy()
            label = f'{cls_name} {conf_val:.2f}'
            
            LOGGER.info(f'处理第 {i+1} 个目标: {label}, mask形状: {mask.shape if hasattr(mask, "shape") else "unknown"}')
            
            # 生成热力图
            res_img, heat_map = get_res_img(box, mask, res_img)
            if heat_map is None:
                LOGGER.warning(f'第 {i+1} 个目标的热力图生成失败')
                continue
                
            # 添加边界框
            res_img = plot_one_box(box, res_img, label=label, color=[random.randint(0, 255) for _ in range(3)])
            
            # 调整大小并保存
            res_img_final = cv2.resize(res_img, dsize=(img.shape[1], img.shape[0]))
            output_path = save_path / f'{target_layer.split("_")[1]}_{i}.jpg'
            
            # 确保图像格式正确
            if res_img_final.max() <= 1:
                res_img_final = (res_img_final * 255).astype(np.uint8)
            
            cv2.imwrite(str(output_path), res_img_final)
            LOGGER.info(f'已保存: {output_path}')
            success_count += 1
            
        return success_count > 0
        
    except Exception as e:
        LOGGER.error(f'处理图片失败 {img_path}: {str(e)}')
        import traceback
        LOGGER.error(f'详细错误信息:\n{traceback.format_exc()}')
        return False

def main(args):
    """主函数"""
    # 设备设置
    device = torch.device(args.device)
    
    # 加载模型
    LOGGER.info(colorstr('加载模型...'))
    try:
        model = YOLOV5TorchObjectDetector(
            args.model_path,
            device,
            img_size=(args.img_size, args.img_size),
            names=['smoke']  # 可以根据需要修改类别
        )
    except Exception as e:
        LOGGER.error(f'模型加载失败: {str(e)}')
        return
    
    # 获取适合的目标层
    target_layers = detect_model_type_and_layers(model)
    
    # 处理输入路径
    if os.path.isdir(args.img_path):
        img_files = [os.path.join(args.img_path, f) for f in os.listdir(args.img_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    else:
        img_files = [args.img_path]
    
    # 处理每个图片
    total_success = 0
    for img_path in img_files:
        LOGGER.info(f'\n处理图片: {img_path}')
        for target_layer in target_layers:
            LOGGER.info(f'使用目标层: {target_layer}')
            try:
                if process_image(
                    img_path=img_path,
                    model=model,
                    device=device,
                    input_size=(args.img_size, args.img_size),
                    method=args.method,
                    output_dir=args.output_dir,
                    target_layer=target_layer
                ):
                    LOGGER.info(f'成功处理层 {target_layer}')
                    total_success += 1
                else:
                    LOGGER.warning(f'处理层 {target_layer} 时出现问题')
            except Exception as e:
                LOGGER.error(f'处理层 {target_layer} 时发生异常: {str(e)}')
    
    LOGGER.info(f'\n处理完成！成功处理 {total_success} 个层-图片组合')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='weights/yolov5s.pt', help='模型路径')
    parser.add_argument('--img-path', type=str, default='data/images', help='输入图片路径')
    parser.add_argument('--output-dir', type=str, default='outputs/', help='输出目录')
    parser.add_argument('--img-size', type=int, default=640, help='输入图片大小')
    parser.add_argument('--method', type=str, default='gradcam', choices=['gradcam', 'gradcampp'], help='GradCAM方法')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)