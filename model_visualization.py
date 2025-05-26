import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
from models.yolo import Model  # 导入YOLO模型类
import matplotlib as mpl
mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题)

class ModelVisualizer:
    def __init__(self, model_path):
        """
        初始化模型可视化器
        Args:
            model_path: 模型权重文件路径(.pt文件)
        """
        self.model_path = Path(model_path)
        # 添加安全加载选项
        torch.serialization.add_safe_globals([Model])
        self.model = torch.load(model_path, map_location='cpu', weights_only=False)
        # 将模型转换为半精度
        self.model['model'] = self.model['model'].half()
        self.metrics = {}
        
    def analyze_parameter_importance(self):
        """分析模型参数重要性
        通过计算每个参数层的L2范数来衡量其重要性
        L2范数越大，表示该层参数对模型的影响越大
        """
        # 获取模型参数
        params = self.model['model'].state_dict()
        
        # 计算每个参数的L2范数作为重要性指标
        importance = {}
        for name, param in params.items():
            if 'weight' in name:  # 只分析权重参数，忽略偏置项
                # 计算L2范数：sqrt(sum(x^2))
                importance[name] = torch.norm(param).item()
        
        # 归一化重要性分数到[0,1]区间
        max_importance = max(importance.values())
        importance = {k: v/max_importance for k, v in importance.items()}
        
        # 绘制参数重要性条形图
        plt.figure(figsize=(15, 8))
        names = list(importance.keys())
        values = list(importance.values())
        
        # 按重要性排序
        sorted_idx = np.argsort(values)
        names = [names[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        # 添加层名称解释
        def explain_layer_name(name):
            parts = name.split('.')
            explanation = []
            for part in parts:
                if part.isdigit():
                    explanation.append(f"第{part}层")
                elif part == 'cv':
                    explanation.append("卷积块")
                elif part.startswith('cv'):
                    explanation.append(f"第{part[2]}个卷积块")
                elif part == 'bn':
                    explanation.append("批量归一化")
                elif part == 'conv':
                    explanation.append("卷积层")
                elif part == 'weight':
                    explanation.append("权重")
            return ' - '.join(explanation)
        
        # 创建带有解释的标签
        explained_names = [f"{name}\n({explain_layer_name(name)})" for name in names]
        
        plt.barh(range(len(names)), values)
        plt.yticks(range(len(names)), explained_names, fontsize=8)
        plt.xlabel('参数重要性 (归一化L2范数)')
        plt.title('模型参数重要性分析\n条形越长表示该层越重要')
        plt.tight_layout()
        plt.savefig('parameter_importance.png')
        plt.close()
        
        return importance
    
    def visualize_training_progress(self):
        """可视化训练过程"""
        if 'train' not in self.model:
            print("警告：模型文件中没有找到训练历史数据。这可能是因为：")
            print("1. 模型是预训练模型")
            print("2. 保存模型时没有包含训练指标")
            print("3. 模型文件格式不完整")
            print("\n您可以：")
            print("1. 使用训练日志文件（如果有）来可视化训练过程")
            print("2. 继续查看参数重要性和层激活分布")
            print("3. 重新训练模型并确保保存训练指标")
            return
            
        # 获取训练指标
        metrics = self.model['train']
        
        # 绘制损失曲线
        plt.figure(figsize=(15, 5))
        
        # 绘制训练损失
        plt.subplot(1, 2, 1)
        if 'box_loss' in metrics:
            plt.plot(metrics['box_loss'], label='Box Loss')
        if 'obj_loss' in metrics:
            plt.plot(metrics['obj_loss'], label='Object Loss')
        if 'cls_loss' in metrics:
            plt.plot(metrics['cls_loss'], label='Class Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        
        # 绘制mAP曲线
        plt.subplot(1, 2, 2)
        if 'mAP_0.5' in metrics:
            plt.plot(metrics['mAP_0.5'], label='mAP@0.5')
        if 'mAP_0.5:0.95' in metrics:
            plt.plot(metrics['mAP_0.5:0.95'], label='mAP@0.5:0.95')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('Mean Average Precision')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
        
    def visualize_layer_activation(self, input_tensor):
        """
        可视化网络层的激活值分布
        通过分析每层输出的分布情况，可以：
        1. 检查是否存在梯度消失/爆炸问题
        2. 评估网络是否正常工作
        3. 了解特征提取的效果
        
        Args:
            input_tensor: 输入张量，用于前向传播
        """
        model = self.model['model']
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                # 捕获每层的输出
                activations[name] = output.detach()
            return hook
        
        # 注册钩子函数到每个卷积层和全连接层
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # 前向传播，收集激活值
        with torch.no_grad():
            model(input_tensor)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 绘制激活值分布
        plt.figure(figsize=(15, 10))
        for i, (name, activation) in enumerate(activations.items()):
            plt.subplot(len(activations), 1, i+1)
            # 将激活值展平并转换为numpy数组
            activation_np = activation.cpu().numpy().flatten()
            plt.hist(activation_np, bins=50)
            plt.title(f'Activation Distribution - {name}\nMean: {activation_np.mean():.3f}, Std: {activation_np.std():.3f}')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('layer_activations.png')
        plt.close()

    def visualize_training_progress_from_log(self, log_file):
        """从训练日志文件可视化训练过程
        Args:
            log_file: 训练日志文件路径（通常是results.csv）
        """
        try:
            print(f"尝试读取日志文件：{log_file}")
            import pandas as pd
            # 读取训练日志
            df = pd.read_csv(log_file)
            print(f"成功读取日志文件，列名：{df.columns.tolist()}")
            
            # 绘制损失曲线
            plt.figure(figsize=(15, 5))
            
            # 绘制训练损失
            plt.subplot(1, 2, 1)
            if 'train/box_loss' in df.columns:
                print("找到box_loss数据")
                plt.plot(df['train/box_loss'], label='Box Loss')
            if 'train/obj_loss' in df.columns:
                print("找到obj_loss数据")
                plt.plot(df['train/obj_loss'], label='Object Loss')
            if 'train/cls_loss' in df.columns:
                print("找到cls_loss数据")
                plt.plot(df['train/cls_loss'], label='Class Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.legend()
            
            # 绘制mAP曲线
            plt.subplot(1, 2, 2)
            if 'metrics/mAP_0.5' in df.columns:
                print("找到mAP_0.5数据")
                plt.plot(df['metrics/mAP_0.5'], label='mAP@0.5')
            if 'metrics/mAP_0.5:0.95' in df.columns:
                print("找到mAP_0.5:0.95数据")
                plt.plot(df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title('Mean Average Precision')
            plt.legend()
            
            plt.tight_layout()
            print("正在保存图表...")
            plt.savefig('training_progress_from_log.png')
            plt.close()
            print("图表保存完成")
            
        except Exception as e:
            print(f"发生错误：{str(e)}")
            print(f"错误类型：{type(e)}")
            import traceback
            print(f"错误堆栈：{traceback.format_exc()}")
            print("请确保提供了正确的训练日志文件路径")

def main():
    # 使用示例
    visualizer = ModelVisualizer(r'E:\Materials_25Spring\Machine_Learning_System\pj\MLS_yolo_Group7\runs\train\exp6\weights\best.pt')
    
    # 分析参数重要性
    importance = visualizer.analyze_parameter_importance()
    
    # 可视化训练过程（从模型文件）
    visualizer.visualize_training_progress()
    
    # 可视化训练过程（从日志文件）
    log_file = r'E:\Materials_25Spring\Machine_Learning_System\pj\MLS_yolo_Group7\runs\train\exp6\results.csv'
    visualizer.visualize_training_progress_from_log(log_file)
    
    # 可视化层激活
    input_tensor = torch.randn(1, 3, 640, 640).half()  # 使用半精度float16
    visualizer.visualize_layer_activation(input_tensor)

if __name__ == '__main__':
    main() 