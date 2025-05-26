import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import cv2

class YOLOv5SHAP:
    def __init__(self, weights='yolov5n.pt', device=''):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.imgsz = check_img_size((640, 640), s=self.stride)
        self.names = self.model.names
        
    def prepare_image(self, img_path):
        """使用OpenCV准备输入图像"""
        # 读取原始图像
        im0 = cv2.imread(img_path)
        if im0 is None:
            raise ValueError(f"无法读取图像: {img_path}")
            
        # 调整图像大小
        im = cv2.resize(im0, self.imgsz)
        
        # BGR转RGB
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        
        # 转换为tensor并归一化
        im = torch.from_numpy(im.copy()).to(self.device)
        im = im.float()
        im /= 255.0
        
        # 添加batch维度
        if len(im.shape) == 3:
            im = im[None]
            
        return im, im0
            
    def predict(self, x):
        """模型预测函数"""
        pred = self.model(x)
        pred = non_max_suppression(pred, 0.25, 0.45)
        return pred

    def model_wrapper(self, x):
        """包装模型输出为单个张量"""
        pred = self.model(x)
        # 将列表输出转换为单个张量
        if isinstance(pred, list):
            # 取第一个检测头的输出
            pred = pred[0]
        return pred
        
    def explain_prediction(self, img_path, background_images=None, n_background=10):
        """使用SHAP解释模型预测"""
        # 准备输入图像
        im, im0s = self.prepare_image(img_path)
        
        # 创建背景数据
        if background_images is None:
            # 创建随机背景数据
            background = torch.randn((n_background,) + im.shape[1:], device=self.device)
        else:
            background = background_images
            
        # 创建SHAP解释器
        explainer = shap.DeepExplainer(
            self.model_wrapper,  # 使用包装后的模型
            background  # 背景数据
        )
        
        # 计算SHAP值
        shap_values = explainer.shap_values(im)
        
        # 可视化SHAP值
        plt.figure(figsize=(10, 10))
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # 取第一个检测头的SHAP值
        shap.image_plot(shap_values, im)
        plt.savefig('shap_visualization.png')
        plt.close()
        
        return shap_values
        
    def plot_feature_importance(self, shap_values, feature_names=None):
        """绘制特征重要性图"""
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(shap_values.shape[-1])]
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.savefig('feature_importance.png')
        plt.close()
        
    def visualize_training_progress(self, training_history):
        """可视化训练过程中的特征重要性变化"""
        plt.figure(figsize=(12, 6))
        
        # 绘制训练损失
        plt.subplot(1, 2, 1)
        plt.plot(training_history['train_loss'], label='Training Loss')
        plt.plot(training_history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制特征重要性变化
        plt.subplot(1, 2, 2)
        for feature in training_history['feature_importance'].keys():
            plt.plot(training_history['feature_importance'][feature], 
                    label=f'Feature {feature}')
        plt.title('Feature Importance Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Importance')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

def main():
    # 使用示例
    shap_analyzer = YOLOv5SHAP(weights='yolov5n.pt')
    
    # 解释单张图片的预测
    shap_values = shap_analyzer.explain_prediction(r'E:\Materials_25Spring\Machine_Learning_System\pj\MLS_yolo_Group7\data\images\bus.jpg')
    
    # 绘制特征重要性
    shap_analyzer.plot_feature_importance(shap_values)
    
    # 模拟训练历史数据
    training_history = {
        'train_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
        'val_loss': [0.6, 0.5, 0.4, 0.3, 0.2],
        'feature_importance': {
            'feature1': [0.3, 0.4, 0.5, 0.6, 0.7],
            'feature2': [0.7, 0.6, 0.5, 0.4, 0.3]
        }
    }
    
    # 可视化训练进度
    shap_analyzer.visualize_training_progress(training_history)

if __name__ == '__main__':
    main() 