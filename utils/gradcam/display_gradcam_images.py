import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题)

def display_gradcam_images():
    """
    显示gradcam文件夹中的所有图片在一个页面上
    """
    # 设置图片文件夹路径
    gradcam_folder = Path("../../outputs/bus/gradcam")
    
    # 获取所有jpg图片文件
    image_files = sorted([f for f in gradcam_folder.glob("*.jpg")])
    
    if not image_files:
        print("在gradcam文件夹中没有找到jpg图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 计算子图布局 - 尽量接近正方形
    n_images = len(image_files)
    cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))
    
    # 创建图形和子图
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    fig.suptitle('GradCAM 可视化结果', fontsize=16, fontweight='bold')
    
    # 如果只有一行或一列，确保axes是二维数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 显示每张图片
    for idx, img_path in enumerate(image_files):
        row = idx // cols
        col = idx % cols
        
        try:
            # 读取图片
            img = mpimg.imread(img_path)
            
            # 显示图片
            axes[row, col].imshow(img)
            axes[row, col].set_title(img_path.name, fontsize=10)
            axes[row, col].axis('off')
            
        except Exception as e:
            print(f"无法读取图片 {img_path}: {e}")
            axes[row, col].text(0.5, 0.5, f"无法加载\n{img_path.name}", 
                              ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 显示图片
    plt.show()
    
    # 可选：保存为文件
    save_path = "gradcam_overview.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片概览已保存为: {save_path}")

if __name__ == "__main__":
    display_gradcam_images() 