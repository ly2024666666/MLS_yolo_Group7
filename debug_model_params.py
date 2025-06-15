#!/usr/bin/env python3
"""
调试BiFPN模型参数名称
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model

def debug_model_params():
    """调试模型参数名称"""
    print("=" * 60)
    print("调试BiFPN模型参数名称")
    print("=" * 60)
    
    try:
        # 创建BiFPN模型
        cfg = "models/BIFPN.yaml"
        device = torch.device('cpu')
        
        print(f"创建BiFPN模型: {cfg}")
        model = Model(cfg, ch=3, nc=1, anchors=None).to(device)
        
        # 获取所有参数名称
        all_params = list(model.state_dict().keys())
        print(f"\n模型总参数数量: {len(all_params)}")
        
        # 查找BiFPN相关参数
        bifpn_params = [k for k in all_params if 'BiFPN' in k or 'bifpn' in k.lower()]
        print(f"BiFPN相关参数数量: {len(bifpn_params)}")
        
        if bifpn_params:
            print("\nBiFPN参数列表:")
            for param in bifpn_params:
                print(f"  {param}")
        else:
            print("\n❌ 没有找到BiFPN参数！")
            print("\n查找包含'Add'的参数:")
            add_params = [k for k in all_params if 'Add' in k]
            for param in add_params:
                print(f"  {param}")
        
        # 打印所有参数（前50个）
        print(f"\n前50个参数名称:")
        for i, param in enumerate(all_params[:50]):
            print(f"  {i:2d}: {param}")
        
        # 查找模型结构中的层
        print(f"\n模型结构:")
        for i, (name, module) in enumerate(model.named_modules()):
            if 'BiFPN' in str(type(module)) or 'Add' in str(type(module)):
                print(f"  {i:2d}: {name} -> {type(module)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_model_params() 