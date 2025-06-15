#!/usr/bin/env python3
"""
测试BiFPN权重适配器修复
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
from utils.failed_bifpn.bifpn_weight_adapter import BiFPNWeightAdapter
from utils.general import LOGGER

def test_bifpn_adapter():
    """测试BiFPN权重适配器"""
    print("=" * 50)
    print("测试BiFPN权重适配器")
    print("=" * 50)
    
    try:
        # 1. 创建BiFPN模型
        cfg = "models/BIFPN.yaml"
        device = torch.device('cpu')  # 使用CPU避免GPU内存问题
        
        print(f"1. 创建BiFPN模型: {cfg}")
        model = Model(cfg, ch=3, nc=1, anchors=None).to(device)
        print(f"   模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 2. 测试权重适配器
        print("2. 测试权重适配器")
        adapter = BiFPNWeightAdapter()
        
        # 检查是否有预训练权重文件
        weights_path = "yolov5n.pt"
        if not Path(weights_path).exists():
            print(f"   警告: 预训练权重文件 {weights_path} 不存在")
            print("   跳过权重适配测试")
            return True
            
        print(f"   加载预训练权重: {weights_path}")
        adapted_count = adapter.adapt_pretrained_weights(model, weights_path)
        print(f"   权重适配完成，适配参数数量: {adapted_count}")
        
        # 3. 测试前向传播
        print("3. 测试前向传播")
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 640, 640)
            print(f"   输入张量形状: {x.shape}")
            
            try:
                output = model(x)
                print(f"   前向传播成功")
                if isinstance(output, (list, tuple)):
                    print(f"   输出数量: {len(output)}")
                    for i, out in enumerate(output):
                        if hasattr(out, 'shape'):
                            print(f"   输出 {i} 形状: {out.shape}")
                        else:
                            print(f"   输出 {i} 类型: {type(out)}")
                else:
                    if hasattr(output, 'shape'):
                        print(f"   输出形状: {output.shape}")
                    else:
                        print(f"   输出类型: {type(output)}")
                    
            except Exception as e:
                print(f"   前向传播失败: {e}")
                return False
        
        print("=" * 50)
        print("✅ BiFPN权重适配器测试通过！")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bifpn_adapter()
    sys.exit(0 if success else 1) 