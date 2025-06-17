# import pickle
# import torch


# # from divide.CSP_block import (Conv,C3,CrossConv,C3SPP,C3TR,Bottleneck,BottleneckCSP,C3Ghost,
# #             C3x,DWConv,DWConvTranspose2d,GhostBottleneck,GhostConv)
# # from divide.spatial import SPP,SPPF,Focus
# # from divide.util import Classify,Concat,Contract,Expand,Proto
# # from divide.backend import DetectMultiBackend
# # from divide.attention import CustomAttentionModule
# # # from divide.BiFPN import BiFPN_Add2,BiFPN_Add3
# # from divide.BIFPN import BiFPN_Feature2,BiFPN_Feature3

# # 自定义 Unpickler：将老路径映射到新路径
# class RenamingUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'models.common':
#             # 模块路径映射字典
#             remap = {
#                 'Conv':       ('divide.conv_block', 'Conv'),
#                 'C3':       ('divide.conv_block', 'C3'),
#                 'CrossConv':       ('divide.conv_block', 'CrossConv'),
#                 'C3SPP':       ('divide.conv_block', 'C3SPP'),
#                 'C3TR':       ('divide.conv_block', 'C3TR'),
#                 'Bottleneck':       ('divide.conv_block', 'Bottleneck'),
#                 'BottleneckCSP':       ('divide.conv_block', 'BottleneckCSP'),
#                 'C3Ghost':       ('divide.conv_block', 'C3Ghost'),
#                 'C3x':       ('divide.conv_block', 'C3x'),
#                 'DWConv':       ('divide.conv_block', 'DWConv'),
#                 'DWConvTranspose2d':       ('divide.conv_block', 'DWConvTranspose2d'),
#                 'GhostBottleneck':       ('divide.conv_block', 'BottGhostBottleneckleneck'),
#                 'GhostConv':       ('divide.conv_block', 'GhostConv'),
#                 'SPP':       ('divide.spatial', 'SPP'),
#                 'SPPF':       ('divide.spatial', 'SPPF'),
#                 'Focus':       ('divide.spatial', 'Focus'),
#                 'Classify':       ('divide.util', 'Classify'),
#                 'Concat':       ('divide.util', 'Concat'),
#                 'Contract':       ('divide.util', 'Contract'),
#                 'Expand':       ('divide.util', 'Expand'),
#                 'Proto':       ('divide.util', 'Proto'),
#                 'DetectMultiBackend':       ('divide.backend', 'DetectMultiBackend'),
#                 'CustomAttentionModule':       ('divide.attention', 'CustomAttentionModule'),
#                 'BiFPN_Feature2':       ('divide.BIFPN', 'BiFPN_Feature2'),
#                 'BiFPN_Feature3':       ('divide.BIFPN', 'BiFPN_Feature3'),
                
#             }
#             if name in remap:
#                 module, name = remap[name]
#         return super().find_class(module, name)

# def torch_load_with_remap(path, map_location='cpu'):
#     with open(path, 'rb') as f:
#         unpickler = RenamingUnpickler(f)
#         result = unpickler.load()
#     if map_location:
#         result = torch.load(path, map_location=map_location, pickle_module=pickle, weights_only=True)
#     return result

# # 尝试加载旧 pt 文件
# ckpt = torch_load_with_remap("yolov5n.pt", map_location="cpu")

# # 直接保存为新的 pt 文件（避免再走原来的反序列化路径）
# torch.save(ckpt, "new_model.pt")
# ckpt = torch.load("new_model.pt", map_location="cpu")

import sys
import types
import torch

# ✅ 从你自己的模块中导入所有被引用的类
from divide.conv_blocks import (Conv,DWConv,DWConvTranspose2d,Bottleneck,CrossConv,GhostConv,GhostBottleneck,GSConv)
from divide.CSP_block import (C3,CrossConv,C3SPP,C3TR,Bottleneck,BottleneckCSP,C3Ghost,
            C3x)
from divide.spatial import SPP, SPPF, Focus
from divide.util import Classify, Concat, Contract, Expand, Proto
from divide.backend import DetectMultiBackend
from divide.attention import CustomAttentionModule
from divide.BIFPN import BiFPN_Feature2, BiFPN_Feature3

fake_common = types.ModuleType("models.common")
fake_common.Conv = Conv
fake_common.C3 = C3
fake_common.CrossConv = CrossConv
fake_common.C3SPP = C3SPP
fake_common.C3TR = C3TR
fake_common.Bottleneck = Bottleneck
fake_common.BottleneckCSP = BottleneckCSP
fake_common.C3Ghost = C3Ghost
fake_common.C3x = C3x
fake_common.DWConv = DWConv
fake_common.DWConvTranspose2d = DWConvTranspose2d
fake_common.GhostBottleneck = GhostBottleneck
fake_common.GhostConv = GhostConv
fake_common.SPP = SPP
fake_common.SPPF = SPPF
fake_common.Focus = Focus
fake_common.Classify = Classify
fake_common.Concat = Concat
fake_common.Contract = Contract
fake_common.Expand = Expand
fake_common.Proto = Proto
fake_common.DetectMultiBackend = DetectMultiBackend
fake_common.CustomAttentionModule = CustomAttentionModule
fake_common.BiFPN_Feature2 = BiFPN_Feature2
fake_common.BiFPN_Feature3 = BiFPN_Feature3

sys.modules["models.common"] = fake_common

ckpt = torch.load("yolov5n.pt", map_location="cpu")
torch.save(ckpt, "yolov5n.pt")
