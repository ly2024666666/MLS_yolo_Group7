# 项目结构
- train.py 训练代码
- fix_py.py 将yolov5n预训练模型结构转为兼容本项目框架的模型
- prepare_data.py 对数据集标签作预处理，转为可用格式
- data文件夹：存放数据集配置
- divide文件夹：存放模型及模块
- run文件夹：存放运行结果
- utils文件夹：模型工具像
# 训练指令
## GSConv+CBAM
python train.py --data smoke.yaml --epochs 20 --weights "yolov5n.pt" --cfg yolov5n.yaml --batch-size 4 --device cpu

## BIFPN
python train.py --data smoke.yaml --epochs 20 --weights "yolov5n.pt" --cfg BIFPN.yaml --batch-size 4 --device cpu

## BIFPN+CBAM
python train.py --data smoke.yaml --epochs 20 --weights "yolov5n.pt" --cfg BIFPNAtt.yaml --batch-size 4 --device cpu