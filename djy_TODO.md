# 修改方式

`models\common.py`以及任意`models\yolov5m.yaml`以及`yolo.py`

# 训练方式

我们的：

```bash
python train.py --data coco128.yaml --epochs 1 --weights "" --cfg yolov5n.yaml --batch-size 4 --device cpu
```

示例的：



# 推理方式：

```bash
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source data/images/bus.jpg --device cpu
```


