参数名	默认值	说明
cfg	""	模型结构配置文件路径
hyp	ROOT / "data/hyps/hyp.scratch-low.yaml"	超参数配置文件路径
imgsz	640	图像输入尺寸
rect	False	是否使用矩形训练
resume	False	是否恢复上次训练
nosave	False	是否只保存最终模型
noval	False	是否只在最后做验证
noautoanchor	False	是否禁用自动anchor计算
noplots	False	是否禁用绘图保存
evolve	None	是否进行超参数进化
evolve_population	ROOT / "data/hyps"	进化种群的加载路径
resume_evolve	None	进化恢复路径
bucket	""	GCP存储桶路径
cache	None	是否缓存图片
image_weights	False	是否根据图像权重抽样
device	""	CUDA设备
multi_scale	False	是否使用多尺度训练
single_cls	False	是否将多类当作单类处理
optimizer	"SGD"	使用的优化器
sync_bn	False	是否使用同步BN（DDP专用）
workers	8	数据加载线程数
project	ROOT / "runs/train"	保存路径的项目目录
name	"exp"	实验名
exist_ok	False	是否允许覆盖已有实验文件夹
quad	False	是否使用quad dataloader
cos_lr	False	是否使用余弦学习率调度
label_smoothing	0.0	标签平滑参数
patience	100	EarlyStopping容忍度
freeze	[0]	冻结层的索引
save_period	-1	每几轮保存一次模型
seed	0	全局随机种子
local_rank	-1	DDP用的多GPU参数
entity	None	wandb 的 entity
upload_dataset	False	是否上传数据集
bbox_interval	-1	bbox 可视化间隔
artifact_alias	"latest"	数据集 artifact 的别名
ndjson_console	False	是否打印 NDJSON 到控制台
ndjson_file	False	是否保存 NDJSON 到文件

所以确实，你的训练没有启用任何额外优化机制，比如：

优化项	是否启用	含义/影响
--optimizer AdamW	❌	没有显式设置优化器，默认使用 SGD
--cos-lr	❌	没启用余弦学习率调度
--label-smoothing 0.1	❌	没启用标签平滑
--single-cls	❌	没启用单类处理（用于所有数据只有一个类别时优化 loss）
--multi-scale	❌	没启用图像尺寸多尺度训练（提高泛化能力）
--rect	❌	没启用矩形训练（适合图像尺寸差异大时）
--cache	❌	没缓存图片，加载速度略慢
--sync-bn	❌	没启用多GPU时的同步BN
--freeze	✅（默认）	只冻结了第0层（即不冻结）
--evolve	❌	没进行超参数进化搜索
--patience	✅（默认）	EarlyStopping 容忍100轮，实际20轮训练不到这个阈值
--label-smoothing	✅（默认）	默认值为 0.0，即无标签平滑