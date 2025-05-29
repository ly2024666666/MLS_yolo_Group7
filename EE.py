# Simplified YOLOv5 training script (single-GPU / CPU only)
import argparse
from copy import deepcopy
from pathlib import Path
import torch
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

from models.yolo import Model
from utils.general import (
    LOGGER, check_dataset, check_img_size, check_requirements,
    check_suffix, increment_path, init_seeds,
    intersect_dicts, one_cycle
)
from utils.loss import ComputeLoss
from utils.dataloaders import create_dataloader
from utils.torch_utils import EarlyStopping, smart_optimizer, de_parallel
import val as validate


def train(opt):
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=True))
    (save_dir / 'weights').mkdir(parents=True, exist_ok=True)
    last, best = save_dir / 'weights' / 'last.pt', save_dir / 'weights' / 'best.pt'
    device = torch.device('cpu')
    init_seeds(opt.seed)

    # Load hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)

    # Load dataset
    data_dict = check_dataset(opt.data)
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = int(data_dict['nc'])

    # Load model
    check_suffix(opt.weights, ".pt")
    if opt.weights.endswith('.pt'):
        ckpt = torch.load(opt.weights, map_location='cpu')
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
        state_dict = intersect_dicts(ckpt['model'].float().state_dict(), model.state_dict())
        model.load_state_dict(state_dict, strict=False)
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)



    # Image size and dataloader
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    train_loader, dataset = create_dataloader(train_path, imgsz, opt.batch_size, gs, False, hyp=hyp, augment=True, shuffle=True)
    val_loader = create_dataloader(val_path, imgsz, opt.batch_size, gs, False, hyp=hyp, rect=True)[0]

    nl = de_parallel(model).model[-1].nl
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp

    # Optimizer and scheduler
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    lf = one_cycle(1, hyp['lrf'], opt.epochs) if opt.cos_lr else lambda x: (1 - x / opt.epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Loss function and training setup
    compute_loss = ComputeLoss(model)
    best_fitness = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    stopper = EarlyStopping(patience=opt.patience)

    # Training loop
    for epoch in range(opt.epochs):
        model.train()
        mloss = torch.zeros(3, device=device)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{opt.epochs - 1}")
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            ni = i + len(train_loader) * epoch
            imgs = imgs.to(device).float() / 255
            targets = targets.to(device)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            mloss = (mloss * i + loss_items) / (i + 1)
            pbar.set_postfix(box=mloss[0].item(), obj=mloss[1].item(), cls=mloss[2].item())

        scheduler.step()

        # Validation
        results, maps, _ = validate.run(
            data_dict,
            batch_size=opt.batch_size,
            imgsz=imgsz,
            model=model,
            iou_thres=0.60,
            single_cls=1,
            dataloader=val_loader,
            verbose=True,
            save_dir=save_dir,
            plots=not opt.noplots,
            compute_loss=compute_loss,
        )
        

        fitness_score = results[2]
        if fitness_score > best_fitness:
            best_fitness = fitness_score
            torch.save(deepcopy(de_parallel(model)).half(), best)
        torch.save(deepcopy(de_parallel(model)).half(), last)

    LOGGER.info(f"\nTraining complete. Best mAP@0.5: {best_fitness:.4f}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml')
    parser.add_argument('--data', type=str, default='data/coco128.yaml')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD')
    parser.add_argument('--cos-lr', action='store_true')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--noplots', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    check_requirements()
    train(opt)
