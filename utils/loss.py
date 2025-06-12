# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou, box_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

class ATSSAssigner:
    """
    ATSS正负样本分配器
    """
    def __init__(self, top_k=9):
        self.top_k = top_k

    def assign(self, anchors, gt_boxes):
        # anchors: [num_anchors, 2] (w, h)
        # gt_boxes: [num_gt, 4] (cx, cy, w, h) 归一化到特征图尺度
        num_anchors = anchors.size(0)
        num_gt = gt_boxes.size(0)
        if num_gt == 0:
            return torch.zeros(num_anchors, dtype=torch.bool, device=anchors.device)
        # 计算IoU
        gt_boxes_xyxy = torch.cat([
            gt_boxes[:, :2] - gt_boxes[:, 2:] / 2,
            gt_boxes[:, :2] + gt_boxes[:, 2:] / 2
        ], dim=1)  # [num_gt, 4]
        anchors_xyxy = torch.cat([
            anchors[:, :2] - anchors[:, 2:] / 2,
            anchors[:, :2] + anchors[:, 2:] / 2
        ], dim=1)  # [num_anchors, 4]
        ious = box_iou(anchors_xyxy, gt_boxes_xyxy)  # [num_anchors, num_gt]
        # 对每个gt，选top_k个IoU最大的anchor为正样本
        is_pos = torch.zeros_like(ious, dtype=torch.bool)
        for gt_idx in range(num_gt):
            topk = min(self.top_k, num_anchors)
            _, topk_idxs = ious[:, gt_idx].topk(topk, largest=True)
            is_pos[topk_idxs, gt_idx] = True
        # 合并所有gt的正样本
        pos_mask = is_pos.any(dim=1)
        return pos_mask

class SimOTAAssigner:
    """
    SimOTA正负样本分配器（YOLOX核心）
    """
    def __init__(self, center_radius=2.5):
        self.center_radius = center_radius

    def assign(self, anchors, gt_boxes, gt_classes, pred_cls, pred_box):
        # anchors: [num_anchors, 2] (cx, cy)
        # gt_boxes: [num_gt, 4] (cx, cy, w, h)
        # pred_cls: [num_anchors, num_classes] (logits)
        # pred_box: [num_anchors, 4] (cx, cy, w, h)
        num_anchors = anchors.size(0)
        num_gt = gt_boxes.size(0)
        device = anchors.device
        if num_gt == 0:
            return torch.zeros(num_anchors, dtype=torch.bool, device=device), torch.full((num_anchors,), -1, device=device)
        # 计算每个anchor与gt的IoU
        gt_boxes_xyxy = torch.cat([
            gt_boxes[:, :2] - gt_boxes[:, 2:] / 2,
            gt_boxes[:, :2] + gt_boxes[:, 2:] / 2
        ], dim=1)  # [num_gt, 4]
        pred_boxes_xyxy = torch.cat([
            pred_box[:, :2] - pred_box[:, 2:] / 2,
            pred_box[:, :2] + pred_box[:, 2:] / 2
        ], dim=1)  # [num_anchors, 4]
        ious = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)  # [num_anchors, num_gt]
        # 分类分支分数
        cls_prob = pred_cls.sigmoid()  # [num_anchors, num_classes]
        gt_cls_onehot = torch.zeros((num_gt, cls_prob.size(1)), device=device)
        gt_cls_onehot[torch.arange(num_gt), gt_classes] = 1.0
        cls_cost = -torch.matmul(cls_prob, gt_cls_onehot.t())  # [num_anchors, num_gt]
        # IoU cost
        iou_cost = -ious
        # 总cost
        cost = cls_cost + 3.0 * iou_cost
        # 动态k
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        topk_ious, _ = ious.topk(min(10, ious.size(0)), dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[pos_idx, gt_idx] = 1
        # 每个anchor只分配给一个gt
        anchor_matching_gt = matching_matrix.sum(1)
        if (anchor_matching_gt > 1).sum() > 0:
            multiple_match_idx = torch.where(anchor_matching_gt > 1)[0]
            for idx in multiple_match_idx:
                gt_idx = cost[idx].argmin()
                matching_matrix[idx] = 0
                matching_matrix[idx, gt_idx] = 1
        assigned_gt = matching_matrix.argmax(1)
        pos_mask = matching_matrix.sum(1).bool()
        return pos_mask, assigned_gt


class ComputeLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False, iou_type='CIoU',Focal=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.iou_type = iou_type
        self.Focal = Focal

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                
                # 根据配置选择不同的IoU损失函数
                # if self.iou_type == 'SIoU':
                #     iou = bbox_iou(pbox, tbox[i], SIoU=True, alpha=1).squeeze()  # SIoU损失，alpha=1
                # elif self.iou_type == 'alpha-SIoU':
                #     iou = bbox_iou(pbox, tbox[i], SIoU=True, alpha=3).squeeze()  # alpha-SIoU损失，alpha=3
                # elif self.iou_type == 'DIoU':
                #     iou = bbox_iou(pbox, tbox[i], DIoU=True).squeeze()
                # elif self.iou_type == 'CIoU':
                #     iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                # elif self.iou_type == 'GIoU':
                #     iou = bbox_iou(pbox, tbox[i], GIoU=True).squeeze()
                # elif self.iou_type == 'EIoU':
                #     iou = bbox_iou(pbox, tbox[i], EIoU=True).squeeze()
                # else:  # IoU (default)
                #     iou = bbox_iou(pbox, tbox[i]).squeeze()  # iou(prediction, target)
                if self.iou_type == 'SIoU':
                    iou = bbox_iou(pbox, tbox[i], lossname=self.iou_type, alpha=1).squeeze()  # SIoU损失，alpha=1
                elif self.iou_type == 'alpha-SIoU':
                    iou = bbox_iou(pbox, tbox[i], lossname=self.iou_type, alpha=3).squeeze()  # alpha-SIoU损失，alpha=3
                else:
                    iou = bbox_iou(pbox, tbox[i], lossname=self.iou_type).squeeze()
                
                    
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch