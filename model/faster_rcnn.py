import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16
from config import cfg

from model.roi_head import RoIHead
from model.rpn import RPN
from model.util.anchor_target_creator import AnchorTargetCreator
from model.util.box_tool import loc2box
from model.util.loss import fast_rcnn_loc_loss
from model.util.proposal_target_creator import ProposalTargetCreator

def decom_vgg16():
    """
    截取VGG16模型，获取提取器和分类器
    """
    model = vgg16(pretrained=True)
    # 截取vgg16的前30层网络结构,因为再往后的就不需要
    # 31层为maxpool再往后就是fc层
    features = list(model.features)[:30]
    classifier = model.classifier
    classifier = list(classifier)
    # 删除最后一层以及两个dropout层
    del classifier[6]
    del classifier[5]
    del classifier[2]
    classifier = nn.Sequential(*classifier)

    # 冻结前4层的卷积层
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    features = nn.Sequential(*features)

    return features, classifier

class FasterRCNN(nn.Module):
    def __init__(self) -> None:
        super(FasterRCNN,self).__init__()
        self.n_class = len(cfg.classes) + 1

        # 获取特征提取网络和分类网络
        self.extractor, self.classifier = decom_vgg16()

        self.head = RoIHead(n_class=self.n_class, classifier=self.classifier).to(cfg.device)
        self.rpn = RPN().to(cfg.device)

        self.nms_thresh = cfg.nms_roi
        self.rpn_sigma = cfg.rpn_sigma
        self.roi_sigma = cfg.roi_sigma

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.optimizer = self.get_optim()

        self.mean = torch.Tensor((0.,0.,0.,0.), device=cfg.device).repeat(self.n_class)[None]
        self.std = torch.Tensor((.1,.1,.2,.2), device=cfg.device).repeat(self.n_class)[None]

        self.score_thresh = .05
    
    def forward(self, x, target_boxes=None, target_labels=None, scale=1.):
        """
        : param x: 原图
        : param target_boxes: 目标框
        : param target_labels: 目标框类别
        : param scale: 缩放
        """

        img_size = x.shape[2:]
        # 提取特征图
        features = self.extractor(x)

        # 经过RPN提取回归值和分数，并返回NMS后的预测框
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size, scale)

        # 非训练阶段，直接返回ROI
        if not self.training:
            roi_locs, roi_scores = self.head(features, rois)
            return roi_locs, roi_scores, rois

        # batch为1，故只取第一个元素
        target_box = target_boxes[0]
        target_label = target_labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois        

        sample_roi, gt_head_loc, gt_head_label = self.proposal_target_creator(roi, target_box, target_label)        
        head_loc, head_score = self.head(features, sample_roi)

        # ------------------ 计算 RPN losses -------------------#
        # 开始计算RPN网络的定位损失
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(target_box, anchor, img_size)
        # 这里使用long类型因为下面cross_entropy方法需要
        gt_rpn_label = gt_rpn_label.long()
        rpn_loc_loss = fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
        # 开始计算RPN网络的分类损失,忽略那些label为-1的
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.to(cfg.device), ignore_index=-1)

        # ------------------计算 ROI_head losses -------------------#
        # 开始计算ROI_head网络的定位损失
        n_sample = head_loc.shape[0]
        head_loc = head_loc.reshape(n_sample, -1, 4)  # torch.Size([128, self.n_class, 4])
        # 该一步主要是获取sample_roi中每个roi所对应的修正系数loc.当然,正样本和负样本所获取的loc情况是不同的
        # 正样本:某个roi中类别概率最大的那个类别的loc;负样本:永远是第1个loc(背景类 index为0)
        gt_head_label = gt_head_label.long()
        head_loc = head_loc[torch.arange(n_sample).long().to(cfg.device), gt_head_label]
        # 开始计算ROI_head网络的定位与分类损失
        roi_loc_loss = fast_rcnn_loc_loss(head_loc, gt_head_loc, gt_head_label, self.roi_sigma)
        roi_cls_loss = F.cross_entropy(head_score, gt_head_label.to(cfg.device))
        losses = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

        return losses    


    @torch.no_grad()
    def predict(self, imgs,sizes=None):
        """
        该方法在非训练阶段的时候使用
        :param imgs: 一个batch的图片
        :param sizes: batch中每张图片的输入尺寸
        :return: 返回所有一个batch中所有图片的坐标,类,类概率值 三个值都是list型数据,里面包含的是numpy数据
        """
        boxes = list()
        labels = list()
        scores = list()
        # 因为batch_size为1所以这个循环就只循环一次
        for img, size in zip([imgs], [sizes]):
            scale = img.shape[3] / size[1]
            # (300, self.n_class*4) (300, self.n_class) (300, 4) 理论上是这样的数据,有时候可能会小于300
            roi_locs, roi_scores, roi = self(img, scale=scale)
            # chenyun版本的代码中是有对训练阶段的roi_locs进行归一化的,然后再在非训练状态下进行逆向归一化
            roi_locs = (roi_locs * self.std + self.mean)  # 减均值除以方差的逆过程

            roi_locs = roi_locs.view(-1, self.n_class, 4)  # [300, self.n_class*4] -> [300, self.n_class, 4]
            roi = roi.view(-1, 1, 4).expand_as(roi_locs)   # [300, 1, 4] -> [300, self.n_class, 4]
            # 将坐标放缩会原始尺寸 chenyun版本是将缩放这一步放到修正坐标之前,我觉得不太合理,就移到修正之后了.精度没变
            pred_boxes = loc2box(roi.reshape(-1, 4),roi_locs.reshape(-1, 4)) / scale
            pred_boxes = pred_boxes  # torch.Size([5700, 4])
            pred_boxes = pred_boxes.view(-1, self.n_class, 4)   # (300*self.n_class, 4) -> (300, self.n_class, 4)
            # 限制预测框的坐标范围
            pred_boxes[:,:, 0::2].clamp_(min=0, max=size[0])
            pred_boxes[:,:, 1::2].clamp_(min=0, max=size[1])
            # 对roi_head网络预测的每类进行softmax处理
            pred_scores = F.softmax(roi_scores, dim=1)
            
            # 每张图片的预测结果(m为预测目标的个数)     # (m, 4)  (m,)  (m,) 跳过cls_id为0的pred_bbox与pred_scores,因为它是背景类
            pred_boxes, pred_label, pred_score = self._suppress(pred_boxes[:,1:,:], pred_scores[:,1:])
            boxes.append(pred_boxes)
            #   [array([[302.97562, 454.60007, 389.80545, 504.98404],
            #           [304.9767 , 550.0696 , 422.17258, 620.1692 ],
            #           [375.89203, 540.1559 , 422.39435, 684.8439 ],
            #           [293.0167, 349.53333, 360.0981, 386.8974]], dtype = float32)]
            labels.append(pred_label)
            #   [array([ 0,  0,  15, 15])]
            scores.append(pred_score)
            #   [array([0.80108094, 0.80108094, 0.80108094, 0.80108094], dtype=float32)]
        return boxes, labels, scores
    
    def get_optim(self):
        # 获取梯度更新的方式,以及 放大 对网络权重中 偏置项 的学习率
        lr = cfg.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.weight_decay}]
        if cfg.use_sgd:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(params)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

