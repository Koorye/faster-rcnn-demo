import numpy as np
import os
import torch
import visdom
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from dataset import ListDataset
from model.faster_rcnn import FasterRCNN
from model.util.eval import Eval

if __name__ == '__main__':
    # 准备训练与验证数据
    trainset = ListDataset(cfg, split='train_small', is_train=True)
    dataloader = DataLoader(trainset, batch_size=1,
                            shuffle=True, num_workers=0)
    testset = ListDataset(cfg, split='val_small', is_train=False)
    test_dataloader = DataLoader(
        testset, batch_size=1, num_workers=0, pin_memory=True)
    # 加载模型与权重
    model = FasterRCNN().to(cfg.device)

    if cfg.load_best:
        paths = os.listdir('weights')
        print(paths)
        if len(paths) != 0:
            best_map = max(list(map(lambda path: float(path.split('_')[1].split('.pt')[0]), paths)))
            model.load(f'weights/map_{best_map}.pt')

    # 创建visdom可视化端口
    vis = visdom.Visdom(env='Faster RCNN')

    best_map = 0

    for epoch in range(cfg.epoch):
        model.train()

        total_loss = 0

        for index, (img, target_box, target_label, scale) in enumerate(tqdm(dataloader)):
            cur_ep = epoch * len(dataloader) + index + 1
            scale = scale.to(cfg.device)
            img, target_box, target_label = img.to(cfg.device).float(
            ), target_box.to(cfg.device), target_label.to(cfg.device)

            loss = model(img, target_box, target_label, scale)
            total_loss += loss.item()
            loss.backward()

            model.optimizer.step()
            model.optimizer.zero_grad()

            vis.line(X=np.array([cur_ep]), Y=torch.tensor([total_loss/cur_ep]), win='Train Loss',
                     update=None if epoch == 1 else 'append', opts=dict(title='Train Loss'))

        model.eval()

        # 每个Epoch计算一次mAP
        # ap_table = [["Index", "Class name", "Precision", "Recall", "AP", "F1-score"]]
        eval_result = Eval(test_dataloader, model)
        eval_map = eval_result[2].mean()

        print("Epoch %d/%d ---- new-mAP:%.4f" % (epoch, cfg.epoch, eval_map))

        # 绘制mAP和Loss曲线
        vis.line(X=np.array([epoch]), Y=np.array([eval_map]), win='mAP',
                 update=None if epoch == 1 else 'append', opts=dict(title='mAP'))

        # 保存最佳模型
        if eval_map > best_map:
           best_map = eval_map
           best_path = model.save(save_path=str(best_map))

        # 调整学习率
        if epoch % 1 == 0:
            model.scale_lr(cfg.lr_decay)
