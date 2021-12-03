from tqdm import tqdm
from config import cfg
from dataset import ListDataset
from model.faster_rcnn import FasterRCNN
from model.util.eval import Eval
from torch.utils.data import DataLoader
import visdom
import numpy as np
import torch

if __name__ == '__main__':
    # 准备训练与验证数据
    trainset = ListDataset(cfg,split='train_small', is_train=True)
    dataloader = DataLoader(trainset, batch_size=1,shuffle=True, num_workers=0)
    testset = ListDataset(cfg, split='val_small', is_train=False)
    test_dataloader = DataLoader(testset, batch_size=1, num_workers=0, pin_memory=True)
    # 加载模型与权重
    model = FasterRCNN().to(cfg.device)
    # if cfg.load_path:
        # model.load(cfg.load_path)
        # print('已加载训练模型')
    # 创建visdom可视化端口
    vis = visdom.Visdom(env='Faster RCNN')

    for epoch in range(cfg.epoch):
        model.train()
        total_loss = 0
        for index, (img, target_box, target_label, scale) in enumerate(tqdm(dataloader)):
            cur_ep = epoch * len(dataloader) + index + 1
            scale = scale.to(cfg.device)
            img, target_box, target_label = img.to(cfg.device).float(), target_box.to(cfg.device), target_label.to(cfg.device)
            loss = model(img, target_box, target_label, scale)
            total_loss += loss.item()
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            vis.line(X=np.array([cur_ep]), Y=torch.tensor([total_loss/cur_ep]), win='Train Loss',
                    update=None if epoch == 1 else 'append', opts=dict(title='Train Loss'))
            break

        model.eval()
        # 每个Epoch计算一次mAP
        # ap_table = [["Index", "Class name", "Precision", "Recall", "AP", "F1-score"]]
        eval_result = Eval(test_dataloader, model)
        eval_map = eval_result[2].mean()
        print("Epoch %d/%d ---- new-mAP:%.4f" % (epoch, cfg.epoch, eval_map))
        # 绘制mAP和Loss曲线
        vis.line(X=np.array([epoch]), Y=np.array([eval_map]), win='mAP',
               update=None if epoch == 1 else 'append', opts=dict(title='mAP'))
        vis.line(X=np.array([epoch]), Y=torch.tensor([loss['total_loss']]), win='Loss',
                update=None if epoch == 1 else 'append', opts=dict(title='Loss'))