# faster-rcnn-demo

Faster RCNN pytorch简单复现

运行环境：

- `torch==1.7.1+cu110`
- `torchvision==0.8.2+cu110`

**注：其他版本不知道是否能运行，最新版本的torchvision.ops中的nms等方法无法用GPU训练！**

## 使用方法

1. 安装环境

```shell
pip install -r requirements.txt
```

2. 开启可视化

```shell
python -m visdom.server
```

3. 开始训练

```shell
python train.py
```

4. 自定义参数

见 config.py