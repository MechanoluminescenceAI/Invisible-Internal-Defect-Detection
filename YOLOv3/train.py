from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/artifacts/images", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="config/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="config/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.3, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)
# model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
record_path1=os.path.join(r'D:\yolov3_pytorch_on_backside_crack','record_data_file.csv')
record_df1 = pd.DataFrame()

for epoch in range(opt.epochs):
    loss_batch_list = []
    loss_x_batch_list = []
    loss_y_batch_list = []
    loss_w_batch_list = []
    loss_h_batch_list = []
    loss_conf_batch_list = []
    loss_cls_batch_list = []
    loss_recall_batch_list = []
    loss_precision_batch_list = []
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()
        loss_batch_list.append(loss.item())
        loss_x_batch_list.append(model.losses["x"])
        loss_y_batch_list.append(model.losses["y"])
        loss_w_batch_list.append(model.losses["w"])
        loss_h_batch_list.append(model.losses["h"])
        loss_conf_batch_list.append(model.losses["conf"])
        loss_cls_batch_list.append(model.losses["cls"])
        loss_recall_batch_list.append(model.losses["recall"])
        loss_precision_batch_list.append(model.losses["precision"])

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

    total_loss = np.round(np.mean(loss_batch_list), 5)
    total_loss_x = np.round(np.mean(loss_x_batch_list), 5)
    total_loss_y = np.round(np.mean(loss_y_batch_list), 5)
    total_loss_w = np.round(np.mean(loss_w_batch_list), 5)
    total_loss_h = np.round(np.mean(loss_h_batch_list), 5)
    total_loss_conf = np.round(np.mean(loss_conf_batch_list), 5)
    total_loss_cls = np.round(np.mean(loss_cls_batch_list), 5)
    total_loss_recall = np.round(np.mean(loss_recall_batch_list), 5)
    total_loss_precision = np.round(np.mean(loss_precision_batch_list), 5)
    new_row1 = {'Epoch': epoch, 'x': total_loss_x, 'y': total_loss_y, 'w': total_loss_w, 'h': total_loss_h,
                'conf': total_loss_conf, 'cls': total_loss_cls, 'recall': total_loss_recall,
                'precision': total_loss_precision, 'Total_loss': total_loss}

    record_df1 = record_df1.append(new_row1, ignore_index=True)
    record_df1.to_csv(record_path1, index=0)
    print("[Epoch %d, Total_loss %d " % (epoch, total_loss))

    if epoch % opt.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))


