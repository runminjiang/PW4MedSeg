import wandb
import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from networks.unetr import UNETR
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training

from utils.data_utils_gau import get_loader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from utils.loss import Weighted_Focal_DiceLoss, my_DiceLoss, CE_DiceLoss, CE_Loss, my_DiceLoss_prob, Focal_Loss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

# 训练python脚本中import torch后，加上下面这句。
torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument("--gpu", default=0, type=int, help="The gpu you want to use.")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default='btcv_spleen', type=str,
                    help="directory to save the tensorboard logs")  # 这个参数每次跑的时候标注一下实验编号
parser.add_argument(
    "--pretrained_dir", default="./runs", type=str,
    help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="dataset/dataset_0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_spleen_10_200.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="model.pt", type=str, help="pretrained model name"
)
parser.add_argument("--stdvar", default=1, type=float, help="standdard variation for kld loss")
parser.add_argument("--w_nll", default=0.3, help="nll loss weight")
parser.add_argument("--save_checkpoint", default='True', help="save checkpoint during training")
parser.add_argument("--max_epochs", default=3500, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-4, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=500, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--model_name", default="probabilistic_unetr_gaussian_mask_newdata", type=str, help="model name")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")  # 原来是14，现在改为1
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=0.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=255.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1, type=float, help="spacing in x direction")  # 和论文里说的isotropic不一样
parser.add_argument("--space_y", default=1, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    wandb_train = wandb.init(project=args.model_name, name=args.logdir)
    args.logdir = "./runs/" + args.logdir
    main_worker(gpu=args.gpu, args=args, wandb_writer=wandb_train)


def main_worker(gpu, args, wandb_writer):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir
    if (args.model_name is not None):
        model = UNETR(
            in_channels=args.in_channels,  # 1
            out_channels=args.out_channels,  # 1
            img_size=(args.roi_x, args.roi_y, args.roi_z),  # 96 96 96
            feature_size=args.feature_size,  # 16
            hidden_size=args.hidden_size,  # 768
            mlp_dim=args.mlp_dim,  # 3072
            num_heads=args.num_heads,  # 12
            pos_embed=args.pos_embed,  # perception
            norm_name=args.norm_name,  # instance
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,  # 0.0
            stdvar=args.stdvar,
        )

        if args.resume_ckpt:  # store true
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            model.load_state_dict(model_dict["state_dict"])
            print("Use pretrained weights")

        if args.resume_jit:  # store true
            if not args.noamp:
                print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    else:
        raise ValueError("Unsupported model " + str(args.model_name))

    # dice_loss = Weighted_Focal_DiceLoss(squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,
    #                                     w_nll=args.w_nll)
    # dice_loss = CE_DiceLoss(squared_pred = True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,w_nll = args.w_nll)
    # dice_loss = my_DiceLoss( squared_pred = True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr)
    # dice_loss = my_DiceLoss_prob( squared_pred = True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr)
    # dice_loss = CE_Loss(squared_pred = True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,w_nll = args.w_nll)
    dice_loss = Focal_Loss(squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr, w_nll=args.w_nll)
    # dice_loss = slice_Loss(squared_pred = True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,w_nll = args.w_nll)
    # post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)  #Execute after model forward to transform model output to discrete values
    # post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.out_channels)
    post_label_0 = AsDiscrete(threshold_values=True)
    post_label_1 = AsDiscrete(to_onehot=True, n_classes=2)
    post_label = [post_label_0, post_label_1]
    post_pred_0 = AsDiscrete(threshold_values=True)
    post_pred_1 = AsDiscrete(to_onehot=True, n_classes=2)
    post_pred = [post_pred_0, post_pred_1]
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
        wandb_writer=wandb_writer
    )
    return accuracy


if __name__ == "__main__":
    main()
