# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import distributed_all_gather

from monai.data import decollate_batch
import math
import SimpleITK as sitk


def dice(x, y):
    intersect = torch.sum(torch.sum(torch.sum(x * y)))
    y_sum = torch.sum(torch.sum(torch.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = torch.sum(torch.sum(torch.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    dice_loss = AverageMeter() #########
    nll_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        #print(str(args.gpu))
        data, target = data.cuda(device = args.gpu), target.cuda(args.gpu)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits, nll_losses = model(data)  #到此为止没有做过softmax
            #print(logits.shape)
            #target[target>0.5] = 1
            #target[target<=0.5] = 0
            loss,dice,nll = loss_func(logits, target, nll_losses) #########
            #print(math.isnan(loss))



            if math.isnan(loss):
                print('logitis nan?:',torch.isinf(logits).any())
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            dice_loss.update(dice.item(), n=args.batch_size) #########
            nll_loss.update(nll.item(), n=args.batch_size) #########
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg, dice_loss.avg, nll_loss.avg #########


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    start_time = time.time()
    val_acc = AverageMeter()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(device = args.gpu), target.cuda(device = args.gpu)
            #print('data:',data.shape)
            #print('target:',target.shape)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data,status = 'val')
                    #print(logits)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()

            #target = torch.sigmoid(target)
            logits = torch.sigmoid(logits)

            

            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label[1](post_label[0](val_label_tensor)) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred[1](post_pred[0](val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            #print(post_pred[1](post_pred[0](val_outputs_list[0])).shape,val_labels_convert.shape)
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            #print(dice(val_labels_convert[0][1],val_output_convert[0][1]),torch.count_nonzero(val_labels_convert[0][0]),val_labels_convert[0][0].shape)
            acc = acc.cuda(args.rank)
            #print(acc)
            #sitk_img = sitk.GetImageFromArray(val_labels_convert[0][1].cpu().numpy(), isVector=False)
    
            #sitk.WriteImage(sitk_img, os.path.join('/local/scratch/v_jiayin_sun/UNETR/BTCV/dataset/dataset_0/labelsTr_thres','fuck.nii.gz'))


            if args.distributed:
                acc_list = distributed_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            else:
                acc_list = acc.detach().cpu().numpy()
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
                #print(avg_acc)
            val_acc.update(avg_acc)
            start_time = time.time()
            #print(val_acc.avg)
    return val_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
    wandb_writer = None
):
    writer = None
    if args.logdir is not None and args.rank == 0:    #默认test
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()  
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss, dice_loss, nll_loss = train_epoch(    #训练函数
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            wandb_writer.log({'loss': train_loss, 'epoch': epoch, 'diceloss': dice_loss, 'nllloss': nll_loss})
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:  #默认每100个epoch做一次validation
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                wandb_writer.log({'val_avg_acc': val_avg_acc, 'epoch': epoch})
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
