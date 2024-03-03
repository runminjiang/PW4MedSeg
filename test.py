import argparse
import os
from functools import partial
import numpy as np
import torch
from networks.unetr import UNETR
from trainer import dice
from utils.data_utils_gau import get_loader
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance
from monai.transforms import Activations, AsDiscrete, Compose

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./runs/btcv_spleen", type=str,
    help="pretrained checkpoint directory"
)
parser.add_argument("--gpu", default=0, type=int, help="The gpu you want to use.")
parser.add_argument("--data_dir", default="./dataset/dataset_0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_spleen_10_200.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="model.pt", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--a_min", default=0.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=255.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")

post_label_0 = AsDiscrete(threshold_values=True)
post_label_1 = AsDiscrete(to_onehot=True, n_classes=2)
post_label = [post_label_0, post_label_1]
post_pred_0 = AsDiscrete(threshold_values=True)
post_pred_1 = AsDiscrete(to_onehot=True, n_classes=2)
post_pred = [post_pred_0, post_pred_1]


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


def main():
    args = parser.parse_args()
    print(args.pretrained_dir)
    device = torch.device(args.gpu)
    args.test_mode = True
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth).cuda(args.gpu)
    elif args.saved_checkpoint == "ckpt":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
        )
        model_dict = torch.load(pretrained_pth, map_location='cpu')
        model.load_state_dict(model_dict["state_dict"])
    model.eval()
    model.cuda(args.gpu)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=(96, 96, 96),
        sw_batch_size=4,
        predictor=model,
        overlap=args.infer_overlap,
    )

    with torch.no_grad():
        dice_list_case = []
        hd_list_case = []

        for i, batch in enumerate(val_loader):
            K = 6
            for k in range(K):
                val_inputs, val_labels = (batch["image"].cuda(device=device), batch["label"].cuda(device=device))
                # print(batch)
                # val_inputs, val_labels = batch.cuda()
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                print("Inference on case {}".format(img_name))
                # val_outputs = sliding_window_inference(val_inputs, 'val', roi_size = (96, 96, 96), sw_batch_size=4, predictor=model, overlap=args.infer_overlap)
                if k == 0:
                    val_outputs = model_inferer(val_inputs, status='val')
                else:
                    val_outputs = val_outputs + model_inferer(val_inputs, status='val')
                # print(val_outputs.shape)
            val_outputs = val_outputs / K
            val_outputs = torch.sigmoid(val_outputs)
            # val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            # --------------------------------hd--------------------------------
            print(val_inputs[0, 0, ...].shape, val_labels.shape)
            val_outputs_hd = post_label[1](post_label[0](val_outputs[0])).unsqueeze(0)
            val_labels_hd = post_pred[1](post_pred[0](val_labels[0])).unsqueeze(0)
            print(val_outputs_hd.shape, val_labels_hd.shape)
            # --------------------------------hd--------------------------------
            val_outputs[val_outputs > 0.5] = 1
            val_outputs[val_outputs <= 0.5] = 0
            # print(val_outputs.shape)
            val_labels = val_labels[:, 0, :, :, :]
            dice_list_sub = []
            hd_list_sub = []
            for i in range(1, 2):
                print((val_outputs[0][0] == i).shape, (val_labels[0] == i).shape)
                organ_Dice = dice(val_outputs[0][0] == i, val_labels[0] == i)
                organ_hd = compute_hausdorff_distance(val_outputs_hd, val_labels_hd, include_background=False,
                                                      percentile=95)
                dice_list_sub.append(organ_Dice.cpu().numpy())
                hd_list_sub.append(organ_hd.cpu().numpy())
            #
            # img_output = val_outputs[0][0].permute(2,0,1).detach().cpu().numpy()
            # sitk_img = sitk.GetImageFromArray(img_output, isVector=False)
            # sitk_img.SetOrigin(numpyOrigin)
            # sitk_img.SetSpacing([1, 1, 1])
            # print(img_output.shape)
            mean_dice = np.mean(dice_list_sub)
            mean_hd = np.mean(hd_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            print("Mean Organ HD95: {}".format(mean_hd))
            dice_list_case.append(mean_dice)
            hd_list_case.append(mean_hd)
            # sitk_img = sitk.GetImageFromArray(val_outputs[0, 0, ...].permute(2,1,0).cpu().numpy(), isVector=False)
            # sitk.WriteImage(sitk_img, os.path.join('/shared/home/v_junhao_wu/local_scratch/btcv/test_output/rk'
            #                                        , 'pre_' + img_name))
            # sitk_img = sitk.GetImageFromArray(val_labels[0, ...].cpu().numpy(), isVector=False)
            # sitk.WriteImage(sitk_img, os.path.join('/shared/home/v_junhao_wu/local_scratch/btcv/test_output/rk'
            #                                        , 'label_' + img_name))
            # sitk_img = sitk.GetImageFromArray(val_inputs[0, 0, ...].cpu().numpy(), isVector=False)
            # sitk.WriteImage(sitk_img,
            #                 os.path.join('/shared/home/v_junhao_wu/local_scratch/btcv/test_output/rk',
            #                              img_name))

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
        print("Overall Mean HD95: {}".format(np.mean(hd_list_case)))
    # --------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
