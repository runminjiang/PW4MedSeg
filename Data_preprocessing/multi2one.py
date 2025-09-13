import argparse
import monai
import numpy as np
import os
import SimpleITK as sitk
from monai.transforms import Spacing
import torch

parser = argparse.ArgumentParser(description='Convert multi-label image to single-label image')
parser.add_argument('--dataset', type=str, required=True, choices=['btcv', 'chaos', 'mscmrseg'],
                    help='Dataset name (btcv, chaos, or mscmrseg)')
parser.add_argument('--organ', type=str, required=True,
                    help='Organ name to extract (spleen, liver, right_kidney, left_kidney, lv_cavity, lv_myocardium, rv_cavity)')
args = parser.parse_args()

dataset = args.dataset
organ = args.organ

# Define label dictionaries for each dataset
if dataset == 'btcv':
    label_dict = {'spleen': 1, 'right_kidney': 2, 'left_kidney': 3, 'liver': 6}
elif dataset == 'chaos':
    label_dict = {'liver': 1, 'right_kidney': 2, 'left_kidney': 3, 'spleen': 4}
elif dataset == 'mscmrseg':
    label_dict = {'lv_cavity': 1, 'lv_myocardium': 2, 'rv_cavity': 3}

# Validate organ choice
if organ not in label_dict:
    available_organs = ', '.join(label_dict.keys())
    parser.error(f"Invalid organ '{organ}' for dataset '{dataset}'. Available organs: {available_organs}")

label_num = label_dict[organ]
multi_labelPath_iso = './dataset/{}/label_iso'.format(dataset)
output_path = './dataset/{}/'.format(dataset)
multi_labels = os.listdir(multi_labelPath_iso)

save_path = os.path.join(output_path, organ, 'labelsTr_gt')
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

print(f"Processing {len(multi_labels)} files to extract {organ} labels...")
for idx, multi_label in enumerate(multi_labels, 1):
    output_file = os.path.join(save_path, multi_label)
    
    # Check if file already exists
    if os.path.exists(output_file):
        print(f"  [{idx}/{len(multi_labels)}] Skipping {multi_label} (already exists)")
        continue
    
    print(f"  [{idx}/{len(multi_labels)}] Processing {multi_label}")
    image_path = os.path.join(multi_labelPath_iso, multi_label)
    sitk_img = sitk.ReadImage(image_path)  # 是个单label单块的文件
    
    numpyImage = sitk.GetArrayFromImage(sitk_img)  # # [z, y, x]
    numpyOrigin = np.array(sitk_img.GetOrigin())  # # [z, y, x]
    numpySpacing = np.array((sitk_img.GetSpacing()))  # # [z, y, x]
    # 需要转化成坐标的形式

    numpyImage[numpyImage != label_num] = 0
    numpyImage[numpyImage == label_num] = 1

    sitk_img = sitk.GetImageFromArray(numpyImage, isVector=False)
    sitk_img.SetOrigin(numpyOrigin)
    sitk_img.SetSpacing([1, 1, 1])
    sitk.WriteImage(sitk_img, output_file)  # 存为nii
