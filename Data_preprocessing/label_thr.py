import argparse
import os
import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser(description='Apply threshold to pseudo labels')
parser.add_argument('--dataset', type=str, required=True, choices=['btcv', 'chaos', 'mscmrseg'],
                    help='Dataset name (btcv, chaos, or mscmrseg)')
parser.add_argument('--organ', type=str, required=True,
                    help='Organ name (spleen, liver, right_kidney, left_kidney, lv_cavity, lv_myocardium, rv_cavity)')
parser.add_argument('--points', type=int, default=200,
                    help='Number of points used in point2gau (default: 200)')
parser.add_argument('--delta', type=int, default=10,
                    help='Delta value used in point2gau (default: 10)')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Threshold value for pseudo labels (default: 0.5)')
args = parser.parse_args()

# Validate organ choice based on dataset
if args.dataset == 'btcv':
    valid_organs = ['spleen', 'right_kidney', 'left_kidney', 'liver']
elif args.dataset == 'chaos':
    valid_organs = ['liver', 'right_kidney', 'left_kidney', 'spleen']
elif args.dataset == 'mscmrseg':
    valid_organs = ['lv_cavity', 'lv_myocardium', 'rv_cavity']

if args.organ not in valid_organs:
    parser.error(f"Invalid organ '{args.organ}' for dataset '{args.dataset}'. Valid organs: {', '.join(valid_organs)}")

dataset = args.dataset
organ = args.organ
points = args.points
delta = args.delta
threshold = args.threshold

ori_path = f'./dataset/{dataset}/{organ}/p_{points}_d_{delta}/labelsTr'
thr_path = f'./dataset/{dataset}/{organ}/p_{points}_d_{delta}/labelsTr_thres'
if not os.path.exists(thr_path):
    os.makedirs(thr_path)
files = os.listdir(ori_path)

print(f"Applying threshold {threshold} to {len(files)} pseudo-labels...")
for idx, i in enumerate(files, 1):
    output_file = os.path.join(thr_path, i)
    
    # Check if thresholded file already exists
    if os.path.exists(output_file):
        print(f"  [{idx}/{len(files)}] Skipping {i} (already exists)")
        continue
        
    label_file = os.path.join(ori_path, i)
    print(f"  [{idx}/{len(files)}] Processing {i}")
    label_img = nib.load(label_file)
    label_data = label_img.get_fdata()

    # 设置阈值
    thresholded_data = np.where(label_data > threshold, label_data, 0)

    # 创建新的NIfTI图像对象
    thresholded_img = nib.Nifti1Image(thresholded_data, label_img.affine, label_img.header)

    # 保存阈值化后的标签
    nib.save(thresholded_img, output_file)
