import os

import nibabel as nib
import numpy as np

dataset = 'btcv'  # btcv or chaos
organ = 'spleen'

ori_path = f'/dataset/{dataset}/{organ}/p_200_d_10/labelsTr'
thr_path = f'/dataset/{dataset}/{organ}/p_200_d_10/labelsTr_thres'
if not os.path.exists(thr_path):
    os.makedirs(thr_path)
files = os.listdir(ori_path)
for i in files:
    label_file = os.path.join(ori_path, i)
    print(label_file)
    label_img = nib.load(label_file)
    label_data = label_img.get_fdata()

    # 设置阈值
    threshold = 0.5  # 假设阈值为0.5
    thresholded_data = np.where(label_data > threshold, label_data, 0)

    # 创建新的NIfTI图像对象
    thresholded_img = nib.Nifti1Image(thresholded_data, label_img.affine, label_img.header)

    # 保存阈值化后的标签
    output_file = os.path.join(thr_path, i)
    nib.save(thresholded_img, output_file)
