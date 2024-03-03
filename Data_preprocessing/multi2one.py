import monai
import numpy as np
import os
import SimpleITK as sitk
from monai.transforms import Spacing
import torch

dataset = 'btcv'  # btcv or chaos
multi_labelPath_iso = './dataset/{}/label_iso'.format(dataset)
output_path = './dataset/{}/'.format(dataset)
multi_labels = os.listdir(multi_labelPath_iso)
label_num = 1  # 指定哪个器官

# 设置保存路径，可自定义
if dataset == 'btcv':
    label_dict = {1: 'spleen', 2: 'right_kidney', 3: 'left_kidney', 6: 'liver'}
elif dataset == 'chaos':
    label_dict = {1: 'liver', 2: 'right_kidney', 3: 'left_kidney', 4: 'spleen'}
else:
    label_dict = {}
    print('please set label_dict')

for multi_label in multi_labels:
    image_path = os.path.join(multi_labelPath_iso, multi_label)
    sitk_img = sitk.ReadImage(image_path)  # 是个单label单块的文件
    # print("img shape:", itk_img.shape)
    ''' return order [z, y, x] , numpyImage, numpyOrigin, numpySpacing '''
    # 医学图像处理 空间转换的比例，一个像素代表多少毫米。 保证前后属性一致
    numpyImage = sitk.GetArrayFromImage(sitk_img)  # # [z, y, x]
    numpyOrigin = np.array(sitk_img.GetOrigin())  # # [z, y, x]
    numpySpacing = np.array((sitk_img.GetSpacing()))  # # [z, y, x]
    print(numpySpacing, numpyImage.shape)  ## numpy Image 是只有0和1(代表图片里这个像素点是黑还是白)
    # 需要转化成坐标的形式

    numpyImage[numpyImage != label_num] = 0
    numpyImage[numpyImage == label_num] = 1

    sitk_img = sitk.GetImageFromArray(numpyImage, isVector=False)
    sitk_img.SetOrigin(numpyOrigin)
    sitk_img.SetSpacing([1, 1, 1])
    print(numpySpacing)
    save_path = os.path.join(output_path,label_dict[label_num],'labelsTr_gt')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    sitk.WriteImage(sitk_img, save_path + '/{}.nii.gz'.format(multi_label))  # 存为nii
