import argparse
import monai
import numpy as np
import os
import SimpleITK as sitk
from monai.transforms import Spacing
import torch

parser = argparse.ArgumentParser(description='Convert anisotropic images to isotropic')
parser.add_argument('--dataset', type=str, required=True, choices=['btcv', 'chaos'],
                    help='Dataset name (btcv or chaos)')
args = parser.parse_args()

dataset = args.dataset
multi_labelPath_ori = './dataset/{}/label'.format(dataset)
multi_imagePath_ori = './dataset/{}/image'.format(dataset)
multi_labelPath_iso = './dataset/{}/label_iso'.format(dataset)
multi_imagePath_iso = './dataset/{}/image_iso'.format(dataset)
multi_labels = os.listdir(multi_labelPath_ori)
multi_images = os.listdir(multi_imagePath_ori)
os.makedirs(multi_imagePath_iso, exist_ok=True)
os.makedirs(multi_labelPath_iso, exist_ok=True)

print(f"Processing {len(multi_labels)} labels...")
for idx, multi_label in enumerate(multi_labels, 1):
    output_path = os.path.join(multi_labelPath_iso, multi_label)
    if os.path.exists(output_path):
        print(f"Skipping label {idx}/{len(multi_labels)}: {multi_label} (already exists)")
        continue
    print(f"Processing label {idx}/{len(multi_labels)}: {multi_label}")
    image_path = os.path.join(multi_labelPath_ori, multi_label)
    sitk_img = sitk.ReadImage(image_path)  # 是个单label单块的文件

    ''' return order [z, y, x] , numpyImage, numpyOrigin, numpySpacing '''
    # 医学图像处理 空间转换的比例，一个像素代表多少毫米。 保证前后属性一致
    numpyImage = sitk.GetArrayFromImage(sitk_img)  # # [z, y, x]
    numpyOrigin = np.array(sitk_img.GetOrigin())  # # [z, y, x]
    numpySpacing = np.array((sitk_img.GetSpacing()))  # # [z, y, x]
    # Transform to isotropic spacing

    spacing_transform = Spacing(pixdim=(1 / numpySpacing[2], 1 / numpySpacing[0], 1 / numpySpacing[1]), mode="nearest")
    numpyImage = spacing_transform(torch.tensor(np.expand_dims(numpyImage, axis=0)))[0][0]

    sitk_img = sitk.GetImageFromArray(numpyImage, isVector=False)
    sitk_img.SetOrigin(numpyOrigin)
    sitk_img.SetSpacing([1, 1, 1])
    sitk.WriteImage(sitk_img, multi_labelPath_iso + '/{}.nii.gz'.format(multi_label))  # 存为nii

print(f"Processing {len(multi_images)} images...")
for idx, multi_image in enumerate(multi_images, 1):
    output_path = os.path.join(multi_imagePath_iso, multi_image)
    if os.path.exists(output_path):
        print(f"Skipping image {idx}/{len(multi_images)}: {multi_image} (already exists)")
        continue
    print(f"Processing image {idx}/{len(multi_images)}: {multi_image}")
    image_path = os.path.join(multi_imagePath_ori, multi_image)
    sitk_img = sitk.ReadImage(image_path)  # 是个单label单块的文件
    # print("img shape:", itk_img.shape)
    ''' return order [z, y, x] , numpyImage, numpyOrigin, numpySpacing '''
    # 医学图像处理 空间转换的比例，一个像素代表多少毫米。 保证前后属性一致
    numpyImage = sitk.GetArrayFromImage(sitk_img)  # # [z, y, x]
    numpyOrigin = np.array(sitk_img.GetOrigin())  # # [z, y, x]
    numpySpacing = np.array((sitk_img.GetSpacing()))  # # [z, y, x]

    # Convert to float32 to avoid dtype issues with PyTorch
    numpyImage = numpyImage.astype(np.float32)
    
    spacing_transform = Spacing(pixdim=(1 / numpySpacing[2], 1 / numpySpacing[0], 1 / numpySpacing[1]), mode="bilinear")
    numpyImage = spacing_transform(torch.tensor(np.expand_dims(numpyImage, axis=0)))[0][0]

    sitk_img = sitk.GetImageFromArray(numpyImage, isVector=False)
    sitk_img.SetOrigin(numpyOrigin)
    sitk_img.SetSpacing([1, 1, 1])
    sitk.WriteImage(sitk_img, multi_imagePath_iso + '/{}.nii.gz'.format(multi_image))  # 存为nii
