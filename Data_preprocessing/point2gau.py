# from __future__ import print_function
import torch
from torch.autograd import Variable
import os
# from __future__ import print_function
import torch
from torch.autograd import Variable
import SimpleITK as sitk
import numpy as np
import pydicom
import os
import random
import cv2

# ---------------------------------------------------------
points = 200
delta = 10
dataset = 'btcv'  # btcv or chaos
organ = 'liver'
# ---------------------------------------------------------


one_labelpath = f"./dataset/{dataset}/{organ}/labelsTr_gt"
rootpath = f'./dataset/{dataset}/{organ}/p_{points}_d_{delta}'
if not os.path.exists(rootpath):
    os.mkdir(rootpath)
point_num = points


def farthest_point_sample(xyz, pnum):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    # xyz = xyz.transpose(2,1)
    # print('xyz',xyz)

    device = xyz.device
    B, N, C = xyz.shape
    # print(B,N,C)
    npoint = int(pnum)
    # print(npoint)

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # 采样点矩阵（B, npoint）
    distance = torch.ones(B, N).to(device) * 1e10  # 采样点到所有点距离（B, N）
    # print(distance,distance.shape)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # batch_size 数组
    # print(batch_indices)
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 初始时随机选择一点

    barycenter = torch.sum((xyz), 1)  # 计算重心坐标 及 距离重心最远的点
    # print(barycenter)
    barycenter = barycenter / xyz.shape[1]
    # print(barycenter.shape)
    barycenter = barycenter.view(B, 1, C)
    # print(barycenter)
    # print('xyz-barycenter',xyz-barycenter)
    dist = torch.sum((xyz - barycenter) ** 2, -1)
    # print('dist',dist)
    farthest = torch.max(dist, 1)[1]  # 将距离重心最远的点作为第一个点
    # print('farthest',farthest)
    # print(torch.max(dist,1))
    for i in range(npoint):
        # print("-------------------------------------------------------")
        # print("The %d farthest pts %s " % (i, farthest))
        centroids[:, i] = farthest  # 更新第i个最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)  # 取出这个最远点的xyz坐标
        dist = torch.sum((xyz - centroid) ** 2, -1)  # 计算点集中的所有点到这个最远点的欧式距离
        # print("dist    : ", dist)
        mask = dist < distance
        # print("mask %i : %s" % (i,mask))
        distance[mask] = dist[mask].float()  # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离
        # print("distance: ", distance)

        farthest = torch.max(distance, -1)[1]  # 返回最远点索引

    return centroids


one_labels = os.listdir(one_labelpath)

for one_label in one_labels:
    image_path = os.path.join(one_labelpath, one_label)
    sitk_img = sitk.ReadImage(image_path)  # 是个单label单块的文件
    # print("img shape:", itk_img.shape)
    ''' return order [z, y, x] , numpyImage, numpyOrigin, numpySpacing '''
    # 医学图像处理 空间转换的比例，一个像素代表多少毫米。 保证前后属性一致
    numpyImage = sitk.GetArrayFromImage(sitk_img)  # # [z, y, x]
    numpyOrigin = sitk_img.GetOrigin()  # # [z, y, x]
    numpySpacing = sitk_img.GetSpacing()  # # [z, y, x]
    print(numpyImage.shape)  ## numpy Image 是只有0和1(代表图片里这个像素点是黑还是白)
    # 需要转化成坐标的形式

    points = np.where(numpyImage == 1)

    points_to_be_sampled = []
    for i in range(len(points[0])):
        z = points[0][i]
        y = points[1][i]
        x = points[2][i]
        points_to_be_sampled.append([x, y, z])
    points_to_be_sampled = random.sample(points_to_be_sampled, int(len(points_to_be_sampled) * 0.5))
    points_to_be_sampled = torch.tensor(points_to_be_sampled)

    # sim_data = Variable(torch.rand(1,3,8))
    points_to_be_sampled = points_to_be_sampled.unsqueeze(0)
    sim_data = Variable(points_to_be_sampled)
    print(points_to_be_sampled.shape)

    centroids = farthest_point_sample(sim_data, point_num)
    centroid = sim_data[0, centroids, :][0]

    point_labeled_image = np.zeros_like(numpyImage)
    for cord in centroid:
        x, y, z = cord
        print(x, y, z)
        point_labeled_image[z, y, x] = 1

    sitk_img = sitk.GetImageFromArray(point_labeled_image, isVector=False)
    sitk_img.SetOrigin(numpyOrigin)
    sitk_img.SetSpacing(numpySpacing)
    point_label_path = os.path.join(rootpath, 'point_label')
    if not os.path.exists(point_label_path):
        os.mkdir(point_label_path)
    sitk.WriteImage(sitk_img, point_label_path + '/{}.nii.gz'.format(one_label))  # 存为nii
    # print('save file {}'.format(one_label))


def point2GAU(img_name, root_path, save_name, delta):
    # img_path: 采样点mask位置 H:\CMU\weaklysupervised\point_label\xxx.nii
    # save_path: 保存mask位置 H:\CMU\weaklysupervised
    # 输出：H:\CMU\weaklysupervised\gaussianmask\xxx.nii; H:\CMU\weaklysupervised\gaussian_thresmask\xxx.nii

    one_label_file = os.path.join(root_path, 'point_label', img_name)
    one_label = sitk.ReadImage(one_label_file)
    numpyImage = sitk.GetArrayFromImage(one_label)  # # [z, y, x]
    numpyOrigin = np.array(one_label.GetOrigin())  # # [z, y, x]
    numpySpacing = np.array((one_label.GetSpacing()))  # # [z, y, x]

    Z, Y, X = np.mgrid[:numpyImage.shape[0], :numpyImage.shape[1], :numpyImage.shape[2]]

    delta = delta

    gaussian_result = np.zeros(numpyImage.shape)
    for z in range(numpyImage.shape[0]):
        for y in range(numpyImage.shape[1]):
            for x in range(numpyImage.shape[2]):
                if numpyImage[z, y, x] != 0:
                    gau_func = np.exp(-1 * abs((Z - z) ** 2 + (Y - y) ** 2 + (X - x) ** 2) / (2 * (delta ** 2)))
                    gaussian_result += gau_func

    Min = np.min(gaussian_result)
    Max = np.max(gaussian_result)
    gaussian_result = (gaussian_result - Min) / (Max - Min)

    # 保存中间过程
    sitk_img = sitk.GetImageFromArray(gaussian_result, isVector=False)
    sitk_img.SetOrigin(numpyOrigin)
    sitk_img.SetSpacing(numpySpacing)
    if not os.path.exists(os.path.join(root_path, 'labelsTr')):
        os.mkdir(os.path.join(root_path, 'labelsTr'))
    sitk.WriteImage(sitk_img, os.path.join(root_path, 'labelsTr', save_name))


point_file_list = os.listdir(os.path.join(rootpath, 'point_label'))
# root_path = '/local/scratch/v_jiayin_sun/cvpr/data/lk/p_10'
for point_file in point_file_list:
    save_name = point_file
    print(point_file)
    point2GAU(point_file, rootpath, save_name, delta)

print(organ + "_delta=" + str(delta) + "_points=" + str(points))
