import argparse
import torch
from torch.autograd import Variable
import SimpleITK as sitk
import numpy as np
import pydicom
import os
import random
import cv2
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import time

parser = argparse.ArgumentParser(description='Generate Gaussian pseudo labels from point annotations')
parser.add_argument('--dataset', type=str, required=True, choices=['btcv', 'chaos', 'mscmrseg'],
                    help='Dataset name (btcv, chaos, or mscmrseg)')
parser.add_argument('--organ', type=str, required=True,
                    help='Organ name (spleen, liver, right_kidney, left_kidney, lv_cavity, lv_myocardium, rv_cavity)')
parser.add_argument('--points', type=int, default=200,
                    help='Number of points to sample (default: 200)')
parser.add_argument('--delta', type=int, default=10,
                    help='Delta value for Gaussian distribution (default: 10)')
parser.add_argument('--workers', type=int, default=32,
                    help='Number of parallel workers (default: 32)')
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

points = args.points
delta = args.delta
dataset = args.dataset
organ = args.organ
num_workers = min(args.workers, cpu_count())

one_labelpath = f"./dataset/{dataset}/{organ}/labelsTr_gt"
rootpath = f'./dataset/{dataset}/{organ}/p_{points}_d_{delta}'
if not os.path.exists(rootpath):
    os.makedirs(rootpath, exist_ok=True)
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


def process_single_point_sampling(args_tuple):
    """Process a single label file for point sampling"""
    one_label, idx, total, one_labelpath, rootpath, point_num = args_tuple
    
    point_label_path = os.path.join(rootpath, 'point_label')
    point_output_file = os.path.join(point_label_path, one_label)
    
    # Check if point label already exists
    if os.path.exists(point_output_file):
        return f"  [{idx}/{total}] Skipping {one_label} (point label already exists)"
    
    try:
        image_path = os.path.join(one_labelpath, one_label)
        sitk_img = sitk.ReadImage(image_path)
        numpyImage = sitk.GetArrayFromImage(sitk_img)
        numpyOrigin = sitk_img.GetOrigin()
        numpySpacing = sitk_img.GetSpacing()

        points = np.where(numpyImage == 1)

        points_to_be_sampled = []
        for i in range(len(points[0])):
            z = points[0][i]
            y = points[1][i]
            x = points[2][i]
            points_to_be_sampled.append([x, y, z])
        
        points_to_be_sampled = random.sample(points_to_be_sampled, int(len(points_to_be_sampled) * 0.5))
        points_to_be_sampled = torch.tensor(points_to_be_sampled)

        points_to_be_sampled = points_to_be_sampled.unsqueeze(0)
        sim_data = Variable(points_to_be_sampled)

        centroids = farthest_point_sample(sim_data, point_num)
        centroid = sim_data[0, centroids, :][0]

        point_labeled_image = np.zeros_like(numpyImage)
        for cord in centroid:
            x, y, z = cord
            point_labeled_image[z, y, x] = 1

        sitk_img = sitk.GetImageFromArray(point_labeled_image, isVector=False)
        sitk_img.SetOrigin(numpyOrigin)
        sitk_img.SetSpacing(numpySpacing)
        
        if not os.path.exists(point_label_path):
            os.makedirs(point_label_path, exist_ok=True)
        
        sitk.WriteImage(sitk_img, point_output_file)
        return f"  [{idx}/{total}] Processed {one_label}"
    except Exception as e:
        return f"  [{idx}/{total}] Error processing {one_label}: {str(e)}"


# Step 1: Point Sampling (Parallel)
one_labels = os.listdir(one_labelpath)
print(f"Using {num_workers} parallel workers (max available: {cpu_count()})")
print(f"\nStep 1: Sampling {points} points from {len(one_labels)} label files (parallel)...", flush=True)

# Prepare arguments for parallel processing
step1_args = [(one_label, idx, len(one_labels), one_labelpath, rootpath, point_num) 
              for idx, one_label in enumerate(one_labels, 1)]

start_time = time.time()
with Pool(processes=num_workers) as pool:
    results = pool.map(process_single_point_sampling, step1_args)

# Print results
for result in results:
    if result:  # Only print non-empty results
        print(result, flush=True)

step1_time = time.time() - start_time
print(f"Step 1 completed in {step1_time:.2f} seconds")


def process_single_gaussian(args_tuple):
    """Process a single point file to generate Gaussian pseudo-label"""
    point_file, idx, total, rootpath, delta = args_tuple
    
    save_name = point_file
    output_path = os.path.join(rootpath, 'labelsTr', save_name)
    
    # Check if output file already exists
    if os.path.exists(output_path):
        return f"  [{idx}/{total}] Skipping {point_file} (Gaussian label already exists)"
    
    try:
        one_label_file = os.path.join(rootpath, 'point_label', point_file)
        one_label = sitk.ReadImage(one_label_file)
        numpyImage = sitk.GetArrayFromImage(one_label)
        numpyOrigin = np.array(one_label.GetOrigin())
        numpySpacing = np.array((one_label.GetSpacing()))

        Z, Y, X = np.mgrid[:numpyImage.shape[0], :numpyImage.shape[1], :numpyImage.shape[2]]

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

        # Save result
        sitk_img = sitk.GetImageFromArray(gaussian_result, isVector=False)
        sitk_img.SetOrigin(numpyOrigin)
        sitk_img.SetSpacing(numpySpacing)
        
        labelsTr_path = os.path.join(rootpath, 'labelsTr')
        if not os.path.exists(labelsTr_path):
            os.makedirs(labelsTr_path, exist_ok=True)
        
        sitk.WriteImage(sitk_img, output_path)
        return f"  [{idx}/{total}] Generated Gaussian for {point_file}"
    except Exception as e:
        return f"  [{idx}/{total}] Error processing {point_file}: {str(e)}"


# Step 2: Gaussian Generation (Parallel)
point_file_list = os.listdir(os.path.join(rootpath, 'point_label'))
print(f"\nStep 2: Generating Gaussian pseudo-labels with delta={delta} (parallel)...", flush=True)

# Prepare arguments for parallel processing
step2_args = [(point_file, idx, len(point_file_list), rootpath, delta)
              for idx, point_file in enumerate(point_file_list, 1)]

start_time = time.time()
with Pool(processes=num_workers) as pool:
    results = pool.map(process_single_gaussian, step2_args)

# Print results
for result in results:
    if result:  # Only print non-empty results
        print(result, flush=True)

step2_time = time.time() - start_time
print(f"Step 2 completed in {step2_time:.2f} seconds")

print(f"\nCompleted: {organ} with {points} points and delta={delta}", flush=True)
print(f"Total processing time: {step1_time + step2_time:.2f} seconds")
