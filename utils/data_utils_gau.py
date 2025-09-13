

import math
import os

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from monai import data, transforms
from monai.data import load_decathlon_datalist


class ComputeDistanceMapd(transforms.MapTransform):
    """
    Compute distance map for boundary loss
    """
    def __init__(self, keys, source_key="label_thres", threshold=0.5):
        super().__init__(keys)
        self.source_key = source_key
        self.threshold = threshold
        
    def __call__(self, data):
        d = dict(data)
        
        # Get the source label
        source = d[self.source_key]
        
        # Convert to binary mask
        if torch.is_tensor(source):
            binary_mask = (source > self.threshold).float()
            binary_mask_np = binary_mask.squeeze(0).cpu().numpy()  # Remove channel dimension
        else:
            binary_mask = (source > self.threshold).astype(np.float32)
            binary_mask_np = np.squeeze(binary_mask, axis=0) if binary_mask.ndim == 4 else binary_mask
            
        # Compute distance maps for boundary loss
        # Create 2-channel distance map (background, foreground)
        dist_map = np.zeros((2,) + binary_mask_np.shape, dtype=np.float32)
        
        # Background distance
        neg_mask = ~binary_mask_np.astype(bool)
        if neg_mask.any():
            dist_map[0] = distance_transform_edt(neg_mask)
        
        # Foreground distance  
        pos_mask = binary_mask_np.astype(bool)
        if pos_mask.any():
            dist_map[1] = distance_transform_edt(pos_mask)
            
        # Store distance map
        for key in self.keys:
            d[key] = torch.from_numpy(dist_map).float()
            
        return d


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)

    # Check if using boundary loss
    use_boundary_loss = hasattr(args, 'loss_type') and args.loss_type == 'ce_dice_boundary'
    
    # Check if using MSCMRSEG dataset (which has small Z dimension ~15-16 slices)
    is_mscmrseg = "mscmrseg" in args.json_list.lower()

    # Build train transform list
    train_transform_list = [
        transforms.LoadImaged(keys=["image", "label", "label_thres"]),
        transforms.AddChanneld(keys=["image", "label", "label_thres"]),
        transforms.Orientationd(keys=["image", "label", "label_thres"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["image", "label", "label_thres"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest", "nearest")
        ),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
        ),
        #transforms.CropForegroundd(keys=["image", "label","label_thres"], source_key="image"),
    ]
    
    # Add padding for MSCMRSEG dataset to ensure images are large enough for ROI
    if is_mscmrseg:
        train_transform_list.append(
            transforms.SpatialPadd(
                keys=["image", "label", "label_thres"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="constant"
            )
        )
    
    train_transform_list.append(
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label","label_thres"],
            label_key="label_thres",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        )
    )
    train_transform_list.extend([
        transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
    ])

    # Only add distance map computation if using boundary loss
    if use_boundary_loss:
        train_transform_list.append(ComputeDistanceMapd(keys=["distance_map"], source_key="label_thres", threshold=0.5))
        train_transform_list.append(transforms.ToTensord(keys=["image", "label", "distance_map"]))
    else:
        train_transform_list.append(transforms.ToTensord(keys=["image", "label"]))

    train_transform = transforms.Compose(train_transform_list)

    # Build val transform list based on whether boundary loss is used
    if use_boundary_loss:
        val_transform_list = [
            transforms.LoadImaged(keys=["image", "label", "label_thres"]),
            transforms.AddChanneld(keys=["image", "label", "label_thres"]),
            transforms.Orientationd(keys=["image", "label", "label_thres"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label", "label_thres"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            #transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
        if is_mscmrseg:
            val_transform_list.append(
                transforms.SpatialPadd(
                    keys=["image", "label", "label_thres"],
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode="constant"
                )
            )
        val_transform_list.extend([
            ComputeDistanceMapd(keys=["distance_map"], source_key="label_thres", threshold=0.5),
            transforms.ToTensord(keys=["image", "label", "distance_map"]),
        ])
        val_transform = transforms.Compose(val_transform_list)
    else:
        # Original validation transform without label_thres and distance_map
        val_transform_list = [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            #transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
        if is_mscmrseg:
            val_transform_list.append(
                transforms.SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode="constant"
                )
            )
        val_transform_list.append(transforms.ToTensord(keys=["image", "label"]))
        val_transform = transforms.Compose(val_transform_list)
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            #transforms.CropForegroundd(keys=["image", "label"], source_key="image"),  
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        print('a')
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir) 
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=20, cache_rate=1.0, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = [train_loader, val_loader]

    return loader
