
import warnings
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from scipy.ndimage import distance_transform_edt

from monai.losses.focal_loss import FocalLoss
from monai.losses.spatial_mask import MaskedLoss
from monai.networks import one_hot
from monai.utils import LossReduction, Weight, look_up_option

from torch.nn import BCEWithLogitsLoss


class Weighted_Focal_DiceLoss(_Loss):
    def __init__(
        self,
        squared_pred: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        threshold: float = 0.5,
        gamm: float = 2,
        alpha: float = 1,
        wlambda: float = 1,
        w_nll: float = 0.003,
        reduction: Union[LossReduction, str] = LossReduction.MEAN
        )-> None:
       
        super().__init__(reduction=LossReduction(reduction).value)
        
        #self.sigmoid = sigmoid
        self.threshold = threshold
        self.gamma = gamm
        self.alpha = alpha
        self.wlambda = wlambda
        self.squared_pred = squared_pred
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.w_nll = w_nll

    def forward(self, input: torch.Tensor, target: torch.Tensor, nll_loss = 0) -> torch.Tensor:
        # input：[4, 1, 96, 96, 96]
        #if self.sigmoid:
        #    input = torch.sigmoid(input)
        #print(nll_loss[0].shape,len(nll_loss)) ##############################

        
        
    #---------------------------------------------wfocal----------------------------------------
        target_thres = target.detach().clone()
        target_thres[target_thres>self.threshold] = 1.0
        target_thres[target_thres<=self.threshold] = 0.0
        #target_thres = target_thres.to(torch.float32)
        bce = F.binary_cross_entropy_with_logits(input,target_thres,reduction = 'none')  
        #bce = F.binary_cross_entropy_with_logits(input,target,reduction = 'none')
        target_thres = target_thres.to(dtype=torch.int64)

        #pt = torch.cat((1*input**self.gamma,1*(1-input)**self.gamma),dim=1)  # 4,2,X,X,X
        w = torch.cat((1-target,target),dim=1)  #  4,2,X,X,X
    
        #pt = torch.gather(pt,1,target_thres) #这边的target也是卡过阈值后的0,1的     4,1,X,X,X
        w = torch.gather(w,1,target_thres) 

        target = target.to(torch.float32)

        #weighted_loss = bce * pt * w *self.alpha #4,1,X,X,X
        #weighted_loss = bce * pt * 1 *self.alpha #wfocal
        weighted_loss = bce * 1 * w *self.alpha #wce
        #weighted_loss = bce * 1 * 1 *self.alpha #ce
        
        #weighted_loss = bce
        loss_weighted_focal = torch.mean(weighted_loss, dim = [2,3,4])
        #print(loss_weighted_focal, loss_weighted_focal.shape)

    #---------------------------------------------dice----------------------------------------
        #eps = 1e-7
        #iflat = input[:, 0].view(-1)
        #tflat = target_thres[:, 0].view(-1)
        #intersection = (iflat * tflat).sum()  

        input = torch.sigmoid(input)

        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        intersection = torch.sum(target_thres * input, dim=reduce_axis)

        if self.squared_pred:
            target_thres = torch.pow(target_thres, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target_thres, dim=reduce_axis)  #[4]
        pred_o = torch.sum(input, dim=reduce_axis) #[4]
        denominator = ground_o + pred_o  #[4]

        #loss_dice =  1 - 2. * intersection / ((iflat).sum() + (tflat).sum() + eps)
        #loss_dict['loss'] = loss_dice + self.wlambda*loss_weighted_focal
        #return loss_dict

        dice: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr) #[4]
        f: torch.Tensor = dice + self.wlambda*loss_weighted_focal 
        #print(f.shape)
        for nll in nll_loss:
            f = f + self.w_nll*nll.unsqueeze(1)
        #print(f.shape)
        #print(dice,torch.mean(dice))

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        # Convert list of tensors to numpy array first to avoid warning
        nll_array = np.array([item.cpu().detach().numpy() for item in nll_loss])
        return f,torch.mean(dice), torch.mean(torch.tensor(nll_array).cuda())


class my_DiceLoss(_Loss):
    def __init__(
        self,
        squared_pred: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        threshold: float = 0.5,
        gamm: float = 2,
        alpha: float = 1,
        wlambda: float = 1,
        w_nll: float = 0.003,
        reduction: Union[LossReduction, str] = LossReduction.MEAN
        )-> None:
       
        super().__init__(reduction=LossReduction(reduction).value)
        
        #self.sigmoid = sigmoid
        self.threshold = threshold
        self.gamma = gamm
        self.alpha = alpha
        self.wlambda = wlambda
        self.squared_pred = squared_pred
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.w_nll = w_nll

    def forward(self, input: torch.Tensor, target: torch.Tensor, nll_loss = 0) -> torch.Tensor:
        # input：[4, 1, 96, 96, 96]
        #if self.sigmoid:
        #    input = torch.sigmoid(input)
        #print(nll_loss[0].shape,len(nll_loss)) ##############################

        
        
    #---------------------------------------------wfocal----------------------------------------
        target_thres = target.detach().clone()
        target_thres[target_thres>self.threshold] = 1.0
        target_thres[target_thres<=self.threshold] = 0.0
        #target_thres = target_thres.to(torch.float32)
        #bce = F.binary_cross_entropy_with_logits(input,target_thres,reduction = 'none')  
        #bce = F.binary_cross_entropy_with_logits(input,target,reduction = 'none')
        target_thres = target_thres.to(dtype=torch.int64)


        

    #---------------------------------------------dice----------------------------------------
        

        input = torch.sigmoid(input)

        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        intersection = torch.sum(target_thres * input, dim=reduce_axis)

        if self.squared_pred:
            target_thres = torch.pow(target_thres, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target_thres, dim=reduce_axis)  #[4]
        pred_o = torch.sum(input, dim=reduce_axis) #[4]
        denominator = ground_o + pred_o  #[4]

        

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr) #[4]
 
        #print(f.shape)
        for nll in nll_loss:
            #print(len(nll_loss), nll.shape)
            f = f + self.w_nll*nll.unsqueeze(1)
        #print(f.shape)
        #print(dice,torch.mean(dice))
        dice = f
        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        

        # Convert list of tensors to numpy array first to avoid warning
        nll_array = np.array([item.cpu().detach().numpy() for item in nll_loss])
        return f,torch.mean(dice), torch.mean(torch.tensor(nll_array).cuda())

class CE_DiceLoss(_Loss):
    def __init__(
        self,
        squared_pred: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        threshold: float = 0.5,
        gamm: float = 2,
        alpha: float = 1,
        wlambda: float = 1,
        w_nll: float = 0.003,
        reduction: Union[LossReduction, str] = LossReduction.MEAN
        )-> None:
       
        super().__init__(reduction=LossReduction(reduction).value)
        
        #self.sigmoid = sigmoid
        self.threshold = threshold
        self.gamma = gamm
        self.alpha = alpha
        self.wlambda = wlambda
        self.squared_pred = squared_pred
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.w_nll = w_nll

    def forward(self, input: torch.Tensor, target: torch.Tensor, nll_loss = 0) -> torch.Tensor:
        # input：[4, 1, 96, 96, 96]
        #if self.sigmoid:
        #    input = torch.sigmoid(input)
        #print(nll_loss[0].shape,len(nll_loss)) ##############################

        
        
    #---------------------------------------------wfocal----------------------------------------
        target_thres = target.detach().clone()
        target_thres[target_thres>self.threshold] = 1.0
        target_thres[target_thres<=self.threshold] = 0.0
        target_thres = target_thres.to(torch.float32)
        bce = F.binary_cross_entropy_with_logits(input,target,reduction = 'none')  
        #bce = F.binary_cross_entropy_with_logits(input,target,reduction = 'none')
        target_thres = target_thres.to(dtype=torch.int64)

        #pt = torch.cat((1*input**self.gamma,1*(1-input)**self.gamma),dim=1)  #  4,2,X,X,X
        #w = torch.cat((1-target,target),dim=1)  #  4,2,X,X,X
    
        #pt = torch.gather(pt,1,target_thres) #     4,1,X,X,X
        #w = torch.gather(w,1,target_thres) 

        target = target.to(torch.float32)

        #weighted_loss = bce * pt * w *self.alpha #4,1,X,X,X
        #weighted_loss = bce * pt * 1 *self.alpha #wfocal
        #weighted_loss = bce * 1 * w *self.alpha #wce
        weighted_loss = bce * 1 * 1 *self.alpha #ce
        
        #weighted_loss = bce
        loss_weighted_focal = torch.mean(weighted_loss, dim = [2,3,4])
        #print(loss_weighted_focal, loss_weighted_focal.shape)

    #---------------------------------------------dice----------------------------------------
        #eps = 1e-7
        #iflat = input[:, 0].view(-1)
        #tflat = target_thres[:, 0].view(-1)
        #intersection = (iflat * tflat).sum()  

        input = torch.sigmoid(input)

        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target_thres = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)  #[4]
        pred_o = torch.sum(input, dim=reduce_axis) #[4]
        denominator = ground_o + pred_o  #[4]

        #loss_dice =  1 - 2. * intersection / ((iflat).sum() + (tflat).sum() + eps)
        #loss_dict['loss'] = loss_dice + self.wlambda*loss_weighted_focal
        #return loss_dict

        dice: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr) #[4]
        f: torch.Tensor = dice + self.wlambda*loss_weighted_focal 
        #print(f.shape)
        for nll in nll_loss:
            f = f + self.w_nll*nll.unsqueeze(1)
        #print(f.shape)
        #print(dice,torch.mean(dice))

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        # Convert list of tensors to numpy array first to avoid warning
        nll_array = np.array([item.cpu().detach().numpy() for item in nll_loss])
        return f,torch.mean(dice), torch.mean(torch.tensor(nll_array).cuda())

class my_DiceLoss_prob(_Loss):
    def __init__(
        self,
        squared_pred: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        threshold: float = 0.5,
        gamm: float = 2,
        alpha: float = 1,
        wlambda: float = 1,
        w_nll: float = 0.003,
        reduction: Union[LossReduction, str] = LossReduction.MEAN
        )-> None:
       
        super().__init__(reduction=LossReduction(reduction).value)
        
        #self.sigmoid = sigmoid
        self.threshold = threshold
        self.gamma = gamm
        self.alpha = alpha
        self.wlambda = wlambda
        self.squared_pred = squared_pred
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.w_nll = w_nll

    def forward(self, input: torch.Tensor, target: torch.Tensor, nll_loss = 0) -> torch.Tensor:
        # input：[4, 1, 96, 96, 96]
        #if self.sigmoid:
        #    input = torch.sigmoid(input)
        #print(nll_loss[0].shape,len(nll_loss)) ##############################

        
        
    #---------------------------------------------wfocal----------------------------------------
        #target_thres = target
        #target_thres[target_thres>self.threshold] = 1.0
        #target_thres[target_thres<=self.threshold] = 0.0
        #target_thres = target_thres.to(torch.float32)
        #bce = F.binary_cross_entropy_with_logits(input,target_thres,reduction = 'none')  
        #bce = F.binary_cross_entropy_with_logits(input,target,reduction = 'none')
        #target_thres = target_thres.to(dtype=torch.int64)


        

    #---------------------------------------------dice----------------------------------------
        

        input = torch.sigmoid(input)

        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)  #[4]
        pred_o = torch.sum(input, dim=reduce_axis) #[4]
        denominator = ground_o + pred_o  #[4]

        

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr) #[4]
 
        #print(f.shape)
        for nll in nll_loss:
            #print(len(nll_loss), nll.shape)
            f = f + self.w_nll*nll.unsqueeze(1)
        #print(f.shape)
        #print(dice,torch.mean(dice))
        dice = f
        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        

        # Convert list of tensors to numpy array first to avoid warning
        nll_array = np.array([item.cpu().detach().numpy() for item in nll_loss])
        return f,torch.mean(dice), torch.mean(torch.tensor(nll_array).cuda())


class CE_Loss(_Loss):
    def __init__(
        self,
        squared_pred: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        threshold: float = 0.5,
        gamm: float = 2,
        alpha: float = 1,
        wlambda: float = 1,
        w_nll: float = 0.003,
        reduction: Union[LossReduction, str] = LossReduction.MEAN
        )-> None:
       
        super().__init__(reduction=LossReduction(reduction).value)
        
        #self.sigmoid = sigmoid
        self.threshold = threshold
        self.gamma = gamm
        self.alpha = alpha
        self.wlambda = wlambda
        self.squared_pred = squared_pred
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.w_nll = w_nll

    def forward(self, input: torch.Tensor, target: torch.Tensor, nll_loss = 0) -> torch.Tensor:
        # input：[4, 1, 96, 96, 96]
        #if self.sigmoid:
        #    input = torch.sigmoid(input)
        #print(nll_loss[0].shape,len(nll_loss)) ##############################

        
        
    #---------------------------------------------wfocal----------------------------------------
        target_thres = target.detach().clone()
        target_thres[target_thres>self.threshold] = 1.0
        target_thres[target_thres<=self.threshold] = 0.0
        target_thres = target_thres.to(torch.float32)
        bce = F.binary_cross_entropy_with_logits(input,target_thres,reduction = 'none')  
        #bce = F.binary_cross_entropy_with_logits(input,target,reduction = 'none')
        target_thres = target_thres.to(dtype=torch.int64)

        #pt = torch.cat((1*input**self.gamma,1*(1-input)**self.gamma),dim=1)  #   4,2,X,X,X
        #w = torch.cat((1-target,target),dim=1)  #  4,2,X,X,X
    
        #pt = torch.gather(pt,1,target_thres) #    4,1,X,X,X
        #w = torch.gather(w,1,target_thres) 

        target = target.to(torch.float32)

        #weighted_loss = bce * pt * w *self.alpha #4,1,X,X,X
        #weighted_loss = bce * pt * 1 *self.alpha #wfocal
        #weighted_loss = bce * 1 * w *self.alpha #wce
        weighted_loss = bce * 1 * 1 *self.alpha #ce
        
        #weighted_loss = bce
        loss_weighted_focal = torch.mean(weighted_loss, dim = [2,3,4])
        #print(loss_weighted_focal, loss_weighted_focal.shape)


        
        #---------------------------------------------dice----------------------------------------

        input = torch.sigmoid(input)

        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        intersection = torch.sum(target_thres * input, dim=reduce_axis)

        if self.squared_pred:
            target_thres = torch.pow(target_thres, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target_thres, dim=reduce_axis)  #[4]
        pred_o = torch.sum(input, dim=reduce_axis) #[4]
        denominator = ground_o + pred_o  #[4]

        #loss_dice =  1 - 2. * intersection / ((iflat).sum() + (tflat).sum() + eps)
        #loss_dict['loss'] = loss_dice + self.wlambda*loss_weighted_focal
        #return loss_dict

        dice: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr) #[4]
        f: torch.Tensor = loss_weighted_focal 
        #print(f.shape)
        for nll in nll_loss:
            f = f + self.w_nll*nll.unsqueeze(1)
        #print(f.shape)
        #print(dice,torch.mean(dice))

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        # Convert list of tensors to numpy array first to avoid warning
        nll_array = np.array([item.cpu().detach().numpy() for item in nll_loss])
        return f,torch.mean(dice), torch.mean(torch.tensor(nll_array).cuda())
    

class Focal_Loss(_Loss):
    def __init__(
        self,
        squared_pred: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        threshold: float = 0.5,
        gamm: float = 2,
        alpha: float = 0.25,
        wlambda: float = 1,
        w_nll: float = 0.003,
        reduction: Union[LossReduction, str] = LossReduction.MEAN
        )-> None:
       
        super().__init__(reduction=LossReduction(reduction).value)
        
        #self.sigmoid = sigmoid
        self.threshold = threshold
        self.gamma = gamm
        self.alpha = alpha
        self.wlambda = wlambda
        self.squared_pred = squared_pred
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.w_nll = w_nll

    def forward(self, input: torch.Tensor, target: torch.Tensor, nll_loss = 0) -> torch.Tensor:
        # input：[4, 1, 96, 96, 96]
        #if self.sigmoid:
        #    input = torch.sigmoid(input)
        #print(nll_loss[0].shape,len(nll_loss)) ##############################

        
        
    #---------------------------------------------wfocal----------------------------------------
        target_thres = target.detach().clone()
        target_thres[target_thres>self.threshold] = 1.0
        target_thres[target_thres<=self.threshold] = 0.0
        #target_thres = target_thres.to(torch.float32)
        #bce = F.binary_cross_entropy_with_logits(input,target_thres,reduction = 'none')  
        bce = F.binary_cross_entropy_with_logits(input,target_thres,reduction = 'none')
        target_thres = target_thres.to(dtype=torch.int64)
        input = torch.sigmoid(input)

        pt = torch.cat((1*input**self.gamma, 1*(1-input)**self.gamma),dim=1)  #  4,2,X,X,X
        w = torch.cat(((1-self.alpha)*torch.ones_like(input),self.alpha*torch.ones_like(input)),dim=1)  #  4,2,X,X,X
    
        pt = torch.gather(pt,1,target_thres) #  4,1,X,X,X
        w = torch.gather(w,1,target_thres) 

        target = target.to(torch.float32)

        #weighted_loss = bce * pt * w *self.alpha #4,1,X,X,X
        weighted_loss = bce * pt * w  #wfocal
        #weighted_loss = bce * 1 * w *self.alpha #wce
        #weighted_loss = bce * 1 * 1 *self.alpha #ce
        
        #weighted_loss = bce
        loss_weighted_focal = torch.mean(weighted_loss, dim = [2,3,4])
        #print(loss_weighted_focal, loss_weighted_focal.shape)


        
        #---------------------------------------------dice----------------------------------------

        

        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        intersection = torch.sum(target_thres * input, dim=reduce_axis)

        if self.squared_pred:
            target_thres = torch.pow(target_thres, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target_thres, dim=reduce_axis)  #[4]
        pred_o = torch.sum(input, dim=reduce_axis) #[4]
        denominator = ground_o + pred_o  #[4]

        #loss_dice =  1 - 2. * intersection / ((iflat).sum() + (tflat).sum() + eps)
        #loss_dict['loss'] = loss_dice + self.wlambda*loss_weighted_focal
        #return loss_dict

        dice: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr) #[4]
        f: torch.Tensor = loss_weighted_focal 
        #print(f.shape)
        for nll in nll_loss:
            f = f + self.w_nll*nll.unsqueeze(1)
        #print(f.shape)
        #print(dice,torch.mean(dice))

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        # Convert list of tensors to numpy array first to avoid warning
        nll_array = np.array([item.cpu().detach().numpy() for item in nll_loss])
        return f,torch.mean(dice), torch.mean(torch.tensor(nll_array).cuda())



# Helper functions for Boundary Loss
def softmax_helper(x):
    """Apply softmax with numerical stability"""
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def sum_tensor(inp, axes, keepdim=False):
    """Sum tensor along specified axes"""
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    Calculate true positives, false positives, false negatives
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


def compute_distance_map(segmentation):
    """
    Compute distance transform for boundary loss
    segmentation: (batch_size, x, y, z) binary mask
    returns: (batch_size, 2, x, y, z) distance maps for background and foreground
    """
    batch_size = segmentation.shape[0]
    # Create output with 2 channels (background, foreground)
    res = np.zeros((batch_size, 2, *segmentation.shape[1:]))
    
    for i in range(batch_size):
        posmask = segmentation[i].cpu().numpy() if torch.is_tensor(segmentation[i]) else segmentation[i]
        negmask = ~posmask
        
        # Distance transform for positive and negative regions
        pos_dist = distance_transform_edt(posmask)
        neg_dist = distance_transform_edt(negmask)
        
        # Normalize and invert (closer to boundary = higher weight)
        res[i, 0] = neg_dist  # Background channel
        res[i, 1] = pos_dist  # Foreground channel
        
    return res


class BoundaryLoss(nn.Module):
    """
    Boundary loss implementation
    Based on: https://github.com/LIVIAETS/surface-loss/
    """
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        
    def forward(self, net_output, distance_map):
        """
        net_output: (batch_size, 2, x, y, z) - network predictions (logits)
        distance_map: (batch_size, 2, x, y, z) - precomputed distance maps
        """
        # Apply softmax to get probabilities
        net_output = softmax_helper(net_output)
        
        # Element-wise multiplication of probabilities and distance maps
        # We use all channels (both background and foreground)
        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", net_output, distance_map)
        
        # Mean over all dimensions
        bd_loss = multipled.mean()
        
        return bd_loss


class CE_Dice_Boundary_Loss(_Loss):
    """
    Combined loss: Cross-Entropy + Dice + Boundary Loss
    LSeg = w_ce * CE + w_dice * Dice + w_boundary * Boundary
    """
    def __init__(
        self,
        squared_pred: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        threshold: float = 0.5,
        w_ce: float = 1.0,      # Weight for CE loss
        w_dice: float = 1.0,     # Weight for Dice loss  
        w_boundary: float = 0.01, # Weight for Boundary loss (usually smaller)
        reduction: Union[LossReduction, str] = LossReduction.MEAN
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        
        self.threshold = threshold
        self.squared_pred = squared_pred
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.w_boundary = w_boundary
        
        # Initialize boundary loss
        self.boundary_loss = BoundaryLoss()
        
    def forward(self, input: torch.Tensor, target: torch.Tensor, distance_map: torch.Tensor = None, nll_loss = 0) -> torch.Tensor:
        """
        input: (batch_size, 1, x, y, z) - network output (logits)
        target: (batch_size, 1, x, y, z) - ground truth labels
        distance_map: (batch_size, 2, x, y, z) - precomputed distance maps for boundary loss
        nll_loss: additional NLL loss if needed
        """
        
        # For binary segmentation, we need 2-channel output for boundary loss
        # Convert single channel to two channels (background, foreground)
        if input.shape[1] == 1:
            # Create 2-channel version for softmax operations
            input_2ch = torch.cat([-input, input], dim=1)  # (batch_size, 2, x, y, z)
        else:
            input_2ch = input
            
        # ---------- CE Loss ----------
        target_thres = target.detach().clone()
        target_thres[target_thres > self.threshold] = 1.0
        target_thres[target_thres <= self.threshold] = 0.0
        target_thres = target_thres.to(torch.float32)
        
        # Binary cross entropy with logits
        if input.shape[1] == 1:
            ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        else:
            ce_loss = F.binary_cross_entropy_with_logits(input[:, 1:2], target, reduction='none')
            
        ce_loss = torch.mean(ce_loss, dim=[2, 3, 4])  # Average over spatial dimensions
        
        # ---------- Dice Loss ----------
        if input.shape[1] == 1:
            input_sigmoid = torch.sigmoid(input)
        else:
            input_soft = softmax_helper(input_2ch)
            input_sigmoid = input_soft[:, 1:2]  # Take foreground channel
            
        if self.squared_pred:
            input_sigmoid = torch.pow(input_sigmoid, 2)
            
        # Threshold the target for Dice calculation
        target_dice = target_thres
        
        # Calculate Dice components
        reduce_axis = list(range(2, len(input_sigmoid.shape)))
        intersection = torch.sum(input_sigmoid * target_dice, dim=reduce_axis)
        
        ground_o = torch.sum(target_dice, dim=reduce_axis)
        pred_o = torch.sum(input_sigmoid, dim=reduce_axis)
        denominator = ground_o + pred_o
        
        dice_loss = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        
        # ---------- Boundary Loss ----------
        boundary_loss_value = torch.tensor(0.0).to(input.device)
        if distance_map is not None and self.w_boundary > 0:
            # Ensure distance_map is on the same device and dtype
            if distance_map.device != input.device:
                distance_map = distance_map.to(input.device)
            distance_map = distance_map.to(torch.float32)
            
            # Compute boundary loss using 2-channel format
            boundary_loss_value = self.boundary_loss(input_2ch, distance_map)
        
        # ---------- Combine Losses ----------
        # Add NLL loss if provided
        total_ce = ce_loss
        for nll in nll_loss:
            if nll is not None:
                total_ce = total_ce + 0.003 * nll.unsqueeze(1)  # Small weight for NLL
        
        # Apply reduction
        if self.reduction == LossReduction.MEAN.value:
            ce_final = torch.mean(total_ce)
            dice_final = torch.mean(dice_loss)
        elif self.reduction == LossReduction.SUM.value:
            ce_final = torch.sum(total_ce)
            dice_final = torch.sum(dice_loss)
        else:
            ce_final = total_ce
            dice_final = dice_loss
            
        # Weighted combination of three losses
        total_loss = (self.w_ce * ce_final + 
                     self.w_dice * dice_final + 
                     self.w_boundary * boundary_loss_value)
        
        # Return total loss and individual components for monitoring
        return total_loss, dice_final, boundary_loss_value
