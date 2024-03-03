
import warnings
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

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

        return f,torch.mean(dice), torch.mean(torch.tensor([item.cpu().detach().numpy() for item in nll_loss]).cuda())


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
        

        return f,torch.mean(dice), torch.mean(torch.tensor([item.cpu().detach().numpy() for item in nll_loss]).cuda())

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

        return f,torch.mean(dice), torch.mean(torch.tensor([item.cpu().detach().numpy() for item in nll_loss]).cuda())

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
        

        return f,torch.mean(dice), torch.mean(torch.tensor([item.cpu().detach().numpy() for item in nll_loss]).cuda())


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

        return f,torch.mean(dice), torch.mean(torch.tensor([item.cpu().detach().numpy() for item in nll_loss]).cuda())
    

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

        return f,torch.mean(dice), torch.mean(torch.tensor([item.cpu().detach().numpy() for item in nll_loss]).cuda())

