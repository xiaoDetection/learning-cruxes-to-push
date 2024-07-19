# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import CascadeRCNN
from mmcv.runner import BaseModule, Sequential


class CruxLearnerBlock(BaseModule):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            expansion,
            init_cfg=None
    ):
        super(CruxLearnerBlock, self).__init__(init_cfg)
        mid_channels = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride, padding=(kernel_size // 2), groups=mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class CruxLearner(BaseModule):
    def __init__(self,
                 num_blocks=3,
                 mask_thr=0.5,
                 k = 10,
                 init_cfg=None):
        super(CruxLearner, self).__init__(init_cfg)
        assert num_blocks > 2, 'at least two blocks'
        self.mask_thr = mask_thr
        self.h_evaluation = Sequential()
        self.k = k

        # the fist block
        self.h_evaluation.append(CruxLearnerBlock(3, 16, 3, 1, 3))

        for _ in range(num_blocks - 2):
            self.h_evaluation.append(CruxLearnerBlock(16, 16, 3, 1, 3))
        
        # the last block
        self.h_evaluation.append(CruxLearnerBlock(16, 1, 5, 1, 3))
    
    # def mask_generation(self, heatmap):
    #     # mask = 1 if heatmap >= 0.5 else 0
    #     mask = -F.threshold(-heatmap, -self.mask_thr, -1)
    #     mask = F.threshold(mask, self.mask_thr, 0)
    #     return mask
    def mask_generation(self, heatmap):
        mask = 1 / (1 + torch.exp(-self.k * (heatmap - self.mask_thr)))
        return mask
    
    def forward(self, x):
        h = self.h_evaluation(x)
        return x * self.mask_generation(h)
    
@DETECTORS.register_module()
class LCPDetector(CascadeRCNN):
    """LCP detector based on CascadeRCNN"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 crux_learner_cfg=None,
                 kl_loss_weight=100
                ):
        super(LCPDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.crux_learner = CruxLearner(**crux_learner_cfg)
        self.train_mode = 'crux_learner'
        self.kl_loss_weight = kl_loss_weight

    def extract_feat(self, img):
        crux = self.crux_learner(img)
        x = self.backbone(img, crux)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      img_df=None, 
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img_df (Tenosr): detection favouring image
        """
        assert self.train_mode in ['catch_up_learner', 'crux_learner']
        if self.train_mode == 'catch_up_learner':
            return self.forward_train_catch_up_learner(
                img=img,
                img_df=img_df,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks=gt_masks,
                proposals=proposals,
                **kwargs
            )
        else:
            return super(LCPDetector, self).forward_train(
                img=img,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks=gt_masks,
                proposals=proposals,
                **kwargs
            )

    def forward_train_catch_up_learner(
        self,
        img,
        img_df,
        **kwargs
    ):
        crux = self.crux_learner(img)
        crux_df = self.crux_learner(img_df)
        _, catch_up_fea = self.backbone.forward_shallow_and_medium_stages(crux, True)
        _, df_fea = self.backbone.forward_shallow_and_medium_stages(crux_df, False)

        # kl loss
        pred = F.log_softmax(torch.flatten(catch_up_fea, 1), dim=-1)
        y = F.log_softmax(torch.flatten(df_fea, 1), dim=-1)
        kl_loss = F.kl_div(pred, y, log_target=True, reduction='batchmean') * self.kl_loss_weight
        return {'kl_loss':kl_loss}
    
    def switch_train_mode(self, train_mode):
        assert train_mode in ['catch_up_learner', 'crux_learner']
        if train_mode == 'catch_up_learner':
            for par in self.parameters():
                par.requires_grad = False
            for par in self.backbone.catch_up_learner.parameters():
                par.requires_grad = True
        elif train_mode == 'crux_learner':
            for par in self.parameters():
                par.requires_grad = True
            for par in self.backbone.parameters():
                par.requires_grad = False
            for par in self.backbone.catch_up_fusion.parameters():
                par.requires_grad = True
            for i in [2, 3]:
                res_layer = getattr(self.backbone, self.backbone.res_layers[i])
                for par in res_layer.parameters():
                    par.requires_grad = True
        self.train_mode = train_mode
        return

if __name__=='__main__':
    curx_learner = CruxLearner()
    input = torch.randn((1, 1, 5, 5))
    mask = curx_learner.mask_generation(input)
    print(input, mask)