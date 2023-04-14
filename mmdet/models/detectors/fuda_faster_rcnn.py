# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector, TwoStageDetectorWithDA
from mmcv.runner import  auto_fp16
import torch

@DETECTORS.register_module()
class FUDAFasterRCNN(TwoStageDetectorWithDA):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 da_image,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FUDAFasterRCNN, self).__init__(
            backbone=backbone,
            da_image=da_image,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """

        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        img_source = img[0]
        img_target = img[1]
        fea_s = self.extract_feat(img_source)
        fea_t = self.extract_feat(img_target)

        img_metas_s = img_metas[0]
        img_metas_t = img_metas[1]

        gt_bboxes = gt_bboxes[0]
        gt_labels = gt_labels[0]

        losses = dict()

        if self.da_image is not None:
            loss_img_s, uncertainty_map_s = self.da_image.forward(fea_s[4], torch.tensor(0))
            loss_img_t, uncertainty_map_t = self.da_image.forward(fea_t[4], torch.tensor(1))
            da_loss = dict(da_img_s_loss=torch.tensor(0) * loss_img_s,
                           da_img_t_loss=torch.tensor(0) * loss_img_t)
            # losses.update(da_loss)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                fea_s,
                img_metas_s,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses_s = self.roi_head.forward_train(fea_s, uncertainty_map_s, img_metas_s, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        roi_losses_t = self.roi_head.forward_train(fea_t, uncertainty_map_t, img_metas_t, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        loss_cls_IUA = torch.abs(roi_losses_s['loss_cls_IUA'] - roi_losses_t['loss_cls_IUA'])
        loss_bbox_IUA = torch.abs(roi_losses_s['loss_bbox_IUA'] - roi_losses_t['loss_bbox_IUA'])

        roi_losses = roi_losses_s
        roi_losses['loss_cls_IUA'] = loss_cls_IUA * torch.tensor(0)
        roi_losses['loss_bbox_IUA'] = loss_bbox_IUA * torch.tensor(0)
        losses.update(roi_losses_s)
        losses.pop('loss_cls_IUA')
        losses.pop('loss_bbox_IUA')


        return losses