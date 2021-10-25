from typing import List, Tuple

import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
    build_box_head,
    FastRCNNOutputLayers,
)
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Instances


@ROI_HEADS_REGISTRY.register()
class CalibrationROIHeads(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = CalibrationFastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }


class CalibrationFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(self, *args, frequencies, method, calibrate_on, gamma=0.0, renorm=True, **kwargs):
        self.calibrate_on = calibrate_on
        self.renorm = renorm

        if method == "gamma":
            self.gamma = 1 / frequencies ** gamma
        elif method == "cb":
            self.gamma = (1 - gamma) / (1 - gamma ** frequencies).reshape(1, -1)
        else:
            raise NotImplementedError(f"method either 'gamma' or 'cb', but got {method}")

        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape):
        args = super().from_config(cfg, input_shape)
        frequencies = _get_frequencies(cfg, by="image")
        frequencies = torch.as_tensor(frequencies, dtype=torch.float32).view(1, -1)
        args.update(
            {
                "frequencies": frequencies,
                "method": cfg.TEST.CALIBRATION.METHOD,
                "calibrate_on": cfg.TEST.CALIBRATION.WITH,
                "gamma": cfg.TEST.CALIBRATION.GAMMA,
                "renorm": cfg.TEST.CALIBRATION.RENORM,
            }
        )
        return args

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        self.gamma = self.gamma.to(scores.device)

        if self.calibrate_on == "logits":
            scores[:, :-1] *= self.gamma
            probs = F.softmax(scores, dim=-1)
        elif self.calibrate_on == "probs":
            probs = F.softmax(scores, dim=-1)
            probs[:, :-1] *= self.gamma
        elif self.calibrate_on == "exps":
            probs = torch.exp(scores)
            probs[:, :-1] *= self.gamma

        if self.renorm:
            probs /= probs.sum(dim=1, keepdim=True)

        return probs.split(num_inst_per_image, dim=0)


def _get_frequencies(cfg, by="image"):
    dataset = cfg.DATASETS.TEST[0]
    if "v0.5" in dataset:
        from .lvis_v0_5_categories import LVIS_CATEGORIES
    elif "v1" in dataset:
        from .lvis_v1_categories import LVIS_CATEGORIES
    else:
        raise NotImplementedError(f"unrecognized dataset {dataset} for getting class frequencies")

    frequencies = [c[f"{by}_count"] for c in LVIS_CATEGORIES]
    return frequencies
