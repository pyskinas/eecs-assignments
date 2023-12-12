"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction

# Mine, for checking how much faster vectorised code runs 
import time
from decimal import *


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    """
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(weights=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            }
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        # Get in and out channels for code readability
        p3_in_c = dummy_out_shapes[0][1][1]
        p4_in_c = dummy_out_shapes[1][1][1]
        p5_in_c = dummy_out_shapes[2][1][1]
        # See channels:
        # print(p3_in_c)
        # print(p4_in_c)
        # print(p5_in_c)

        # Get device and dtype ? Probably not since the only input we 
        # have in out_channels, which is an int.
        
        # Get channel transfromation params.
        self.fpn_params["c5"] = nn.Conv2d(in_channels=p5_in_c, out_channels=out_channels, 
          kernel_size=1, stride=1, padding=0)
        self.fpn_params["c4"] = nn.Conv2d(in_channels=p4_in_c, out_channels=out_channels, 
          kernel_size=1, stride=1, padding=0)
        self.fpn_params["c3"] = nn.Conv2d(in_channels=p3_in_c, out_channels=out_channels, 
          kernel_size=1, stride=1, padding=0)

        # Get FPN outputs.
        self.fpn_params["p5"] = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
          kernel_size=3, stride=1, padding=1)
        self.fpn_params["p4"] = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
          kernel_size=3, stride=1, padding=1)
        self.fpn_params["p3"] = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
          kernel_size=3, stride=1, padding=1)
        
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################
        # Get channel transformations from lateral connections
        c_t = {}
        for key, value in backbone_feats.items():
            c_t[key] = self.fpn_params[key](value)
            
        
        # Feed top layer (c5) through 3x3 conv; bc it has no connection to other pyramid 
        # layers.
        fpn_feats["p5"] = self.fpn_params["p5"](c_t["c5"])

        # Get remaining pyramid outputs with upsampling and 3x3 conv. k is in {4,3}
        for k in range(4, 2, -1):
            # Get keys. Since we use "p5", "p4", "c4" in the same line, I use numbers.
            K, Kp1 = str(k), str(k+1)

            # Sum 1x1 conv output with upscaled above layer in the pyramid
            fpn_feats["p"+K] = F.interpolate(fpn_feats["p"+Kp1], scale_factor=(2,2)) + c_t["c"+K]
           
            # Feed sum through 3x3 cov
            fpn_feats["p"+K] = self.fpn_params["p"+K](fpn_feats["p"+K])

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Examples suggest you go over the width first, so left to right, 
        # top to bottom (in that order).
        _, _, H, W = feat_shape
        location_coords[level_name] = torch.zeros((H,W,2), device=device, dtype=dtype)

        # Initial Brute force approach:
        # start_time = Decimal(time.time())
        # for i in range(H):
        #     for j in range(W):
        #         location_coords[level_name][i,j,0] = level_stride*(i+0.5)
        #         location_coords[level_name][i,j,1] = level_stride*(j+0.5)
        # end_time = Decimal(time.time())
        # brute_time = end_time - start_time

        # # Reinitialise to do vectorised version; and check time taken.
        # location_coords[level_name] = torch.zeros((H,W,2), device=device, dtype=dtype)

        # Vectorised approach; Roughly 16x Faster, but vec_time is sometimes 0,
        # which gives errors so it doesn't always work. Uncomment to see.
        # start_time = Decimal(time.time())
        location_coords[level_name][:, :, 0] = (location_coords[level_name][:, :, 0].t() + torch.arange(H, device=device, dtype=dtype)).t() + 0.5
        location_coords[level_name][:, :, 1] += torch.arange(W, device=device, dtype=dtype) + 0.5
        location_coords[level_name] *= level_stride
        # end_time = Decimal(time.time())
        # vec_time = end_time-start_time

        # print("Brute force time: " + brute_time)
        # print("Vectorised time: " + vec_time)
        # diff = brute_time/vec_time

        # print(f"Vectorised implementation was {diff:.2f}x faster!")

        # Flatten H,W dims
        location_coords[level_name] = torch.flatten(
            location_coords[level_name], start_dim=0, end_dim=1)

        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
     # First, sort the scores and their associated boxes
    a = torch.argsort(scores).flip([0])
    scores = scores[a]
    boxes = boxes[a]

    # I made my program for (x1,y1) as top-left, and (x2,y2) as bottom right
    # coordinates. but having at the test inputs, it's bottom-left, top-right ???
    # So here i'll permute the boxes' columns to match what torchvision may be doing
    DEVICE = boxes.device
    boxes = torch.cat((
        torch.index_select(boxes,1,torch.LongTensor([0,3]).to(DEVICE)),
        torch.index_select(boxes, 1, torch.LongTensor([2,1]).to(DEVICE))), dim=1)
  
    # Make keep an array so we can append to it
    keep = []

    # Second, while scores is not empty, we keep going
    # To avoid a potentially infinite loop, we'll stop when count gets to N+1
    lim = (scores.shape[0] + 1) *10
    count = 0
    while (scores.numel()):

        # Go down the list for each top element (xt1 = x1 of top_box )
        xt1, yt1, xt2, yt2 = boxes[0]

        # Get remaining elements from boxes
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        
        # Get IoUs between the top box and all other boxes
        # Get 'union' (not fully tested, but should work) has 2 intersections
        union = get_union(x1, x2, xt1, xt2, y1, y2, yt1, yt2)

        # Get intersection
        x_in = get_intersection(x1, x2, xt1, xt2)
        y_in = get_intersection(y2, y1, yt2, yt1) # NOTE: order is swapped because x1 < x2, but y1 > y2.
        intersection = x_in*y_in

        # union has overlapping intersection, so we need to subtract it
        union = union - intersection

        # Beautiful intersection over union (IoU)
        iou = intersection/union 

        # Save index of top_box
        keep.append(a[0].item())

        # Get indices for elimination
        elim_ind = iou>iou_threshold

        # Increment count, preliminary debugging
        # count += 1
        # if not count%5:
        #     print(boxes[:5])
        #     print(iou[:5])

        # Eliminate items at `elim_ind` indices for `a`, `scores` and `boxes`
        a = eliminate(a, elim_ind)
        scores = eliminate(scores, elim_ind)
        boxes = eliminate(boxes, elim_ind)
           
    
    keep = torch.asarray(keep).long()
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep

def get_union(
  x1 : torch.Tensor, x2 : torch.Tensor, xt1 : torch.Tensor, xt2 : torch.Tensor,
  y1 : torch.Tensor, y2 : torch.Tensor, yt1 : torch.Tensor, yt2 : torch.Tensor
  ):
    # Get unions (4 cases: boxes are to the left, right, over or within the top box)
    # Assuming a union is the square that contains both boxes
    # return torch.max(
    #     torch.max(torch.abs(x2 - xt1), torch.abs(x1 - xt2)),
    #     torch.max(torch.abs(x2-x1), torch.abs(xt1-xt2))
    # )

    # Assuming a union is the sum of areas of both square (makes more sense to me)
    top_square = (xt2 - xt1)*(yt1 - yt2)
    squares = (x2 - x1)*(y1-y2)
    return squares + top_square

def get_intersection(x1 : torch.Tensor, x2 : torch.Tensor, xt1 : torch.Tensor, xt2 : torch.Tensor):
    # Get intersections (4 cases: boxes are to the left, right, over or within the top box)
    a = torch.clamp(torch.min(x2 - xt1, xt2 - x1) , 0)
    b = torch.min(x2-x1, xt2 - xt1)
    c = torch.min(a,b)
    return c

def eliminate(x : torch.Tensor, ind: torch.Tensor):
    shape = x.shape
    x[ind] = -3
    if len(shape) == 1:
        return x[x != -3]
    else:
        return x[x != -3].reshape(-1, shape[1])


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
