import torch.nn.functional as F
import torch


def EPE(output, target):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h, w),
                                     mode='bicubic',
                                     align_corners=False)
    # Calculate the EPE along the second dimension
    EPE_map = torch.norm(target - upsampled_output, 2, 1)
    return EPE_map.mean()
