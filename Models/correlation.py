import torch.nn.functional as F
from spatial_correlation_sampler import spatial_correlation_sample
from torch.nn.modules.module import Module


class Correlation(Module):
    def __init__(self, kernel_size=1, patch_size=21, stride=1, padding=0, dilation_patch=2):
        super(Correlation, self).__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation_patch = dilation_patch

    # @staticmethod
    def forward(self, input1, input2):
        # input1 = input1.contiguous()
        # input2 = input2.contiguous()
        out_corr = spatial_correlation_sample(input1,
                                              input2,
                                              kernel_size=self.kernel_size,
                                              patch_size=self.patch_size,
                                              stride=self.stride,
                                              padding=self.padding,
                                              dilation_patch=self.dilation_patch)
        b, ph, pw, h, w = out_corr.size()
        out_corr = out_corr.view(b, ph * pw, h, w) / input1.size(1)
        return F.leaky_relu_(out_corr, 0.1)
