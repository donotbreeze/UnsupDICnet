import torch
from Models.correlation import Correlation
import math
import torch.nn as nn
from torchvision import ops
import torch.nn.functional as F
from torch.autograd import Variable


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, activation=True):
    if activation:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def predict_mask(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


def deformable_conv(in_planes, out_planes, kernel_size=3, strides=1, padding=1, use_bias=True):
    return ops.DeformConv2d(in_planes, out_planes, kernel_size, strides, padding, bias=use_bias)


def upsample_kernel2d(w, device):
    c = w // 2
    kernel = 1 - torch.abs(c - torch.arange(w, dtype=torch.float32, device=device)) / (c + 1)
    kernel = kernel.repeat(w).view(w, -1) * kernel.unsqueeze(1)
    return kernel.view(1, 1, w, w)


def Upsample(img, factor):
    if factor == 1:
        return img
    B, C, H, W = img.shape
    batch_img = img.view(B * C, 1, H, W)
    batch_img = F.pad(batch_img, [0, 1, 0, 1], mode='replicate')
    kernel = upsample_kernel2d(factor * 2 - 1, img.device)
    upsamp_img = F.conv_transpose2d(batch_img, kernel, stride=factor, padding=(factor - 1))
    upsamp_img = upsamp_img[:, :, : -1, :-1]
    _, _, H_up, W_up = upsamp_img.shape
    return upsamp_img.view(B, C, H_up, W_up)


# UnsupDICnet
class UnsupDICnet(nn.Module):
    def __init__(self):
        super(UnsupDICnet, self).__init__()

        self.deform_bias = True
        self.upfeat_ch = [16, 16, 16, 16]

        self.conv1a = conv(1, 16, kernel_size=3, stride=1)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv1c = conv(16, 16, kernel_size=3, stride=1)

        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        self.conv2c = conv(32, 32, kernel_size=3, stride=1)

        self.conv3a = conv(32, 64, kernel_size=3, stride=2)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        self.conv3c = conv(64, 64, kernel_size=3, stride=1)

        self.conv4a = conv(64, 128, kernel_size=3, stride=2)
        self.conv4b = conv(128, 128, kernel_size=3, stride=1)
        self.conv4c = conv(128, 128, kernel_size=3, stride=1)

        self.corr = Correlation(kernel_size=1, patch_size=9, stride=1, padding=0, dilation_patch=2)
        self.leakyRELU = nn.LeakyReLU(0.1)

        self.conv4_0 = conv(81, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(209, 64, kernel_size=3, stride=1)
        self.conv4_2 = conv(273, 32, kernel_size=3, stride=1)
        self.pred_flow4 = predict_flow(305)
        self.pred_mask4 = predict_mask(305)
        self.upfeat3 = deconv(305, self.upfeat_ch[0], kernel_size=4, stride=2, padding=1)

        self.conv3_0 = conv(163, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(291, 64, kernel_size=3, stride=1)
        self.conv3_2 = conv(355, 32, kernel_size=3, stride=1)
        self.pred_flow3 = predict_flow(387)
        self.pred_mask3 = predict_mask(387)
        self.upfeat2 = deconv(387, self.upfeat_ch[1], kernel_size=4, stride=2, padding=1)

        self.conv2_0 = conv(131, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(259, 64, kernel_size=3, stride=1)
        self.conv2_2 = conv(323, 32, kernel_size=3, stride=1)
        self.pred_flow2 = predict_flow(355)
        self.pred_mask2 = predict_mask(355)
        self.upfeat1 = deconv(355, self.upfeat_ch[2], kernel_size=4, stride=2, padding=1)

        self.conv1_0 = conv(115, 128, kernel_size=3, stride=1)
        self.conv1_1 = conv(243, 64, kernel_size=3, stride=1)
        self.conv1_2 = conv(307, 32, kernel_size=3, stride=1)
        self.pred_flow1 = predict_flow(339)

        self.deform3 = deformable_conv(64, 64)
        self.deform2 = deformable_conv(32, 32)
        self.deform1 = deformable_conv(16, 16)

        self.conv3f = conv(16, 64, kernel_size=3, stride=1, padding=1, activation=False)
        self.conv2f = conv(16, 32, kernel_size=3, stride=1, padding=1, activation=False)
        self.conv1f = conv(16, 16, kernel_size=3, stride=1, padding=1, activation=False)

        self.refine1 = nn.Sequential(
            conv(2, 32, kernel_size=3, stride=1, padding=1, activation=True),
            conv(32, 32, kernel_size=3, stride=1, padding=1, activation=True)
        )
        self.refine2 = nn.Sequential(
            conv(32, 32, kernel_size=3, stride=1, padding=1, activation=True),
            conv(32, 32, kernel_size=3, stride=1, padding=1, activation=True)
        )
        self.refine3 = nn.Sequential(
            conv(32, 32, kernel_size=3, stride=1, padding=1, activation=True),
            conv(32, 2, kernel_size=3, stride=1, padding=1, activation=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        device = x.device
        grid = grid.to(device)
        # vgrid = Variable(grid) + flo
        vgrid = Variable(grid) + torch.flip(flo, [1])

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def forward(self, im1, im2):
        # im1 = x[:,:3,:,:]
        # im2 = x[:,3:,:,:]

        c11 = self.conv1c(self.conv1b(self.conv1a(im1)))
        c21 = self.conv1c(self.conv1b(self.conv1a(im2)))
        c12 = self.conv2c(self.conv2b(self.conv2a(c11)))
        c22 = self.conv2c(self.conv2b(self.conv2a(c21)))
        c13 = self.conv3c(self.conv3b(self.conv3a(c12)))
        c23 = self.conv3c(self.conv3b(self.conv3a(c22)))
        c14 = self.conv4c(self.conv4b(self.conv4a(c13)))
        c24 = self.conv4c(self.conv4b(self.conv4a(c23)))

        corr4 = self.corr(c14, c24)  # 64 & 64 --> 81
        x = torch.cat((self.conv4_0(corr4), corr4), 1)  # 128 + 81 =  209
        x = torch.cat((self.conv4_1(x), x), 1)  # 64 + 209 = 273
        x = torch.cat((self.conv4_2(x), x), 1)  # 32 + 273 = 305
        flow4 = self.pred_flow4(x)
        mask4 = self.pred_mask4(x)

        feat3 = self.leakyRELU(self.upfeat3(x))
        flow3 = Upsample(flow4, 2)
        mask3 = Upsample(mask4, 2)
        warp3 = flow3.unsqueeze(1)
        warp3 = torch.repeat_interleave(warp3, 9, 1)
        S1, S2, S3, S4, S5 = warp3.shape
        warp3 = warp3.view(S1, S2 * S3, S4, S5)
        warp3 = self.deform3(c23, warp3)
        tradeoff3 = feat3
        warp3 = (warp3 * torch.sigmoid(mask3)) + self.conv3f(tradeoff3)
        warp3 = self.leakyRELU(warp3)
        corr3 = self.corr(c13, warp3)  # 64 & 64 --> 81
        x = torch.cat((corr3, c13, feat3, flow3), 1)  # 81 + 64 + 16 + 2 = 163
        x = torch.cat((self.conv3_0(x), x), 1)  # 128 + 163 = 291
        x = torch.cat((self.conv3_1(x), x), 1)  # 64 + 291 = 355
        x = torch.cat((self.conv3_2(x), x), 1)  # 32 + 355 = 387
        flow3 = flow3 + self.pred_flow3(x)
        mask3 = self.pred_mask3(x)

        feat2 = self.leakyRELU(self.upfeat2(x))
        flow2 = Upsample(flow3, 2)
        mask2 = Upsample(mask3, 2)
        warp2 = flow2.unsqueeze(1)
        warp2 = torch.repeat_interleave(warp2, 9, 1)
        S1, S2, S3, S4, S5 = warp2.shape
        warp2 = warp2.view(S1, S2 * S3, S4, S5)
        warp2 = self.deform2(c22, warp2)
        tradeoff2 = feat2
        warp2 = (warp2 * torch.sigmoid(mask2)) + self.conv2f(tradeoff2)
        warp2 = self.leakyRELU(warp2)
        corr2 = self.corr(c12, warp2)
        x = torch.cat((corr2, c12, feat2, flow2), 1)  # 81 + 32 + 16 + 2 = 131
        x = torch.cat((self.conv2_0(x), x), 1)  # 128 + 131 = 259
        x = torch.cat((self.conv2_1(x), x), 1)  # 64 + 259 = 323
        x = torch.cat((self.conv2_2(x), x), 1)  # 32 + 323 = 355
        flow2 = flow2 + self.pred_flow2(x)
        mask2 = self.pred_mask2(x)

        feat1 = self.leakyRELU(self.upfeat1(x))
        flow1 = Upsample(flow2, 2)
        mask1 = Upsample(mask2, 2)
        warp1 = flow1.unsqueeze(1)
        warp1 = torch.repeat_interleave(warp1, 9, 1)
        S1, S2, S3, S4, S5 = warp1.shape
        warp1 = warp1.view(S1, S2 * S3, S4, S5)
        warp1 = self.deform1(c21, warp1)
        tradeoff1 = feat1
        warp1 = (warp1 * torch.sigmoid(mask1)) + self.conv1f(tradeoff1)
        warp1 = self.leakyRELU(warp1)
        corr1 = self.corr(c11, warp1)
        x = torch.cat((corr1, c11, feat1, flow1), 1)  # 81 + 16 + 16 + 2 = 115
        x = torch.cat((self.conv1_0(x), x), 1)  # 128 + 115 = 243
        x = torch.cat((self.conv1_1(x), x), 1)  # 64 + 243 = 307
        x = torch.cat((self.conv1_2(x), x), 1)  # 32 + 307 = 339
        flow1 = flow1 + self.pred_flow1(x)

        out1 = self.refine1(flow1)
        out2 = self.refine2(out1) + out1
        flow0 = self.refine3(out2) + flow1

        return [flow4, flow3, flow2, flow0]


# end


# Estimate the flow
def estimate(tensorFirst, tensorSecond, model, train=False):
    assert (tensorFirst.size(2) == tensorSecond.size(2))
    assert (tensorFirst.size(3) == tensorSecond.size(3))

    intHeight = tensorFirst.size(2)
    intWidth = tensorFirst.size(3)

    tensorPreprocessedFirst = tensorFirst.view(-1, 1, intHeight, intWidth)
    tensorPreprocessedSecond = tensorSecond.view(-1, 1, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tensorPreprocessedFirst = torch.nn.functional.interpolate(
        input=tensorPreprocessedFirst,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode='bicubic',
        align_corners=False)
    tensorPreprocessedSecond = torch.nn.functional.interpolate(
        input=tensorPreprocessedSecond,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode='bicubic',
        align_corners=False)

    # Input images to the model
    raw_output = model(tensorPreprocessedFirst, tensorPreprocessedSecond)

    if train:
        return raw_output
    else:
        return raw_output[-1].cpu()

# end
