import torch
import torch.nn as nn
from torch.nn import functional as F
from models.Upsample import UpsampleDeterministicP3D, UpsampleDeterministicP3D_only_Z

affine_par = True
inplace = False


class Conv3d(nn.Conv3d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
            groups=1,
            bias=False,
    ):
        super(Conv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
            .mean(dim=4, keepdim=True)
        )
        weight = weight - weight_mean
        std = torch.sqrt(
            torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12
        ).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def conv3x3x3(
        in_planes,
        out_planes,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        dilation=(1, 1, 1),
        bias=False,
        weight_std=False,
):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
    else:
        return nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )


class NoBottleneck(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            downsample=None,
            fist_dilation=1,
            multi_grid=1,
            weight_std=False,
    ):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.relu = nn.ReLU(inplace=True)

        self.gn1 = nn.GroupNorm(8, inplanes)
        self.conv1 = conv3x3x3(
            inplanes,
            planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=dilation * multi_grid,
            dilation=dilation * multi_grid,
            bias=False,
            weight_std=self.weight_std,
        )

        self.gn2 = nn.GroupNorm(8, planes)
        self.conv2 = conv3x3x3(
            planes,
            planes,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=dilation * multi_grid,
            dilation=dilation * multi_grid,
            bias=False,
            weight_std=self.weight_std,
        )

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        skip = x

        seg = self.gn1(x)
        seg = self.relu(seg)
        seg = self.conv1(seg)

        seg = self.gn2(seg)
        seg = self.relu(seg)
        seg = self.conv2(seg)

        if self.downsample is not None:
            skip = self.downsample(x)

        seg = seg + skip
        return seg


class unet_for_contrastive_proj(nn.Module):
    def __init__(self, block, layers, in_channels, out_channels, weight_std=False):
        self.weight_std = weight_std
        super(unet_for_contrastive_proj, self).__init__()

        self.conv_4_32 = nn.Sequential(
            conv3x3x3(
                in_channels,
                32,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                weight_std=self.weight_std,
            )
        )

        self.conv_32_64 = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            conv3x3x3(
                32,
                64,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                weight_std=self.weight_std,
            ),
        )

        self.conv_64_128 = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            conv3x3x3(
                64,
                128,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                weight_std=self.weight_std,
            ),
        )

        self.conv_128_256 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            conv3x3x3(
                128,
                256,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                weight_std=self.weight_std,
            ),
        )

        self.layer0 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, 128, layers[2], stride=(1, 1, 1))
        self.layer3 = self._make_layer(block, 256, 256, layers[3], stride=(1, 1, 1))
        self.layer4 = self._make_layer(
            block, 256, 256, layers[4], stride=(1, 1, 1), dilation=(2, 2, 2)
        )

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            conv3x3x3(
                256,
                128,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dilation=(1, 1, 1),
                weight_std=self.weight_std,
            ),
        )

        self.seg_x4 = self._make_layer(block, 128, 64, 1, stride=(1, 1, 1))
        self.seg_x2 = self._make_layer(block, 64, 32, 1, stride=(1, 1, 1))
        self.seg_x1 = self._make_layer(block, 32, 32, 1, stride=(1, 1, 1))

        self.seg_cls = nn.Sequential(nn.Conv3d(32, out_channels, kernel_size=1))

        self.relu = nn.ReLU()
        self.proj = nn.Sequential(
            nn.Conv3d(256, 32, kernel_size=1), nn.ReLU(inplace=True)
        )

    def _make_layer(
            self,
            block,
            inplanes,
            outplanes,
            blocks,
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            multi_grid=1,
    ):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.GroupNorm(8, inplanes),
                nn.ReLU(inplace=True),
                conv3x3x3(
                    inplanes,
                    outplanes,
                    kernel_size=(1, 1, 1),
                    stride=stride,
                    padding=(0, 0, 0),
                    weight_std=self.weight_std,
                ),
            )

        layers = []
        generate_multi_grid = (
            lambda index, grids: grids[index % len(grids)]
            if isinstance(grids, tuple)
            else 1
        )
        layers.append(
            block(
                inplanes,
                outplanes,
                stride,
                dilation=dilation,
                downsample=downsample,
                multi_grid=generate_multi_grid(0, multi_grid),
                weight_std=self.weight_std,
            )
        )
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    outplanes,
                    dilation=dilation,
                    multi_grid=generate_multi_grid(i, multi_grid),
                    weight_std=self.weight_std,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):

        self.shape = [x.shape[-3], x.shape[-2], x.shape[-1]]
        ## seg-encoder
        x = self.conv_4_32(x)
        x = self.layer0(x)
        skip1 = x

        x = self.conv_32_64(x)
        x = self.layer1(x)
        skip2 = x

        x = self.conv_64_128(x)
        x = self.layer2(x)
        skip3 = x

        x = self.conv_128_256(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x_proj = self.proj(x)

        x = self.fusionConv(x)

        ## seg-decoder
        seg_x4 = UpsampleDeterministicP3D(2)(x)
        seg_x4 = seg_x4 + skip3
        seg_x4 = self.seg_x4(seg_x4)

        seg_x2 = UpsampleDeterministicP3D(2)(seg_x4)
        seg_x2 = seg_x2 + skip2
        seg_x2 = self.seg_x2(seg_x2)

        seg_x1 = UpsampleDeterministicP3D(2)(seg_x2)
        seg_x1 = seg_x1 + skip1
        seg_x1 = self.seg_x1(seg_x1)

        seg = self.seg_cls(seg_x1)

        return seg, x_proj


class unet_for_contrastive_proj_dt(nn.Module):
    def __init__(self, block, layers, in_channels, out_channels, weight_std=False):
        self.weight_std = weight_std
        super(unet_for_contrastive_proj_dt, self).__init__()

        self.conv_4_32 = nn.Sequential(
            conv3x3x3(
                in_channels,
                32,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                weight_std=self.weight_std,
            )
        )

        self.conv_32_64 = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            conv3x3x3(
                32,
                64,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                weight_std=self.weight_std,
            ),
        )

        self.conv_64_128 = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            conv3x3x3(
                64,
                128,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                weight_std=self.weight_std,
            ),
        )

        self.conv_128_256 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            conv3x3x3(
                128,
                256,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                weight_std=self.weight_std,
            ),
        )

        self.layer0 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, 128, layers[2], stride=(1, 1, 1))
        self.layer3 = self._make_layer(block, 256, 256, layers[3], stride=(1, 1, 1))
        self.layer4 = self._make_layer(
            block, 256, 256, layers[4], stride=(1, 1, 1), dilation=(2, 2, 2)
        )

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            conv3x3x3(
                256,
                128,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dilation=(1, 1, 1),
                weight_std=self.weight_std,
            ),
        )

        self.seg_x4 = self._make_layer(block, 128, 64, 1, stride=(1, 1, 1))
        self.seg_x2 = self._make_layer(block, 64, 32, 1, stride=(1, 1, 1))
        self.seg_x1 = self._make_layer(block, 32, 32, 1, stride=(1, 1, 1))

        self.seg_cls = nn.Sequential(nn.Conv3d(32, out_channels, kernel_size=1))
        self.dt_cls = nn.Sequential(nn.Conv3d(32, out_channels, kernel_size=1))

        self.relu = nn.ReLU()
        self.proj = nn.Sequential(
            nn.Conv3d(256, 32, kernel_size=1), nn.ReLU(inplace=True)
        )

    def _make_layer(
            self,
            block,
            inplanes,
            outplanes,
            blocks,
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            multi_grid=1,
    ):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.GroupNorm(8, inplanes),
                nn.ReLU(inplace=True),
                conv3x3x3(
                    inplanes,
                    outplanes,
                    kernel_size=(1, 1, 1),
                    stride=stride,
                    padding=(0, 0, 0),
                    weight_std=self.weight_std,
                ),
            )

        layers = []
        generate_multi_grid = (
            lambda index, grids: grids[index % len(grids)]
            if isinstance(grids, tuple)
            else 1
        )
        layers.append(
            block(
                inplanes,
                outplanes,
                stride,
                dilation=dilation,
                downsample=downsample,
                multi_grid=generate_multi_grid(0, multi_grid),
                weight_std=self.weight_std,
            )
        )
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    outplanes,
                    dilation=dilation,
                    multi_grid=generate_multi_grid(i, multi_grid),
                    weight_std=self.weight_std,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):

        self.shape = [x.shape[-3], x.shape[-2], x.shape[-1]]
        ## seg-encoder
        x = self.conv_4_32(x)
        x = self.layer0(x)
        skip1 = x

        x = self.conv_32_64(x)
        x = self.layer1(x)
        skip2 = x

        x = self.conv_64_128(x)
        x = self.layer2(x)
        skip3 = x

        x = self.conv_128_256(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x_proj = self.proj(x)

        x = self.fusionConv(x)

        ## seg-decoder
        seg_x4 = UpsampleDeterministicP3D(2)(x)
        seg_x4 = seg_x4 + skip3
        seg_x4 = self.seg_x4(seg_x4)

        seg_x2 = UpsampleDeterministicP3D(2)(seg_x4)
        seg_x2 = seg_x2 + skip2
        seg_x2 = self.seg_x2(seg_x2)

        seg_x1 = UpsampleDeterministicP3D(2)(seg_x2)
        seg_x1 = seg_x1 + skip1
        seg_x1 = self.seg_x1(seg_x1)

        seg = self.seg_cls(seg_x1)
        out_dis = self.dt_cls(seg_x1)

        return seg, x_proj, out_dis


class unet_for_contrastive_proj_dt_ccls_rc(nn.Module):
    def __init__(self, block, layers, in_channels, out_channels, weight_std=False):
        self.weight_std = weight_std
        super(unet_for_contrastive_proj_dt_ccls_rc, self).__init__()

        self.conv_4_32 = nn.Sequential(
            conv3x3x3(
                in_channels,
                32,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                weight_std=self.weight_std,
            )
        )

        self.conv_32_64 = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            conv3x3x3(
                32,
                64,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                weight_std=self.weight_std,
            ),
        )

        self.conv_64_128 = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            conv3x3x3(
                64,
                128,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                weight_std=self.weight_std,
            ),
        )

        self.conv_128_256 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            conv3x3x3(
                128,
                256,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                weight_std=self.weight_std,
            ),
        )

        self.layer0 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, 128, layers[2], stride=(1, 1, 1))
        self.layer3 = self._make_layer(block, 256, 256, layers[3], stride=(1, 1, 1))
        self.layer4 = self._make_layer(
            block, 256, 256, layers[4], stride=(1, 1, 1), dilation=(2, 2, 2)
        )

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            conv3x3x3(
                256,
                128,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dilation=(1, 1, 1),
                weight_std=self.weight_std,
            ),
        )

        self.seg_x4 = self._make_layer(block, 128, 64, 1, stride=(1, 1, 1))
        self.seg_x2 = self._make_layer(block, 64, 32, 1, stride=(1, 1, 1))
        self.seg_x1 = self._make_layer(block, 32, 32, 1, stride=(1, 1, 1))

        self.seg_cls = nn.Sequential(nn.Conv3d(32, out_channels, kernel_size=1))
        self.dt_cls = nn.Sequential(nn.Conv3d(32, out_channels, kernel_size=1))

        self.proj = nn.Sequential(
            nn.Conv3d(256, 32, kernel_size=1), nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gn = nn.GroupNorm(8, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(
            self,
            block,
            inplanes,
            outplanes,
            blocks,
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            multi_grid=1,
    ):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.GroupNorm(8, inplanes),
                nn.ReLU(inplace=True),
                conv3x3x3(
                    inplanes,
                    outplanes,
                    kernel_size=(1, 1, 1),
                    stride=stride,
                    padding=(0, 0, 0),
                    weight_std=self.weight_std,
                ),
            )

        layers = []
        generate_multi_grid = (
            lambda index, grids: grids[index % len(grids)]
            if isinstance(grids, tuple)
            else 1
        )
        layers.append(
            block(
                inplanes,
                outplanes,
                stride,
                dilation=dilation,
                downsample=downsample,
                multi_grid=generate_multi_grid(0, multi_grid),
                weight_std=self.weight_std,
            )
        )
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    outplanes,
                    dilation=dilation,
                    multi_grid=generate_multi_grid(i, multi_grid),
                    weight_std=self.weight_std,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):

        self.shape = [x.shape[-3], x.shape[-2], x.shape[-1]]
        ## seg-encoder
        x = self.conv_4_32(x)
        x = self.layer0(x)
        skip1 = x

        x = self.conv_32_64(x)
        x = self.layer1(x)
        skip2 = x

        x = self.conv_64_128(x)
        x = self.layer2(x)
        skip3 = x

        x = self.conv_128_256(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x_proj = self.proj(x)

        x_cls = (
            (UpsampleDeterministicP3D_only_Z(8)(torch.relu(self.gn(x))))
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )  # 1, 64，256，128，128
        x_cls = torch.squeeze(
            torch.squeeze(
                self.pool(
                    x_cls.view(
                        x_cls.size(0) * x_cls.size(1),
                        x_cls.size(2),
                        x_cls.size(3),
                        x_cls.size(4),
                    )
                ),
                dim=-1,
            ),
            dim=-1,
        )  # 64，256
        x_cls = self.fc2(self.relu(self.fc1(x_cls)))

        x = self.fusionConv(x)

        ## seg-decoder
        seg_x4 = UpsampleDeterministicP3D(2)(x)
        seg_x4 = seg_x4 + skip3
        seg_x4 = self.seg_x4(seg_x4)

        seg_x2 = UpsampleDeterministicP3D(2)(seg_x4)
        seg_x2 = seg_x2 + skip2
        seg_x2 = self.seg_x2(seg_x2)

        seg_x1 = UpsampleDeterministicP3D(2)(seg_x2)
        seg_x1 = seg_x1 + skip1
        seg_x1 = self.seg_x1(seg_x1)

        seg = self.seg_cls(seg_x1)
        out_dis = self.dt_cls(seg_x1)

        return seg, x_proj, out_dis, x_cls


def U_CorResNet_Fix_Contrastive_Proj(
        in_channels=1, out_channels=1, weight_std=True, layers=[1, 2, 2, 2, 2]
):
    model = unet_for_contrastive_proj(
        NoBottleneck, layers, in_channels, out_channels, weight_std
    )
    return model


def U_CorResNet_Fix_Contrastive_Proj_DT(
        in_channels=1, out_channels=1, weight_std=True, layers=[1, 2, 2, 2, 2]
):
    model = unet_for_contrastive_proj_dt(
        NoBottleneck, layers, in_channels, out_channels, weight_std
    )
    return model


def U_CorResNet_Fix_Contrastive_Proj_DT_CCLS_RC(
        in_channels=1, out_channels=1, weight_std=True, layers=[1, 2, 2, 2, 2]
):
    model = unet_for_contrastive_proj_dt_ccls_rc(
        NoBottleneck, layers, in_channels, out_channels, weight_std
    )
    return model
