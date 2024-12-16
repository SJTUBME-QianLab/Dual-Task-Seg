
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn, Tensor

softmax_helper = lambda x: F.softmax(x, 1)


class dice_loss_multi_class_3D(nn.Module):
    def __init__(self, channel=3):
        super(dice_loss_multi_class_3D, self).__init__()
        self.loss_lambda = [1, 2, 3]
        self.channel = channel

    def forward(self, logits, gt):
        dice = 0
        eps = 1e-7

        assert (
            len(logits.shape) == 5
        ), "This loss is for 3D data (BCDHW), please check your output!"

        softmaxpred = logits

        for i in range(self.channel):
            inse = torch.sum(softmaxpred[:, i, :, :, :] * gt[:, i, :, :, :])
            l = torch.sum(softmaxpred[:, i, :, :, :])
            r = torch.sum(gt[:, i, :, :, :])
            dice += ((inse + eps) / (l + r + eps)) * self.loss_lambda[i] / 2.0

        return 1 - 2.0 * dice / self.channel


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
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
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(
            tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1
        )
        fp = torch.stack(
            tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1
        )
        fn = torch.stack(
            tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1
        )
        tn = torch.stack(
            tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1
        )

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=True, do_bg=False, smooth=1.0):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]

        dc = dc.mean()

        return 1 - dc


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class DC_and_CE_loss(nn.Module):
    def __init__(
        self, soft_dice_kwargs, ce_kwargs, aggregate="sum", weight_ce=1, weight_dice=1
    ):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son")  # reserved for other stuff (later)
        return result, ce_loss, dc_loss


class GANLoss(nn.Module):
    def __init__(
        self,
        use_lsgan=None,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
    ):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.Tensor = tensor
        if use_lsgan is not None:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):

        if target_is_real:
            real_tensor = self.Tensor(input.size()).fill_(self.real_label)
            real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = real_label_var

        else:
            fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
            fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = fake_label_var

        return target_tensor

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)

        return self.loss(input, target_tensor.cuda())


class DSC_loss(nn.Module):
    def __init__(self):
        super(DSC_loss, self).__init__()
        self.epsilon = 0.000001
        return

    def forward(self, pred, target):  # soft mode. per item.
        batch_num = pred.shape[0]
        pred = pred.contiguous().view(batch_num, -1)
        target = target.contiguous().view(batch_num, -1)
        DSC = (2 * (pred * target).sum(1) + self.epsilon) / (
            (pred + target).sum(1) + self.epsilon
        )
        return 1 - DSC.sum() / float(batch_num)


def l1_regularization_loss(y_true, y_pred, age_gap=0, age_range=60):

    epsilon = np.exp(-age_gap / age_range)
    # epsilon =1
    l1_loss = epsilon * torch.mean(torch.abs(y_pred - y_true), dim=(-1, -2, -3, -4))

    return torch.mean(l1_loss)


class BCELoss(nn.Module):
    def __init__(self, ignore_index=None, **kwargs):
        super(BCELoss, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights=None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
            bce = weights[1] * (target * torch.log(output)) + weights[0] * (
                (1 - target) * torch.log((1 - output))
            )
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1 - target) * torch.log((1 - output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):
        assert predict.shape == target.shape, "predict & target shape do not match"
        total_loss = 0
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                bce_loss = self.criterion(predict[:, i], target[:, i])
                total_loss += bce_loss

        return total_loss.mean()


class Box_Recall_loss(nn.Module):
    def __init__(self):
        super(Box_Recall_loss, self).__init__()
        self.epsilon = 0.000001
        self.margin = 20
        return

    def cal_loss(self, pred, target):

        binary_mask = pred >= 0.5
        batch_num, D, H, W = (
            pred.shape[0],
            pred.shape[-3],
            pred.shape[-2],
            pred.shape[-1],
        )

        if binary_mask.sum().item() == 0:
            binary_mask = torch.ones_like(pred).float().cuda()

        cropped_image = torch.zeros_like(pred).float().cuda()

        arr = torch.nonzero(binary_mask)
        minA = arr[:, -3].min().item()
        maxA = arr[:, -3].max().item()
        minB = arr[:, -2].min().item()
        maxB = arr[:, -2].max().item()
        minC = arr[:, -1].min().item()
        maxC = arr[:, -1].max().item()

        bbox = [
            int(max(minA, 0)),
            int(min(maxA, D)),
            int(max(minB - self.margin, 0)),
            int(min(maxB + self.margin + 1, H)),
            int(max(minC - self.margin, 0)),
            int(min(maxC + self.margin + 1, W)),
        ]

        cropped_image[
            :, :, bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]
        ] = pred[:, :, bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]]

        cropped_image = cropped_image.contiguous().view(batch_num, -1)
        target = target.contiguous().view(batch_num, -1)

        num = torch.sum(torch.mul(cropped_image, target), dim=1) + self.epsilon
        den = torch.sum(target, dim=1) + self.epsilon

        DSC = num / den
        return 1 - DSC.mean()

    def forward(self, pred, target):

        arr = torch.nonzero(target)
        minA = arr[:, -3].min().item()
        maxA = arr[:, -3].max().item()
        loss = 0
        for sli in range(minA - 2, maxA - 1):
            loss += self.cal_loss(
                # pred[:, :, sli : sli + 5], target[:, :, sli : sli + 5]
                pred[:, :, max(0, sli) : min(target.shape[-3], sli + 5)],
                target[:, :, max(0, sli) : min(target.shape[-3], sli + 5)],
            )

        return loss / ((maxA - 1) - (minA - 2))


class Recall_loss(nn.Module):
    def __init__(self):
        super(Recall_loss, self).__init__()
        self.epsilon = 0.000001
        return

    def forward(self, pred, target):
        batch_num = pred.shape[0]
        pred = pred.contiguous().view(batch_num, -1)
        target = target.contiguous().view(batch_num, -1)

        num = torch.sum(torch.mul(pred, target), dim=1) + self.epsilon
        den = torch.sum(target, dim=1) + self.epsilon

        DSC = num / den
        return 1 - DSC.mean()


class F_Score_loss_weight_beta(nn.Module):
    def __init__(self):
        super(F_Score_loss_weight_beta, self).__init__()
        self.epsilon = 0.000001
        return

    def forward(self, pred, target, beta):
        batch_num = pred.shape[0]
        target1 = target * (1 + beta * beta)
        target2 = target * (beta * beta)
        pred = pred.contiguous().view(batch_num, -1)
        target1 = target1.contiguous().view(batch_num, -1)
        target2 = target2.contiguous().view(batch_num, -1)

        num = torch.sum(torch.mul(pred, target1), dim=1) + self.epsilon
        den = torch.sum(target2, dim=1) + torch.sum(pred, dim=1) + self.epsilon

        DSC = num / den
        return 1 - DSC.mean()


def get_contrastive_loss_v2(emb_i, emb_j, cont_loss_func, k=2):
    average_emb_i = torch.sum(emb_i, (-1, -2)) / ((2 * k + 1) * (2 * k + 1))
    average_emb_j = torch.sum(emb_j, (-1, -2)) / ((2 * k + 1) * (2 * k + 1))
    emb = torch.cat((average_emb_i, average_emb_j), 0)
    label = np.concatenate(
        [np.ones(average_emb_i.shape[0]), np.zeros(average_emb_j.shape[0])]
    )
    label = torch.from_numpy(label)
    loss = cont_loss_func(emb, label)
    return loss
