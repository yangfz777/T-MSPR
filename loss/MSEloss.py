import torch
import torch.nn as nn
import torch.nn.functional as F
class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss


def normalize(tensor, method="minmax", epsilon=1e-6):
    """
    对张量进行规范化处理

    :param tensor: 输入张量
    :param method: 规范化方法，支持 "minmax" 和 "zscore"
    :param epsilon: 避免除零的小常数
    :return: 规范化后的张量
    """
    if method == "minmax":
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        return (tensor - min_val) / (max_val - min_val + epsilon)
    elif method == "zscore":
        mean = torch.mean(tensor)
        std = torch.std(tensor) + epsilon
        return (tensor - mean) / std
    else:
        raise ValueError("Unsupported normalization method")

def compute_multichannel_loss(predictions, targets, loss_type="mse", normalize_method=None):
    total_loss = 0
    # print("output_features min/max:", predictions.min().item(), predictions.max().item())
    # print("target min/max:", targets.min().item(), targets.max().item())
    targets_min = targets.min(dim=0, keepdim=True).values
    targets_min = targets_min.min(dim=1, keepdim=True).values
    targets_min = targets_min.min(dim=2, keepdim=True).values

    targets_max = targets.max(dim=0, keepdim=True).values
    targets_max = targets_max.max(dim=1, keepdim=True).values
    targets_max = targets_max.max(dim=2, keepdim=True).values
    # 归一化
    targets_normalized = (targets - targets_min) / (targets_max - targets_min + 1e-8)

    predictions_min = predictions.min(dim=0, keepdim=True).values
    predictions_min = predictions_min.min(dim=1, keepdim=True).values
    predictions_min = predictions_min.min(dim=2, keepdim=True).values

    predictions_max = predictions.max(dim=0, keepdim=True).values
    predictions_max = predictions_max.max(dim=1, keepdim=True).values
    predictions_max = predictions_max.max(dim=2, keepdim=True).values
    # 归一化
    predictions_normalized = (predictions - predictions_min) / (predictions_max - predictions_min + 1e-8)
    # print("epoch_features_normalized min/max:", epoch_features_normalized.min().item(), epoch_features_normalized.max().item())
    # 如果需要规范化，先对 predictions 和 targets 进行规范化
    if normalize_method is not None:
        predictions = normalize(predictions, method=normalize_method)
        targets = normalize(targets, method=normalize_method)

    for channel in range(predictions_normalized.shape[-1]):
        pred_channel = predictions_normalized[..., channel]
        target_channel = targets_normalized[..., channel]

        if loss_type == "mse":
            loss = F.mse_loss(pred_channel, target_channel)
        elif loss_type == "cross_entropy":
            # 检查目标类型
            if target_channel.dtype == torch.float32:
                # 如果是概率分布，直接使用浮点型
                loss = F.cross_entropy(pred_channel, target_channel, label_smoothing=0.1)
            else:
                # 如果是类别索引
                target_channel = torch.argmax(target_channel, dim=-1)  # 转为类别索引
                loss = F.cross_entropy(pred_channel, target_channel.long())
        elif loss_type == "l1":
            smooth_l1_loss_fn = nn.SmoothL1Loss()
            loss = smooth_l1_loss_fn(pred_channel, target_channel)
        else:
            raise ValueError("Unsupported loss type")

        total_loss += loss

    total_loss = total_loss / predictions.shape[-1]
    return total_loss