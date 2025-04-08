import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    return F.normalize(x, p=2, dim=axis)


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    return dist.clamp(min=1e-12).sqrt()


def hard_example_mining(dist_mat, labels, return_inds=False):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class GlobalTripletLoss(object):
    """
    Global triplet loss using occluded features as anchors and global features as positive and negative samples.
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, occluded_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
            occluded_feat = normalize(occluded_feat, axis=-1)

        dist_mat = euclidean_dist(occluded_feat, global_feat)

        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss, dist_ap, dist_an


# # 示例用法
# global_feat = torch.randn(10, 2048, requires_grad=True)  # 全局特征
# occluded_feat = torch.randn(10, 2048, requires_grad=True)  # 遮挡特征
# labels = torch.randint(0, 2, (10,))  # 标签
#
# triplet_loss = GlobalTripletLoss(margin=1.0, hard_factor=0.1)
# loss, dist_ap, dist_an = triplet_loss(global_feat, occluded_feat, labels, normalize_feature=True)
#
# print('Loss:', loss.item())
# print('Positive distances:', dist_ap)
# print('Negative distances:', dist_an)
