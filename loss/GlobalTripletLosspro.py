import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension."""
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """Calculate pairwise Euclidean distance."""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, num_pos):
    """Find the hardest positive and negative sample for each anchor."""
    assert len(dist_mat.size()) == 2
    N = dist_mat.size(0)
    assert dist_mat.size(1) == num_pos * 2

    dist_ap = dist_mat[:, :num_pos]
    dist_an = dist_mat[:, num_pos:]

    # Find the hardest positive and negative examples
    dist_ap, _ = torch.max(dist_ap, dim=1, keepdim=True)
    dist_an, _ = torch.min(dist_an, dim=1, keepdim=True)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


class TripletLosspro(nn.Module):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        super(TripletLosspro, self).__init__()
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, anchor_feat, positive_feat, negative_feat, labels, normalize_feature=False):
        if normalize_feature:
            anchor_feat = normalize(anchor_feat, axis=-1)
            positive_feat = normalize(positive_feat, axis=-1)
            negative_feat = normalize(negative_feat, axis=-1)

        # Compute pairwise distances between anchor and positive/negative features
        dist_ap = euclidean_dist(anchor_feat, positive_feat)
        dist_an = euclidean_dist(anchor_feat, negative_feat)

        # Concatenate distances to create a distance matrix of shape [64, 64]
        dist_mat = torch.cat((dist_ap, dist_an), dim=1)

        # Compute the hardest positive and negative examples
        dist_ap, dist_an = hard_example_mining(dist_mat, labels, num_pos=positive_feat.size(0))

        # Adjust distances with hard factor
        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        # Compute loss
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss, dist_ap, dist_an


# # Example usage
# if __name__ == "__main__":
#     # Assuming global_feat1, global_feat2, global_feat3 are the outputs of the three branches
#     global_feat1 = torch.randn(32, 768)
#     global_feat2 = torch.randn(32, 768)
#     global_feat3 = torch.randn(32, 768)
#     labels = torch.randint(0, 10, (32,))  # Example labels
#
#     triplet_loss = TripletLoss(margin=1.0, hard_factor=0.2)
#     loss, dist_ap, dist_an = triplet_loss(global_feat1, global_feat2, global_feat3, labels, normalize_feature=True)
#     print("Loss:", loss.item())
#     print("Distance AP:", dist_ap)
#     print("Distance AN:", dist_an)
