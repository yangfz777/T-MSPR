import torch
import torch.nn.functional as F


def contrastive_loss(f_local_non_occluded, f_global, f_local_occluded, tau=0.07):
    """
    计算对比学习损失，最大化未遮挡特征与全局特征的相似性，
    最小化遮挡特征与全局特征的相似性。

    参数：
    - f_local_non_occluded: 未遮挡区域的局部特征，形状为 (batch_size, feature_dim)
    - f_global: 全局特征，形状为 (batch_size, feature_dim)
    - f_local_occluded: 遮挡区域的局部特征，形状为 (batch_size, feature_dim)
    - tau: 温度参数 (default=0.07)

    返回：
    - 对比学习损失 (标量)
    """

    # 计算未遮挡局部特征与全局特征之间的相似性 (正样本对)
    sim_pos = F.cosine_similarity(f_local_non_occluded, f_global, dim=-1) / tau

    # 计算遮挡局部特征与全局特征之间的相似性 (负样本对)
    sim_neg = F.cosine_similarity(f_local_occluded, f_global, dim=-1) / tau

    # 对正样本计算exp(similarity)
    exp_sim_pos = torch.exp(sim_pos)

    # 对负样本计算exp(similarity)
    exp_sim_neg = torch.exp(sim_neg)

    # 计算对比学习损失
    loss = -torch.log(exp_sim_pos / (exp_sim_pos + exp_sim_neg.sum(dim=-1)))

    # 返回批次平均损失
    return loss.mean()


# 示例使用
batch_size = 32
feature_dim = 128

# 随机生成未遮挡局部特征、遮挡局部特征和全局特征作为示例
f_local_non_occluded = torch.randn(batch_size, feature_dim)
f_global = torch.randn(batch_size, feature_dim)
f_local_occluded = torch.randn(batch_size, feature_dim)

# 计算损失
loss = contrastive_loss(f_local_non_occluded, f_global, f_local_occluded)
print(f'Contrastive Loss: {loss.item()}')
