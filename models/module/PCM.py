import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PCM(nn.Module):
    def __init__(self, num_classes=5, init_tau=10.0):
        super().__init__()
        self.num_classes = num_classes
        self.theta_beta = nn.Parameter(torch.tensor(-2.0))
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def forward(self, support_features, support_labels, query_features):
        s_gap = F.adaptive_avg_pool2d(support_features, 1).squeeze(-1).squeeze(-1)
        q_gap = F.adaptive_avg_pool2d(query_features, 1).squeeze(-1).squeeze(-1)

        C = s_gap.shape[1]
        device = s_gap.device

        p_init = torch.zeros(self.num_classes, C, device=device)
        for c in range(self.num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                p_init[c] = s_gap[mask].mean(dim=0)

        q_norm = F.normalize(q_gap, p=2, dim=1)
        p_norm = F.normalize(p_init, p=2, dim=1)
        cos_sim = torch.matmul(q_norm, p_norm.t())
        y_hat = F.softmax(self.tau * cos_sim, dim=1)

        entropy = -torch.sum(y_hat * torch.log(y_hat + 1e-8), dim=1)
        w_q = 1.0 - entropy / math.log(self.num_classes)

        w_q = w_q.unsqueeze(1)
        weighted_q = w_q * y_hat

        p_query_numerator = torch.matmul(weighted_q.t(), q_gap)
        p_query_denominator = weighted_q.sum(dim=0).unsqueeze(1) + 1e-8
        p_query = p_query_numerator / p_query_denominator

        beta = torch.sigmoid(self.theta_beta)
        p_cal = (1 - beta) * p_init + beta * p_query

        return p_cal