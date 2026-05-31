import torch
import torch.nn as nn
import torch.nn.functional as F


class CaDReNet(nn.Module):
    def __init__(self, backbone, pcm, mppg, way, shots, tau_cls=10.0):
        super().__init__()
        self.backbone = backbone
        self.pcm = pcm
        self.mppg = mppg
        self.way = way
        self.shots = shots
        self.tau_cls = nn.Parameter(torch.tensor(tau_cls))

    def forward(self, inp):
        shot = self.shots[0]

        support_images = inp[:self.way * shot]
        query_images = inp[self.way * shot:]

        support_l4 = self.backbone(support_images)
        query_l4 = self.backbone(query_images)

        support_labels = torch.arange(self.way).repeat_interleave(shot).cuda()

        p_cal = self.pcm(support_l4, support_labels, query_l4)

        if self.training:
            f_q_hat, f_rec = self.mppg(query_l4, p_cal)
            q_gap = F.adaptive_avg_pool2d(f_q_hat, 1).squeeze(-1).squeeze(-1)
        else:
            q_gap = F.adaptive_avg_pool2d(query_l4, 1).squeeze(-1).squeeze(-1)
            f_rec = query_l4

        q_norm = F.normalize(q_gap, p=2, dim=1)
        p_norm = F.normalize(p_cal, p=2, dim=1)
        cos_sim = torch.matmul(q_norm, p_norm.t())

        log_prediction = F.log_softmax(self.tau_cls * cos_sim, dim=1)

        return log_prediction, query_l4, f_rec

    def meta_test(self, inp, way, shot, query_shot):
        support_images = inp[:way * shot]
        query_images = inp[way * shot:]

        support_l4 = self.backbone(support_images)
        query_l4 = self.backbone(query_images)

        support_labels = torch.arange(way).repeat_interleave(shot).cuda()

        p_cal = self.pcm(support_l4, support_labels, query_l4)

        q_gap = F.adaptive_avg_pool2d(query_l4, 1).squeeze(-1).squeeze(-1)

        q_norm = F.normalize(q_gap, p=2, dim=1)
        p_norm = F.normalize(p_cal, p=2, dim=1)
        cos_sim = torch.matmul(q_norm, p_norm.t())

        _, max_index = torch.max(cos_sim, 1)

        return max_index
