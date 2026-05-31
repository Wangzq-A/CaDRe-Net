class MPPG(nn.Module):
    def __init__(self, in_channels, mask_ratio=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio

        self.W_Q = nn.Linear(in_channels, in_channels, bias=False)
        self.W_K = nn.Linear(in_channels, in_channels, bias=False)
        self.W_V = nn.Linear(in_channels, in_channels, bias=False)

        self.gate_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def generate_spatial_mask(self, B, H, W, device):
        num_patches = H * W
        num_mask = int(num_patches * self.mask_ratio)

        mask = torch.ones(B, num_patches, device=device)
        for i in range(B):
            mask_idx = torch.randperm(num_patches)[:num_mask]
            mask[i, mask_idx] = 0
        return mask.view(B, 1, H, W)

    def forward(self, query_features, p_cal):
        B, C, H, W = query_features.shape

        mask = self.generate_spatial_mask(B, H, W, query_features.device)
        f_q_tilde = query_features * mask

        f_q_flat = f_q_tilde.flatten(2).transpose(1, 2)

        Q = self.W_Q(f_q_flat)
        K = self.W_K(p_cal)
        V = self.W_V(p_cal)

        attention_scores = torch.matmul(Q, K.t()) / math.sqrt(C)
        A = F.softmax(attention_scores, dim=-1)

        f_rec_flat = torch.matmul(A, V)
        f_rec = f_rec_flat.transpose(1, 2).view(B, C, H, W)

        f_cat = torch.cat([query_features, f_rec], dim=1)
        G = torch.sigmoid(self.gate_conv(f_cat))

        f_q_hat = G * query_features + (1 - G) * f_rec

        return f_q_hat, f_rec