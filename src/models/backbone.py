import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from timm.models.layers import DropPath, trunc_normal_

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    print("WARNING: mamba_ssm not installed. The model will fail at runtime.")
    selective_scan_fn = None

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)

class CrossTemporalSS2D(nn.Module):
    def __init__(
        self, 
        d_model, 
        d_state=16, 
        ssm_ratio=2.0, 
        dt_rank="auto", 
        act_layer=nn.SiLU,
        d_conv=3,
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ssm_ratio = ssm_ratio
        self.d_inner = int(self.ssm_ratio * d_model)
        self.d_conv = d_conv
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.act = act_layer()

        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.d_inner * 2, self.d_inner * 2, kernel_size=3, padding=1, groups=self.d_inner*2, bias=False),
            LayerNorm2d(self.d_inner * 2),
            nn.SiLU()
        )
        self.fusion_mix = nn.Conv2d(self.d_inner * 2, self.d_inner * 2, kernel_size=1, bias=False)
        
        nn.init.constant_(self.fusion_mix.weight, 0)
        if self.fusion_mix.bias is not None:
            nn.init.constant_(self.fusion_mix.bias, 0)

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        self.alpha = nn.Parameter(torch.zeros(1))

        K = 4 
        self.K = K
        
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> k d n",
            k=K,
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A) 
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(K, self.d_inner))
        
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.scan_merge_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)

    def forward_core_ssm(self, x, dt, A, B, C, D):
        # x: [Batch, K, L, D]
        # dt: [Batch, K, L, D]
        # A: [K, D, N]
        # B: [Batch, K, L, N]
        # C: [Batch, K, L, N]
        # D: [K, D]
        
        batch_size, K, L, D_inner = x.shape
        ys = []
        
        for k in range(K):
            # [FIX] Transpose inputs to (B, D, L) for selective_scan_fn
            # x_k: [B, L, D] -> [B, D, L]
            x_k = x[:, k].transpose(1, 2).contiguous() 
            dt_k = dt[:, k].transpose(1, 2).contiguous()
            
            A_k = A[k] # [D, N]
            D_k = D[k] # [D]
            
            # B_k, C_k: [B, L, N] -> [B, N, L]
            B_k = B[:, k].transpose(1, 2).contiguous()
            C_k = C[:, k].transpose(1, 2).contiguous()
            
            y_k = selective_scan_fn(
                x_k, dt_k, A_k, B_k, C_k, D_k.float(), 
                z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True
            ) 
            # Output y_k is [B, D, L] -> Transpose back to [B, L, D]
            ys.append(y_k.transpose(1, 2))
            
        return torch.stack(ys, dim=1)

    def forward(self, x_t1, x_t2):
        B, H, W, C = x_t1.shape
        L = H * W

        xz_t1 = self.in_proj(x_t1.view(B, L, C))
        xz_t2 = self.in_proj(x_t2.view(B, L, C))
        
        x_t1, z_t1 = xz_t1.chunk(2, dim=-1)
        x_t2, z_t2 = xz_t2.chunk(2, dim=-1)

        x_t1 = x_t1.transpose(1, 2).view(B, self.d_inner, H, W)
        x_t2 = x_t2.transpose(1, 2).view(B, self.d_inner, H, W)

        x_t1 = self.act(self.conv2d(x_t1))
        x_t2 = self.act(self.conv2d(x_t2))

        x_joint = torch.cat([x_t1, x_t2], dim=1)
        delta_joint = self.fusion_mix(self.fusion_conv(x_joint))
        delta_1, delta_2 = delta_joint.chunk(2, dim=1)
        
        x_t1_mixed = x_t1 + delta_1
        x_t2_mixed = x_t2 + delta_2

        x_t1_mixed = x_t1_mixed.permute(0, 2, 3, 1)
        x_t2_mixed = x_t2_mixed.permute(0, 2, 3, 1)

        def scan_expand(x):
            v1 = x.view(B, -1, self.d_inner)
            v2 = x.flip([1, 2]).view(B, -1, self.d_inner)
            v3 = x.transpose(1, 2).contiguous().view(B, -1, self.d_inner)
            v4 = x.transpose(1, 2).flip([1, 2]).contiguous().view(B, -1, self.d_inner)
            return torch.stack([v1, v2, v3, v4], dim=1)

        x1_scan = scan_expand(x_t1_mixed)
        x2_scan = scan_expand(x_t2_mixed)

        x1_flat = x1_scan.view(B * 4, L, self.d_inner)
        x2_flat = x2_scan.view(B * 4, L, self.d_inner)
        
        ssm_params_t1 = self.x_proj(x1_flat)
        ssm_params_t2 = self.x_proj(x2_flat)
        
        mixing_rate = torch.tanh(self.alpha)
        
        mixed_params_t1 = ssm_params_t1 + mixing_rate * ssm_params_t2 
        mixed_params_t2 = ssm_params_t2 + mixing_rate * ssm_params_t1
        
        mixed_params_t1 = mixed_params_t1.view(B, 4, L, -1)
        mixed_params_t2 = mixed_params_t2.view(B, 4, L, -1)
        
        dt_t1, B_t1, C_t1 = torch.split(mixed_params_t1, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_t2, B_t2, C_t2 = torch.split(mixed_params_t2, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt_t1 = self.dt_proj(dt_t1)
        dt_t2 = self.dt_proj(dt_t2)

        A_final = -torch.exp(self.A_log.float())
        
        y_t1_scan = self.forward_core_ssm(x1_scan, dt_t1, A_final, B_t1, C_t1, self.D)
        y_t2_scan = self.forward_core_ssm(x2_scan, dt_t2, A_final, B_t2, C_t2, self.D)

        def scan_merge(y):
            y1 = y[:, 0].view(B, H, W, self.d_inner)
            y2 = y[:, 1].view(B, H, W, self.d_inner).flip([1, 2])
            y3 = y[:, 2].view(B, W, H, self.d_inner).transpose(1, 2)
            y4 = y[:, 3].view(B, W, H, self.d_inner).flip([1, 2]).transpose(1, 2)
            merged = y1 + y2 + y3 + y4 
            return self.scan_merge_proj(merged)

        y_t1 = scan_merge(y_t1_scan).view(B, L, self.d_inner)
        y_t2 = scan_merge(y_t2_scan).view(B, L, self.d_inner)

        y_t1 = self.out_norm(y_t1) * F.silu(z_t1)
        y_t2 = self.out_norm(y_t2) * F.silu(z_t2)

        return self.out_proj(y_t1), self.out_proj(y_t2)

class CT_VMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ct_ss2d = CrossTemporalSS2D(d_model, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1, x2, H, W):
        B, L, C = x1.shape
        n_x1 = self.norm1(x1).view(B, H, W, C)
        n_x2 = self.norm1(x2).view(B, H, W, C)
        out1, out2 = self.ct_ss2d(n_x1, n_x2)
        x1 = x1 + self.drop_path(out1)
        x2 = x2 + self.drop_path(out2)
        x1 = x1 + self.drop_path(self.ffn(self.norm2(x1)))
        x2 = x2 + self.drop_path(self.ffn(self.norm2(x2)))
        return x1, x2

class CTVMambaBackbone(nn.Module):
    def __init__(self, in_channels=3, embed_dims=[96, 192, 384, 768], depths=[2, 2, 9, 2], drop_rate=0.0, drop_path_rate=0.2):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = len(depths)
        self.stem_proj = nn.Conv2d(in_channels, embed_dims[0], kernel_size=4, stride=4)
        self.stem_norm = nn.LayerNorm(embed_dims[0])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        self.stages = nn.ModuleList()
        curr_idx = 0
        
        for i in range(self.num_layers):
            stage_blocks = nn.ModuleList([
                CT_VMambaBlock(d_model=embed_dims[i], dropout=drop_rate, drop_path=dpr[curr_idx + j]) 
                for j in range(depths[i])
            ])
            curr_idx += depths[i]
            downsample = None
            if i < self.num_layers - 1:
                downsample = nn.Sequential(
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=2, stride=2),
                    LayerNorm2d(embed_dims[i+1])
                )
            self.stages.append(nn.ModuleDict({'blocks': stage_blocks, 'downsample': downsample}))
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        outs_t1, outs_t2 = [], []
        B, C, H, W = x1.shape
        x1 = self.stem_proj(x1).flatten(2).transpose(1, 2)
        x2 = self.stem_proj(x2).flatten(2).transpose(1, 2)
        x1 = self.stem_norm(x1)
        x2 = self.stem_norm(x2)
        H, W = H // 4, W // 4

        for layer_dict in self.stages:
            blocks = layer_dict['blocks']
            downsample = layer_dict['downsample']
            for blk in blocks:
                x1, x2 = blk(x1, x2, H, W)
            f1 = x1.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            f2 = x2.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs_t1.append(f1)
            outs_t2.append(f2)
            if downsample is not None:
                x1_img = x1.view(B, H, W, -1).permute(0, 3, 1, 2)
                x2_img = x2.view(B, H, W, -1).permute(0, 3, 1, 2)
                x1_down = downsample[0](x1_img)
                x2_down = downsample[0](x2_img)
                H, W = H // 2, W // 2
                x1_down = downsample[1](x1_down)
                x2_down = downsample[1](x2_down)
                x1 = x1_down.flatten(2).transpose(1, 2)
                x2 = x2_down.flatten(2).transpose(1, 2)
        return outs_t1, outs_t2