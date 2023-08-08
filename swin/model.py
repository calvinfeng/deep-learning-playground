import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn, einsum
from typing import List


class PatchMerging(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downscaling_factor: int):
        """_summary_

        Args:
            in_channels (int): Input channel size from the previous stage. This is 3 for the first stage because of RGB.
            out_channels (int): Output channel size for the next stage.
            downscaling_factor (int): Same as patch side length. Patch size is downscaling_factor x downscaling_factor.
        """
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        new_H, new_W = H // self.downscaling_factor, W // self.downscaling_factor
        x = self.patch_merge(x).view(B, -1, new_H, new_W)
        x = self.linear(x.permute(0, 2, 3, 1))
        return x


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, embed_dim: int, fn: function):
        """Apply layer norm to input and then apply the function.

        Args:
            embed_dim (int): Normalize the input to this dimension.
            fn (function): Function to apply after normalization.
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size**2, window_size**2)
    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size: int):
    """Compute distances between all pairs of points in a window.

    Args:
        window_size (int): Size of the window, which is number of patches in one dimension.

    Returns:
        torch.Tensor: Shape(window_size ** 2, window_size ** 2, 2) containing the
        distances between all pairs
    """
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, shifted: bool, window_size: int,
                 relative_pos_embedding: bool):
        """Perform self-attention on each window of the input. If shifted is True, the window is shifted by half of the
        window size. This allows "global attention" across stages. In essence, shifted window attention combines the
        ideas of local attention and global attention.

        Args:
            embed_dim (int): _description_
            num_heads (int): _description_
            head_dim (int): _description_
            shifted (bool): _description_
            window_size (int): _description_
            relative_pos_embedding (bool): _description_
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size,
                                                             displacement=displacement,
                                                             upper_lower=True,
                                                             left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size,
                                                            displacement=displacement,
                                                            upper_lower=False,
                                                            left_right=True), requires_grad=False)

        inner_dim = head_dim * num_heads
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)
        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))
        self.to_out = nn.Linear(inner_dim, embed_dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        # x is of shape (batch_size, num_patches_height, num_patches_width, embed_dim)
        B, H, W, _ = x.shape
        num_heads = self.num_heads
        window_size = self.window_size

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        num_win_h = H // self.window_size # Number of windows along height axis
        num_win_w = W // self.window_size # Number of windows along width axis

        # Each query, key, value tensor is of shape
        # (batch_size, num_heads, num_windows, num_patches, head_dim)
        eqn = 'b (num_win_h win_h) (num_win_w win_w) (h d) -> b h (num_win_h num_win_w) (win_h win_w) d'
        q, k, v = map(lambda t: rearrange(t, eqn, h=num_heads, win_h=window_size, win_w=window_size), qkv)

        # Dot product returns (batch_size, num_heads, num_windows, num_patches, num_patches)
        # It represents the similarity between patches[i] to all other patches[j] within a window.
        # That means, self-attention is only applied within a window.
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -num_win_w:] += self.upper_lower_mask
            dots[:, :, num_win_w - 1::num_win_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)

        out = rearrange(out, 'b h (num_win_h num_win_w) (win_h win_w) d -> b (num_win_h win_h) (num_win_w win_w) (h d)',
                        h=num_heads, win_h=window_size, win_w=window_size, num_win_h=num_win_h, num_win_w=num_win_w)

        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, head_dim:int, mlp_dim:int, shifted:bool,
                 window_size:int, relative_pos_embedding:bool):
        """_summary_

        Args:
            dim (int): _description_
            num_heads (int): _description_
            head_dim (int): _description_
            mlp_dim (int): _description_
            shifted (bool): _description_
            window_size (int): _description_
            relative_pos_embedding (bool): _description_
        """
        super().__init__()
        self.attention_block = Residual(
            PreNorm(embed_dim,
                    WindowAttention(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    head_dim=head_dim,
                                    shifted=shifted,
                                    window_size=window_size,
                                    relative_pos_embedding=relative_pos_embedding),
                    ),
            )
        self.mlp_block = Residual(PreNorm(embed_dim, FeedForward(input_dim=embed_dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, layers: int, downscaling_factor: int, num_heads: int,
                 head_dim: int, window_size: int, relative_pos_embedding: bool):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels,
                                            out_channels=hidden_dim,
                                            downscaling_factor=downscaling_factor)
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(embed_dim=hidden_dim, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dim * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(embed_dim=hidden_dim, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dim * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        # Patch partition will transform x in channel-last format.
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        # Stage module needs to return x in channel-first format.
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads:List[int], channels=3, num_classes=21, head_dim=32,
                 window_size=7, downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()
        self.stage1 = StageModule(in_channels=channels, hidden_dim=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dim=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dim=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dim=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)


def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)
