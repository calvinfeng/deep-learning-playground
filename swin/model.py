import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn, einsum


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
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
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

        B, H, W, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        num_win_h = H // self.window_size # Number of windows along height axis
        num_win_w = W // self.window_size # Number of windows along width axis

        # Each query, key, value tensor is of shape
        # (batch_size, num_heads, num_windows, num_patches, head_dim)
        q, k, v = map(
            lambda t: rearrange(t, 'b (num_win_h win_h) (num_win_w win_w) (h d) -> b h (num_win_h num_win_w) (win_h win_w) d',
                                h=self.num_heads,
                                win_h=self.window_size,
                                win_w=self.window_size), qkv)

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
                        h=self.num_heads,
                        win_h=self.window_size,
                        win_w=self.window_size,
                        num_win_h=num_win_h,
                        num_win_w=num_win_w)

        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


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


class SwinBlock(nn.Module):
    def __init__(self, dim:int, num_heads:int, head_dim:int, mlp_dim:int, shifted:bool, window_size:int, relative_pos_embedding:bool):
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
        pass
