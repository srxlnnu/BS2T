import torch
from torch import nn, einsum

from einops import rearrange, repeat

# translated from tensorflow code
# https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2

# positional embedding helpers

def rel_to_abs(x):
    b, h, l, _, device, dtype = x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim = 3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l-1):]
    return final_x

def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = repeat(logits, 'b h x y j -> b h x i y j', i = h)
    return logits

# positional embeddings

class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(fmap_size, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(fmap_size, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(fmap_size * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(fmap_size * 2 - 1, dim_head) * scale)

    def forward(self, q):
        q = rearrange(q, 'b h (x y) d -> b h x y d', x = self.fmap_size)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes

class spectralAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))

        q *= self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim += self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return out


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(6, 1, bias=False),

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape
        # fmap1 = fmap.permute(0, 2, 3, 1)
        # fmap2 = fmap.permute(0, 2, 1, 3)
        # fmap3 = torch.matmul(fmap1, fmap2, out=None)

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        spe = self.avg_pool(fmap)
        q, k, v, spe = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v, spe))
        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        sim += self.pos_emb(q)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        s1,s2,s3,s4 = attn.shape
        simm = torch.reshape(attn, (s1,s2,s3*s4,1))
        sim2 = einsum('b h i d, b h d j -> b h i j', simm, spe)
        # 16 2 6561 6
        sim22 = sim2.mean(3)
        # attn2 = simwanl22.softmax(dim = -1)
        attn2 = torch.reshape(sim22, (s1,s2,s3,s4))
        # attn2 = attn2 + attn
        out2 = einsum('b h i j, b h j d -> b h i d', attn2, v)
        out3 = (out+out2)/2
        out = rearrange(out3, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return out


class BottleBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out,
        proj_factor,
        downsample,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()

        # shortcut

        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride = stride, padding = padding, bias = False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion

        attention_dim = dim_out // proj_factor

        self.net = nn.Sequential(
            nn.Conv2d(dim, attention_dim, 1, bias = False),
            nn.BatchNorm2d(attention_dim),
            activation,
            Attention(
                dim = attention_dim,
                fmap_size = fmap_size,
                heads = heads,
                dim_head = dim_head,
                rel_pos_emb = rel_pos_emb
            ),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(attention_dim),
            activation,
            nn.Conv2d(attention_dim, dim_out, 1, bias = False),
            nn.BatchNorm2d(dim_out)
        )

        # init last batch norm gamma to zero

        nn.init.zeros_(self.net[-1].weight)

        # final activation

        self.activation = activation

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)


class BottleStack(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out = 2048,
        proj_factor = 4,
        num_layers = 3,
        heads = 4,
        dim_head = 128,
        downsample = True,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()
        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)
            layer_downsample = is_first and downsample
            layer_fmap_size = fmap_size // (2 if downsample and not is_first else 1)

            layers.append(BottleBlock(
                dim = dim,
                fmap_size = layer_fmap_size,
                dim_out = dim_out,
                proj_factor = proj_factor,
                heads = heads,
                dim_head = dim_head,
                downsample = layer_downsample,
                rel_pos_emb = rel_pos_emb,
                activation = activation
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze(dim=4)
        _, c, h, w = x.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'
        assert h == self.fmap_size and w == self.fmap_size, f'height and width ({h} {w}) of feature map must match the fmap_size given at init {self.fmap_size}'
        result = self.net(x)
        return torch.unsqueeze(result, dim=4)

class crossBottleStack(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out = 2048,
        proj_factor = 4,
        num_layers = 3,
        heads = 4,
        dim_head = 128,
        downsample = True,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()
        self.dim = dim
        self.fmap_size = fmap_size

        i=0
        is_first = i == 0
        dim1 = (dim if is_first else dim_out)
        layer_downsample = is_first and downsample
        layer_fmap_size = fmap_size // (2 if downsample and not is_first else 1)
        self.net1= crossBottleBlock(
            dim = dim,
            fmap_size = layer_fmap_size,
            dim_out = dim_out,
            proj_factor = proj_factor,
            heads = heads,
            dim_head = dim_head,
            downsample = layer_downsample,
            rel_pos_emb = rel_pos_emb,
            activation = activation
        )
        i=1
        is_first = i == 0
        dim = (dim if is_first else dim_out)
        layer_downsample = is_first and downsample
        layer_fmap_size = fmap_size // (2 if downsample and not is_first else 1)
        self.net2= crossBottleBlock(
            dim = dim,
            fmap_size = layer_fmap_size,
            dim_out = dim_out,
            proj_factor = proj_factor,
            heads = heads,
            dim_head = dim_head,
            downsample = layer_downsample,
            rel_pos_emb = rel_pos_emb,
            activation = activation
        )
        i=2
        is_first = i == 0
        dim = (dim if is_first else dim_out)
        layer_downsample = is_first and downsample
        layer_fmap_size = fmap_size // (2 if downsample and not is_first else 1)
        self.net3= crossBottleBlock(
            dim = dim,
            fmap_size = layer_fmap_size,
            dim_out = dim_out,
            proj_factor = proj_factor,
            heads = heads,
            dim_head = dim_head,
            downsample = layer_downsample,
            rel_pos_emb = rel_pos_emb,
            activation = activation
        )

    def forward(self, x1, x2):
        x1 = x1.squeeze(dim=4)
        x2 = x2.squeeze(dim=4)
        _, c, h, w = x1.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'
        assert h == self.fmap_size and w == self.fmap_size, f'height and width ({h} {w}) of feature map must match the fmap_size given at init {self.fmap_size}'
        for i in range(3):
            x1, x2 = self.net1(x1, x2)
            x1, x2 = self.net2(x1, x2)
            x1, x2 = self.net3(x1, x2)
        return torch.unsqueeze(x1, dim=4)
class  selfatten(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out = 2048,
        proj_factor = 4,
        num_layers = 3,
        heads = 4,
        dim_head = 128,
        downsample = True,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()
        self.dim = dim
        self.fmap_size = fmap_size
        attention_dim = dim_out
        self.net =Attention(
                dim = attention_dim,
                fmap_size = fmap_size,
                heads = heads,
                dim_head = dim_head,
                rel_pos_emb = rel_pos_emb
            )
    def forward(self, x):
        x = x.squeeze(dim=4)
        _, c, h, w = x.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'
        assert h == self.fmap_size and w == self.fmap_size, f'height and width ({h} {w}) of feature map must match the fmap_size given at init {self.fmap_size}'
        out = self.net(x)
        return torch.unsqueeze(out, dim=4)