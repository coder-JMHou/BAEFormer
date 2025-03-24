import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np


class Proj(nn.Module):
    def __init__(self, in_features, out_features=32):
        super(Proj, self).__init__()
        self.proj1 = nn.Conv2d(in_features, out_features, 3, 1, 1)
        self.act = nn.ReLU()
        self.proj2 = nn.Conv2d(out_features, out_features, 3, 1, 1)

    def forward(self, x):
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        return x


class ProjOut(nn.Module):
    def __init__(self, in_features=32, out_features=8):
        super(ProjOut, self).__init__()
        self.proj = nn.Conv2d(in_features, out_features, 3, 1, 1)
        self.act = nn.ReLU()

    def forward(self, out):
        out = self.act(out)
        out = self.proj(out)
        return out


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):  # [b c h w]
        x = x.permute(0, 2, 3, 1)  # [b c h w]->[b h w c]
        mu = x.mean(-1, keepdim=True)  # 计算c的平均
        sigma = x.var(-1, keepdim=True, unbiased=False)  # 计算c的方差
        x = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        x = x.permute(0, 3, 1, 2)  # [b h w c]->[b c h w]
        return x


class Attention(nn.Module):
    def __init__(self, in_features, heads=8, attn_type='height'):
        super(Attention, self).__init__()
        self.attn_type = attn_type
        self.scale = in_features ** -0.5
        self.heads = heads
        self.query_conv = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1)

    def forward(self, pan, lms):
        b, c, h, w = pan.size()  # [B,C,H,W]
        query = self.query_conv(pan)  # [B,C,H,W]
        key = self.key_conv(lms)  # [B,C,H,W]
        value = self.value_conv(lms)  # [B,C,H,W]
        if self.attn_type == 'height':
            query = query.permute(0, 3, 2, 1).reshape(b * w, h, self.heads, c // self.heads).permute(0, 2, 1,
                                                                                                     3)  # [b,w,h,c]->[bw,N,h,c]
            key = key.permute(0, 3, 2, 1).reshape(b * w, h, self.heads, c // self.heads).permute(0, 2, 1,
                                                                                                 3)  # [b,w,h,c]->[bw,N,h,c]
            value = value.permute(0, 3, 2, 1).reshape(b * w, h, self.heads, c // self.heads).permute(0, 2, 1,
                                                                                                     3)  # [b,w,h,c]->[bw,N.h,c]
        else:
            query = query.permute(0, 2, 3, 1).reshape(b * h, w, self.heads, c // self.heads).permute(0, 2, 1,
                                                                                                     3)  # [b,h,w,c]->[bh,N,w,c]
            key = key.permute(0, 2, 3, 1).reshape(b * h, w, self.heads, c // self.heads).permute(0, 2, 1,
                                                                                                 3)  # [b,h,w,c]->[bh,N,w,c]
            value = value.permute(0, 2, 3, 1).reshape(b * h, w, self.heads, c // self.heads).permute(0, 2, 1,
                                                                                                     3)  # [b,h,w,c]->[bh,N,w,c]
        attn = (query @ key.transpose(-2, -1)) * self.scale  # [bw,N,h,h] or [bh,N,w,w]
        attn = attn.softmax(dim=-1)
        output = attn @ value  # [bw,N,h,c] or [bh,N,w,c]
        if self.attn_type == 'height':
            output = output.reshape(b, w, self.heads, h, c // self.heads).permute(0, 2, 4, 3, 1).reshape(b, c, h,
                                                                                                         w)  # [b,c,h,w]
        else:
            output = output.reshape(b, h, self.heads, w, c // self.heads).permute(0, 2, 4, 1, 3).reshape(b, c, h,
                                                                                                         w)  # [b,c,h,w]
        output = self.out_conv(output)  # [b c h w]
        return output, attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU):
        super(Mlp, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.act = act_layer()
        self.conv21 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0, stride=1,
                            groups=hidden_features)
        self.conv22 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=3 // 2, stride=1,
                            groups=hidden_features)
        self.conv23 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, padding=5 // 2, stride=1,
                            groups=hidden_features)
        self.conv24 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, padding=7 // 2, stride=1,
                            groups=hidden_features)
        self.conv3 = nn.Conv2d(hidden_features * 4, in_features, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x1 = self.conv21(x)
        x2 = self.conv22(x)
        x3 = self.conv23(x)
        x4 = self.conv24(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.act(x)
        x = self.conv3(x)
        return x


class Adaptive_kernel(nn.Module):
    def __init__(self, k_size=3, heads=8, kernel_type='height'):
        super(Adaptive_kernel, self).__init__()
        self.kernel_size = k_size
        self.kernel_type = kernel_type
        self.heads = heads
        self.conv1 = nn.Conv2d(in_channels=heads, out_channels=k_size, kernel_size=(1, k_size), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=k_size, out_channels=k_size, kernel_size=3,padding=1, groups=k_size)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_features=1, out_features=k_size)
        self.norm = nn.LayerNorm(heads)
        self.act1 = nn.ReLU()
     

    def forward(self, attn, value):
        b, n, p, p = attn.size()                                    # [bw n h h] or [bh n w w]
        k = self.kernel_size
        attn = self.conv1(attn)  # [bw k h h]
        attn = self.act1(attn)
        attn = self.conv2(attn)
        attn = self.act1(attn)

        attn = attn.permute(0, 2, 3, 1).reshape(b, 1, p, p, 1, k).repeat([1, n, 1, 1, 1, 1])#.to('cpu')  # [bw n h h 1 k]
        value = self.pool(value)                                   # [bw n 1 1]
        value = self.linear(value).reshape(b, n, 1, 1, 1, k).repeat([1, 1, p, p, 1, 1])    # [bw n h h 1 k]
        adaptive_kernel = torch.mul(attn, value)                                              # [bw,n,h,h,1,k]
        return adaptive_kernel


class Adaptive_Attention(nn.Module):
    def __init__(self, in_features, k_size=3, padding=1, heads=8, attn_type='height'):
        super(Adaptive_Attention, self).__init__()
        self.kernel_size = k_size
        self.padding = padding
        self.heads = heads
        self.attn_type = attn_type
        self.value_conv = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1)
        self.adaptive = Adaptive_kernel(k_size, kernel_type=attn_type, heads=heads)
        self.out_conv = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1)

    def forward(self, output, attn):  # [b,c,h,w]  [bw,n,h,h] or [bh,n,w,w]
        b, c, h, w = output.size()
        value = self.value_conv(output)  # [b c h w]
        if self.attn_type == 'height':
            value = value.permute(0, 3, 2, 1).reshape(b * w, h, self.heads, -1).permute(0, 2, 1,
                                                                                        3)  # [b c h w]->[b w h c]->[bw n h c]
        else:
            value = value.permute(0, 2, 3, 1).reshape(b * h, w, self.heads, -1).permute(0, 2, 1,
                                                                                        3)  # [b c h w]->[b h w c]->[bh n w c]
        self.adaptive_kernel = self.adaptive(attn, value)  # [bw n h h 1 k]
        attn = self.adaptive_conv(attn)
        output = attn @ value  # [bw n h c] or [bh n w c]
        if self.attn_type == 'height':
            output = output.reshape(b, w, self.heads, h, c // self.heads).permute(0, 2, 4, 3, 1).reshape(b, c, h,
                                                                                                         w)  # [b,c,h,w]
        else:
            output = output.reshape(b, h, self.heads, w, c // self.heads).permute(0, 2, 4, 1, 3).reshape(b, c, h,
                                                                                                         w)  # [b,c,h,w]
        output = self.out_conv(output)
        return output, attn

    def adaptive_conv(self, attn):  # [bw n h h] or [bh n  w w]
        b, n, p, p = attn.size()  # p=h or p=w
        pad = self.padding
        k = self.kernel_size
        kernel = self.adaptive_kernel  # [bw n h h 1 k] or [bh n w w 1 k]
        attn_pad = torch.zeros(b, n, p, p + 2 * pad).to(attn.device)  # [bw n h h+2]
        attn_pad[:, :, :, pad:-pad] = attn
        # attn = F.pad(attn, [pad, pad], 'constant')
        attn = F.unfold(attn, (1, k))
        attn = attn.reshape(b, n, 1, k, p, p).permute(0, 1, 4, 5, 2, 3)  # [bw n 1 k h h]->[bw n h h 1 k]
        attn = torch.sum((attn * kernel), [4, 5])
        return attn


class Block1(nn.Module):
    def __init__(self,
                 in_features,
                 heads=8,
                 mlp_ratio=4.,
                 act_layer=nn.GELU):
        super(Block1, self).__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)
        self.attn_h = Attention(in_features, heads=heads, attn_type='height')
        self.attn_w = Attention(in_features, heads=heads, attn_type='width')
        self.norm3 = nn.LayerNorm(in_features)
        self.conv = nn.Conv2d(in_channels=in_features*2, out_channels=in_features, kernel_size=3,padding=1)
        mlp_hidden_features = int(in_features * mlp_ratio)
        self.mlp = Mlp(in_features=in_features, hidden_features=mlp_hidden_features, act_layer=act_layer)

    def forward(self, pan, lms):
        b, c, h, w = pan.size()
        pan_norm = pan.permute(0, 2, 3, 1).reshape(b, h * w, c)
        lms_norm = lms.permute(0, 2, 3, 1).reshape(b, h * w, c)
        pan_norm = self.norm1(pan_norm)
        lms_norm = self.norm2(lms_norm)
        pan_norm = pan_norm.reshape(b, h, w, c).permute(0, 3, 1, 2)  # b c h w
        lms_norm = lms_norm.reshape(b, h, w, c).permute(0, 3, 1, 2)  # b c h w
        output_h, attn_map_h = self.attn_h(pan_norm, lms_norm)
        output_w, attn_map_w = self.attn_w(pan_norm, lms_norm)

        output_h = lms + output_h
        output_w = lms + output_w

        output = self.conv(torch.cat((output_h, output_w), 1))
        output_norm = output.permute(0, 2, 3, 1).reshape(b, h*w, c)
        output_norm = self.norm3(output_norm)
        output_norm = output_norm.reshape(b, h, w, c).permute(0, 3, 1, 2)  # b c h w
        
        output = output + self.mlp(output_norm)

        return output, attn_map_h, attn_map_w


class Block2(nn.Module):
    def __init__(self,
                 in_features,
                 heads=8,
                 act_layer=nn.GELU):
        super(Block2, self).__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.attn_h = Adaptive_Attention(in_features, heads=heads, attn_type='height')
        self.attn_w = Adaptive_Attention(in_features, heads=heads, attn_type='width')
        self.norm3 = nn.LayerNorm(in_features)
        self.conv = nn.Conv2d(in_channels=in_features*2, out_channels=in_features, kernel_size=3,padding=1)
        mlp_hidden_features = int(in_features * 4)
        self.mlp = Mlp(in_features=in_features, hidden_features=mlp_hidden_features, act_layer=act_layer)

    def forward(self, output, attn_map_h, attn_map_w):
        b, c, h, w = output.size()
        output_norm = output.permute(0, 2, 3, 1).reshape(b, h * w, c)  # b c h w->b h*w c
        output_norm = self.norm1(output_norm)
        output_norm = output_norm.reshape(b, h, w, c).permute(0, 3, 1, 2)

        output_h, attn_map_h = self.attn_h(output_norm, attn_map_h)
        output_w, attn_map_w = self.attn_w(output_norm, attn_map_w)

        output_h = output + output_h
        output_w = output + output_w
        
        output = self.conv(torch.cat((output_h, output_w), 1))
        output_norm = output.permute(0, 2, 3, 1).reshape(b, h*w, c)
        output_norm = self.norm3(output_norm)
        output_norm = output_norm.reshape(b, h, w, c).permute(0, 3, 1, 2)  # b c h w
        
        output = output + self.mlp(output_norm)

        return output, attn_map_h, attn_map_w


class Net(nn.Module):
    def __init__(self, pan_features=1, lms_features=8, in_features=32, heads=8):
        super(Net, self).__init__()
        self.proj_pan = Proj(pan_features, in_features)
        self.proj_lms = Proj(lms_features, in_features)
        self.proj_out = ProjOut(in_features, lms_features)
        self.block0 = Block1(in_features, heads)
        self.block1 = Block2(in_features, heads)
        self.block2 = Block2(in_features, heads)
        self.block3 = Block2(in_features, heads)
        self.block4 = Block2(in_features, heads)
        self.conv1 = nn.Conv2d(lms_features, in_features, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_features, in_features, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_features, in_features, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_features, in_features, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_features, in_features, 3, 1, 1)
        self.actor = nn.ReLU()

    def forward(self, pan, lms):
        pan_proj = self.proj_pan(pan)
        lms_proj = self.proj_lms(lms)
        pan = pan.repeat([1, 8, 1, 1])
        x = pan - lms
        
        out, attn_map_h, attn_map_w = self.block0(pan_proj, lms_proj)
        x = self.conv1(x)
        x = self.actor(x)
        out = out + x

        out, attn_map_h, attn_map_w = self.block1(out, attn_map_h, attn_map_w)
        x = self.conv2(x)
        x = self.actor(x)
        out = out + x

        out, attn_map_h, attn_map_w = self.block2(out, attn_map_h, attn_map_w)
        x = self.conv3(x)
        x = self.actor(x)
        out = out + x

        out, attn_map_h, attn_map_w = self.block3(out, attn_map_h, attn_map_w)
        x = self.conv4(x)
        x = self.actor(x)
        out = out + x

        out, attn_map_h, attn_map_w = self.block4(out, attn_map_h, attn_map_w)
        x = self.conv5(x)
        x = self.actor(x)
        out = out + x

        output = self.proj_out(out)
        return lms + output