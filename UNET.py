import torch
import torch.nn as nn
import torch.nn.functional as F

#defining different component modules
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None, residual = False):   #residual takes care of residual connections
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.residual = residual
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = mid_channels, kernel_size = 3, padding = 1, stride = 1),
                                         nn.GroupNorm(num_groups = 8, num_channels = mid_channels),
                                         nn.GELU(),
                                         nn.Conv2d(in_channels = mid_channels, out_channels = out_channels, kernel_size = 3, padding = 1, stride = 1),
                                         nn.GroupNorm(num_groups = 8, num_channels = out_channels),)
        
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim = 256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(kernel_size = 2),
                                          DoubleConv(in_channels = in_channels, out_channels = in_channels, residual = True),
                                          DoubleConv(in_channels = in_channels, out_channels = out_channels),)
        self.emb_layer = nn.Sequential(nn.SiLU(),
                                       nn.Linear(in_features = emb_dim, out_features = out_channels),)
        
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        emb = self.emb_layer(t)[:, :, None, None]
        
        return x + emb
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim = 256):
        super().__init__()
        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv = nn.Sequential(DoubleConv(in_channels = in_channels, out_channels = in_channels, residual = True),
                                  DoubleConv(in_channels = in_channels, out_channels = out_channels, mid_channels = in_channels//2),)
        self.emb_layer = nn.Sequential(nn.SiLU(),
                                       nn.Linear(in_features = emb_dim, out_features = out_channels),)
        
    def forward(self, x, skip_x, t):
        x = self.up(x)
        #skip connects
        x = torch.cat([skip_x, x], dim = 1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        
        return x + emb
    
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim = channels, num_heads = 4, batch_first = True)
        self.ln = nn.LayerNorm(normalized_shape = [channels])
        self.ff_self = nn.Sequential(nn.LayerNorm([channels]),
                                     nn.Linear(in_features = channels, out_features = channels),
                                     nn.GELU(),
                                     nn.Linear(in_features = channels, out_features = channels),) 
        
    def forward(self, x):
        # x = x.view(-1, self.channels, self.size*self.size).swapaxes(1, 2)
        # x_ln = self.ln(x)
        # attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        # attention_value = attention_value + x
        # attention_value = self.ff_self(attention_value) + attention_value
        
        # return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
        B, C, H, W = x.shape

        #flatten spatial dimensions
        x = x.view(B, C, H*W).transpose(1,2)    #BxHWxC

        x_ln = self.ln(x)
        attn, _ = self.mha(x_ln, x_ln, x_ln)
        x = attn + x
        x = self.ff_self(x) + x
        #reshape back
        x = x.transpose(1, 2).view(B, C, H, W)

        return x

class UNet(nn.Module):
    def __init__(self, device, c_in = 6, c_out = 3, img_size= 512, time_dim = 256, downattn= [True, True, True], upattn = [True, True, True]):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.down_attn = downattn
        self.up_attn = upattn
        self.img_size = img_size
        
        #initial convolution
        self.inc = DoubleConv(c_in, 32)
        
        #Down Layers
        #Down Sampling 1
        self.down1 = Down(32, 64)
        #Self Attention 1
        self.sa1 = SelfAttention(64, self.img_size//2) if self.down_attn[0] else nn.Identity()
        #Down Sampling 2
        self.down2 = Down(64, 128)
        #Self Attention 1
        self.sa2 = SelfAttention(128, self.img_size//4) if self.down_attn[1] else nn.Identity()
        #Down Sampling 3
        self.down3 = Down(128, 256)
        #Self Attention 3
        self.sa3 = SelfAttention(256, self.img_size//8) if self.down_attn[2] else nn.Identity()
        #Down Sampling 4
        self.down4 = Down(256, 512)
        #Self Attention 4
        self.sa4 = SelfAttention(512, self.img_size//16) if self.down_attn[2] else nn.Identity()

        
        #NOTE: for SelfAttention layers, the first argument is Channel Dimension and the second is the Image Resolution
        
        #Middle Layers
        self.mid1 = DoubleConv(512, 1024)
        self.mid2 = DoubleConv(1024, 1024)
        self.mid3 = DoubleConv(1024, 512)
        
        #Up Layers
        #Up Sampling 1
        self.up1 = Up(768, 256)
        # self.sa4 = SelfAttention(128, 16)
        self.sa5 = SelfAttention(256, self.img_size//4) if self.up_attn[0] else nn.Identity()
        #Up Sampling 1
        self.up2 = Up(384, 128)
        # self.sa5 = SelfAttention(64, 32)
        self.sa6 = SelfAttention(128, self.img_size//2) if self.up_attn[1] else nn.Identity()
        #Up Sampling 1
        self.up3 = Up(192, 64)
        # self.sa6 = SelfAttention(64, 64)
        self.sa7 = SelfAttention(64, self.img_size) if self.up_attn[2] else nn.Identity()

        self.up4 = Up(96, 32)
        # self.sa6 = SelfAttention(64, 64)
        self.sa8 = SelfAttention(32, self.img_size) if self.up_attn[2] else nn.Identity()
        
        #Final Convolution
        self.outc = nn.Conv2d(in_channels = 32, out_channels = c_out, kernel_size = 1, padding = 0)
        
    #function to return time(position) encoding
    def pos_encoding(self, t, channels):
        inv_freq = 1.0/(10000**(torch.arange(start = 0, end = channels, step = 2, device = self.device).float()/channels))
        
        pos_enc = torch.cat([torch.sin(t.repeat(1, channels//2)*inv_freq), torch.sin(t.repeat(1, channels//2)*inv_freq)], dim = -1)
        
        return pos_enc
    
    def forward(self, x, t):
        #Getting the time embedding
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
            
        #Down Sampling stream, saving o/ps for skip connects
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x5 = self.down4(x4, t)
        x5 = self.sa4(x5)
        
        #Middle layer
        x5 = self.mid1(x5)
        x5 = self.mid2(x5)
        x5 = self.mid3(x5)
        
        #Up sampling stream with skip connects
        x = self.up1(x5, x4, t)
        x = self.sa5(x)
        x = self.up2(x, x3, t)
        x = self.sa6(x)
        x = self.up3(x, x2, t)
        x = self.sa7(x)
        x = self.up4(x, x1, t)
        x = self.sa8(x)
        
        x = self.outc(x)
        
        return x