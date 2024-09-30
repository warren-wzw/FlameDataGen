import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.utils import save_image

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, stride=1,padding=0, kernels_per_layer=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, C, height, width = x.size()
        # Query, Key, Value
        Q = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, N, C//8)
        K = self.key_conv(x).view(batch_size, -1, height * width)  # (B, C//8, N)
        V = self.value_conv(x).view(batch_size, -1, height * width)  # (B, C, N)
        # Attention
        attention = F.softmax(torch.bmm(Q, K), dim=1)  # (B, N, N)
        out = torch.bmm(V, attention.permute(0, 2, 1))  # (B, C, N) 
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h) #X*Wq
        k = self.proj_k(h) #X*Wk
        v = self.proj_v(h) #X*Wv

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))#QKt/sqrt(dk)[1,4096,4096]
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)#softmax(QKt/sqrt(dk))

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v) #softmax(QKt/sqrt(dk))*V=Attention(Q,K,V)=head
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        out =  h + x #resnet
        return out
        
class TransformerEncoder(nn.Module):
    
    def __init__(self, in_channels=512, num_heads=4, dim_feedforward=512, num_layers=3):
        super(TransformerEncoder, self).__init__()
        
        # Patch embedding: Flatten spatial dimensions and map to reduced feature dimension
        self.patch_embed = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels // 2, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reshape back to original dimensions after Transformer processing
        self.unpatch_embed = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
    
    def forward(self, x):
        # x: (B, C, H, W)
        batch_size, C, H, W = x.shape
        
        # Patch Embedding: Flatten the spatial dimensions
        x = self.patch_embed(x).view(batch_size, C // 2, -1).permute(2, 0, 1)  # (N, B, C)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)  # (N, B, C)
        
        # Reshape back to (B, C, H, W)
        x = x.permute(1, 2, 0).view(batch_size, C // 2, H, W)
        x = self.unpatch_embed(x)
        
        return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, padding=0,relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]
      
"""model"""
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, TimeStep, d_model, dim): #d_model=channel dim=channel * 4
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(TimeStep).float()
        emb = pos[:, None] * emb[None, :] #[500,1]*[1,64]=[500,64]
        assert list(emb.shape) == [TimeStep, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)#[500,64,2]
        assert list(emb.shape) == [TimeStep, d_model // 2, 2]
        emb = emb.view(TimeStep, d_model) #[500,dmodel]

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False), #[batch,dmodel]
            nn.Linear(d_model, dim), #[batch,dim]
            Swish(),
            nn.Linear(dim, dim),) #[batch,dim]
        
    def forward(self, t):
        emb = self.timembedding(t)
        return emb

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim): #d_model=channel dim=channel * 4
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
    def forward(self, t):
        emb = self.condEmbedding(t)
        return emb

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb, cemb):
        x = self.c1(x) + self.c2(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x, temb, cemb):
        _, _, H, W = x.shape
        x = self.t(x)
        x = self.c(x)
        return x
            
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = SelfAttentionBlock(out_ch) 
            #self.attn = nn.Identity()
        else:
            self.attn = nn.Identity()


    def forward(self, x, temb, labels):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h += self.cond_proj(labels)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h
    
class UNet(nn.Module):
    def __init__(self, TimeStep, num_labels, channel, ch_mult, num_res_blocks, dropout):
        #channel=128,ch_mult=[1, 2, 2, 2],num_res_blocks=2
        super().__init__()
        tdim = channel * 4
        self.time_embedding = TimeEmbedding(TimeStep, channel, tdim)
        self.cond_embedding = ConditionalEmbedding(num_labels, channel, tdim)
        self.head = nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [channel]  # record output channel when dowmsample for upsample
        now_ch = channel
        for i, mult in enumerate(ch_mult):
            out_ch = channel * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = channel * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )

    def forward(self, x, t, labels): #
        """Timestep embedding"""
        temb = self.time_embedding(t)
        cemb = self.cond_embedding(labels)
        """Downsampling"""
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)
        """Middle"""
        for layer in self.middleblocks:
            h = layer(h, temb, cemb)   
        """Upsampling"""
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)
        h = self.tail(h)

        assert len(hs) == 0
        return h

class UNet_self(nn.Module):
    def __init__(self, TimeStep, num_labels, channel, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = channel * 4
        self.time_embedding = TimeEmbedding(TimeStep, channel, tdim)
        self.cond_embedding = ConditionalEmbedding(num_labels, channel, tdim)
        self.head = nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [channel]  # record output channel when dowmsample for upsample
        now_ch = channel
        for i, mult in enumerate(ch_mult):
            out_ch = channel * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)


        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = channel * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        
        self
        self.bottleneck=TransformerEncoder(in_channels=256)
        self.emb_proj = nn.Sequential(
            Swish(),
            nn.Linear(512, 256),
        )
 

    def forward(self, x, t, labels):
        """Timestep embedding"""
        temb = self.time_embedding(t)
        cemb = self.cond_embedding(labels)
        """Downsampling"""
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)
        """Middle"""
        # for layer in self.middleblocks:
        #     h = layer(h, temb, cemb)   
        emb=self.emb_proj(temb)[:, :, None, None]+self.emb_proj(cemb)[:, :, None, None]
        h=h+emb
        h=self.bottleneck(h)
        """Upsampling"""
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
    
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)#extarct alphat or  1-alphat
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T): #beta_1=0.0001 beta_T=0.028
        #Beat 1/t represent the gradual increase from smaller noise to larger noise within a time step range
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device) #random time step
        noise = torch.randn_like(x_0)
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()
        self.model = model
        self.T = T
        self.w = w
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            sampledImgs=torch.clip(x_t, -1, 1)
            sampledImgs = sampledImgs * 0.5 + 0.5  # return [0 ~ 1]
            save_image(sampledImgs, os.path.join("./output/output_images/",  f"Output.png"), nrow=5)
        return sampledImgs