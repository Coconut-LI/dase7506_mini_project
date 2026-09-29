"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""
# from model import GPT


# def build_model(config):
#     return GPT(config)
import torch
from torch import nn
from torch.nn import functional as F

#说是白加白不加
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        x_compute = x.to(compute_dtype)
        rms = x_compute.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        normalized = (x_compute * rms).to(x.dtype)
        return normalized * self.weight.to(dtype=x.dtype)
    
#完全看不懂这块代码，只知道原理，后面学一下
def apply_rope(x):
    # x: (batch, heads, seq_len, head_dim)
    seq_len = x.size(-2)
    head_dim = x.size(-1)

    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        10000.0 ** (
            torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
            / head_dim
        )
    )
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]

    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1,
    )
    return rotated.flatten(-2)

class SwiGLU(nn.Module):
    def __init__(self, width, hidden_width):
        super().__init__()
        self.gate = nn.Linear(width, hidden_width)
        self.value = nn.Linear(width, hidden_width)
        self.proj = nn.Linear(hidden_width, width)

    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.value(x))

class Block(nn.Module):  #定义一个计算模块，用来处理数据
    def __init__(self,width=128,heads=4,dropout=0.0):#初始化
        super().__init__() #初始化继承的父类
        self.heads = heads
        self.norm1,self.norm2 = nn.RMSNorm(width), nn.RMSNorm(width) #定义归一化函数，后面方便使用
        self.qkv, self.proj = nn.Linear(width, 3*width), nn.Linear(width, width)#这里定义qkv发生的变化，就是把输入切3份。后者proj用来？我的印象是用它来做logits变可能性
        #self.mlp = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width)) #这里首先把输入表达扩大4倍，然后经过非线性，最后再回到原输入维度。这个整体是用来？
        self.mlp = SwiGLU(width, 5*width)
        head_dim = width // heads
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        # 各 Block 的 __init__ 增加 dropout=0.0 参数
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape #定义这里输入的每次训练多少序列，每个序列多少token，每个token用多少位表示
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width//self.heads).permute(2, 0, 3, 1, 4)#这里把qkv分成q,k,v然后调整维度的排列顺序，qkv-训练次数-头-单个token长度-width/头
        #加入rope修改
        q = apply_rope(q)
        k = apply_rope(k)
        #加入qk-norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True) #对这里做遮挡，不让模型看下一个词
        # x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width)) #相当于把头和单token长度交换，然后头和width/头重新整合
        # return x + self.mlp(self.norm2(x))   #这里相当于对于输入x做了3件事情：1.用qkv算了注意力 2.把注意力加回x 3.增加了注意力的x再次经过mlp对表达进行非线性化 3.什么是mlp？
        # x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width)) #相当于把头和单token长度交换，然后头和width/头重新整合
        # return x + self.mlp(self.norm2(x))
        x = x + self.resid_dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.resid_dropout(self.mlp(self.norm2(x)))
    
class Block1(nn.Module):  #定义一个计算模块，用来处理数据
    def __init__(self,width=128,heads=4,dropout=0.0):#初始化
        super().__init__() #初始化继承的父类
        self.heads = heads
        self.norm1,self.norm2 = nn.RMSNorm(width), nn.RMSNorm(width) #定义归一化函数，后面方便使用
        self.qkv, self.proj = nn.Linear(width, 3*width), nn.Linear(width, width)#这里定义qkv发生的变化，就是把输入切3份。后者proj用来？我的印象是用它来做logits变可能性
        #self.mlp = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width)) #这里首先把输入表达扩大4倍，然后经过非线性，最后再回到原输入维度。这个整体是用来？
        self.mlp = SwiGLU(width, 2*width)
        head_dim = width // heads
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        # 各 Block 的 __init__ 增加 dropout=0.0 参数
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape #定义这里输入的每次训练多少序列，每个序列多少token，每个token用多少位表示
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width//self.heads).permute(2, 0, 3, 1, 4)#这里把qkv分成q,k,v然后调整维度的排列顺序，qkv-训练次数-头-单个token长度-width/头
        #加入rope修改
        q = apply_rope(q)
        k = apply_rope(k)
        #qk-norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True) #对这里做遮挡，不让模型看下一个词
        # x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width)) #相当于把头和单token长度交换，然后头和width/头重新整合
        # return x + self.mlp(self.norm2(x))
        x = x + self.resid_dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.resid_dropout(self.mlp(self.norm2(x)))

class Block2(nn.Module):  #定义一个计算模块，用来处理数据
    def __init__(self,width=128,heads=4,dropout=0.0):#初始化
        super().__init__() #初始化继承的父类
        self.heads = heads
        self.norm1,self.norm2 = nn.RMSNorm(width), nn.RMSNorm(width) #定义归一化函数，后面方便使用
        self.qkv, self.proj = nn.Linear(width, 3*width), nn.Linear(width, width)#这里定义qkv发生的变化，就是把输入切3份。后者proj用来？我的印象是用它来做logits变可能性
        self.mlp = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width)) #这里首先把输入表达扩大4倍，然后经过非线性，最后再回到原输入维度。这个整体是用来？
        #self.mlp = SwiGLU(width, 2*width)
        head_dim = width // heads
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        # 各 Block 的 __init__ 增加 dropout=0.0 参数
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape #定义这里输入的每次训练多少序列，每个序列多少token，每个token用多少位表示
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width//self.heads).permute(2, 0, 3, 1, 4)#这里把qkv分成q,k,v然后调整维度的排列顺序，qkv-训练次数-头-单个token长度-width/头
        #加入rope修改
        q = apply_rope(q)
        k = apply_rope(k)
        #qk-norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True) #对这里做遮挡，不让模型看下一个词
        # x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width)) #相当于把头和单token长度交换，然后头和width/头重新整合
        # return x + self.mlp(self.norm2(x))
        x = x + self.resid_dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.resid_dropout(self.mlp(self.norm2(x)))

#扩大了block1的attention
class Block3(nn.Module):  #定义一个计算模块，用来处理数据
    def __init__(self,width=128,heads=4,attn_width=None):#初始化
        super().__init__() #初始化继承的父类
        self.heads = heads
        self.attn_width = width if attn_width is None else attn_width
        if self.attn_width % heads != 0:
            raise ValueError("attn_width must be divisible by heads")

        self.head_dim = self.attn_width // heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")

        self.norm1,self.norm2 = nn.RMSNorm(width), nn.RMSNorm(width) #定义归一化函数，后面方便使用
        self.qkv, self.proj = nn.Linear(width, 3*self.attn_width), nn.Linear(self.attn_width, width)#这里定义qkv发生的变化，就是把输入切3份。后者proj用来？我的印象是用它来做logits变可能性
        #self.mlp = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width)) #这里首先把输入表达扩大4倍，然后经过非线性，最后再回到原输入维度。这个整体是用来？
        self.mlp = SwiGLU(width, 2*width)
        #head_dim = width // heads
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x):
        batch, length, width = x.shape #定义这里输入的每次训练多少序列，每个序列多少token，每个token用多少位表示
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)#这里把qkv分成q,k,v然后调整维度的排列顺序，qkv-训练次数-头-单个token长度-width/头
        #加入rope修改
        q = apply_rope(q)
        k = apply_rope(k)
        #qk-norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True) #对这里做遮挡，不让模型看下一个词
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, self.attn_width)) #相当于把头和单token长度交换，然后头和width/头重新整合
        return x + self.mlp(self.norm2(x))
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        dropout = config.get("dropout",0.0)
        self.config = dict(config) #什么东西
        self.context = config['context']  #不能改
        width = config['width'] #为什么这里不是self.width?
        self.token = nn.Embedding(config['vocab'],width) #这里是对输入的词表做embedding
        attn_width = config.get("attn_width", width)
        #换成rope
        #self.pos = nn.Embedding(self.context, width)  
        #self.blocks = nn.ModuleList([Block(width, config['heads']) for _ in range(config['depth'])]) #这里定义这个模型有多少层，重复多少次刚刚定义的前馈模型
        blocks = [Block(width, config['heads'],dropout) for _ in range(config['depth'] - 1) ]
        blocks.append(Block1(width, config['heads'],dropout))
        blocks.append(Block2(width, config['heads'],dropout))
        #blocks.append(Block3(width, config['heads'], attn_width))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'],bias=False) #这里是想做什么？
        self.apply(self.initialize) #对这个模型里的矩阵进行初始化
        self.head.weight = self.token.weight #共享权重

    @staticmethod #啥意思？
    def initialize(module): #这一块都是初始化，现在不懂也没有非常大的影响
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        #这里对x加入位置信息
        #换成rope
        #x = self.token(ids) + self.pos(torch.arange(ids.shape[1],device=ids.device))
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)  #让x经过前向
        return self.norm(x) #让结果归一化

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(),dim=-1) #算概率

def build_model(config):
    return GPT(config)
    


