from .base_models import *
from .layers import activation_layer
import numpy as np
from torch import nn
import torch
import math
# from collections import Counter, OrderedDict
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
from torch.nn.functional import log_softmax
# from torch.optim.lr_scheduler import LambdaLR
# from functools import partial
# from torch.nn.utils.rnn import pad_sequence


class Embedder(nn.Module):
    def __init__(self, n_input, d_model):
        super().__init__()
        self.embed = nn.Linear(n_input, d_model)
        self.d_model = d_model
        self.embed_scale = math.sqrt(self.d_model)

    def forward(self, x):
        """
        :param x: tokenlized sequence
        :return:
        """
        # 乘以一个较大的系数，放大词嵌入向量，
        # 希望与位置编码向量相加后，词嵌入向量本身的影响更大
        return self.embed(x) * self.embed_scale

# 预先计算好所有可能的位置编码，然后直接查表就能得到
# 注意维度顺序
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X的形状为 (batch_size, seq_length, d_model)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, head_number, d_model, dropout=0.1):
        """
        :param head_number: 自注意力头的数量
        :param d_model: 隐藏层的维度
        """
        super().__init__()
        self.h = head_number
        self.d_model = d_model
        self.dk = d_model // head_number
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)

    def head_split(self, tensor, batch_size):
        # 将(batch_size, seq_len, d_model) reshape成 (batch_size, seq_len, h, d_model//h)
        # 然后再转置第1和第2个维度，变成(batch_size, h, seq_len, d_model/h)
        return tensor.view(batch_size, -1, self.h, self.dk).transpose(1, 2)

    def head_concat(self, similarity, batch_szie):
        # 恢复计算注意力之前的形状
        return similarity.transpose(1, 2).contiguous() \
            .view(batch_szie, -1, self.d_model)

    def cal_attention(self, q, k, v, mask=None):
        """
        论文中的公式 Attention(K,Q,V) = softmax(Q@(K^T)/dk**0.5)@V
        ^T 表示矩阵转置
        @ 表示矩阵乘法
        """
        similarity = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            mask = mask.unsqueeze(1)
            # 将mask为0的位置填充为绝对值非常大的负数
            # 这样经过softmax后，其对应的权重就会非常接近0, 从而起到掩码的效果
            similarity = similarity.masked_fill(mask == 0, -1e9)
        similarity = self.softmax(similarity)
        similarity = self.dropout(similarity)

        output = torch.matmul(similarity, v)
        return output


    def forward(self, q, k, v, mask=None):
        """
        q,k,v即自注意力公式中的Q,K,V，mask表示掩码
        """
        batch_size, seq_length, d = q.size()
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        # 分成多个头
        q = self.head_split(q, batch_size)
        k = self.head_split(k, batch_size)
        v = self.head_split(v, batch_size)
        similarity = self.cal_attention(q, k, v, mask)
        # 合并多个头的结果
        similarity = self.head_concat(similarity, batch_size)

        # 再使用一个线性层， 投影一次
        output = self.output(similarity)
        return output

# 论文中的前馈神经网络。实际上就是一个两层的全连接，中间加上个relu和dropout
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff, dropout=None):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dff, d_model)
        if dropout is not None:
            self.dropout = nn.Dropout(0.1)
        else:
            self.dropout = None

    def forward(self, x):
        """
        :param x: 来自多头自注意力层的输出
        :return:
        """
        x = self.fc1(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        output = self.fc2(x)
        return output


# 编码器层。
# 每个编码器层由两个sublayer组成，即一个多头注意力层和一个前馈网络
class EncoderLayer(nn.Module):
    def __init__(self, head_number, d_model, d_ff, dropout=0.1, res_connect=True):
        super().__init__()

        # mha
        self.mha = MultiHeadAttention(head_number, d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # mlp
        self.mlp = FeedForwardNetwork(d_model, d_ff, dropout=None)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.res_connect = res_connect

    def forward(self, x, mask=None):
        x2 = self.norm1(x)

        y = self.dropout1(self.mha(x2, x2, x2, mask))
        # 注意残差连接是和norm之前的输入相加，norm之后的不在一个数量级
        if self.res_connect:
            y = y + x

        y2 = self.norm2(y)
        y2 = self.dropout2(self.mlp(y2))
        if self.res_connect:
            y2 = y + y2


        return y2

class Encoder(nn.Module):
    def __init__(self, stack=6, multi_head=8, d_model=512, d_ff=2048, dropout=0.1):
        """
        :param stack: 堆叠多少个编码器层
        :param multi_head: 多头注意力头的数量
        :param d_model: 隐藏层的维度
        """
        super().__init__()
        self.encoder_stack = []
        for i in range(stack):
            encoder_layer = EncoderLayer(multi_head, d_model, d_ff, dropout=dropout)
            self.encoder_stack.append(encoder_layer)
        self.encoder = nn.ModuleList(self.encoder_stack)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for encoder_layer in self.encoder:
            x = encoder_layer(x, mask)
        x = self.norm(x)
        return x


# 解码器层
# 一个解码器层由三个sublayer组成，即一个masked自注意力层，一个cross自注意力层和一个前馈网络层
# cross自注意力层意思是q,k,v分别来自编码器和解码器
class DecoderLayer(nn.Module):
    def __init__(self, head_number, d_model, d_ff, dropout=0.1):
        super().__init__()
        # shifted right self attention layer
        self.mha1 = MultiHeadAttention(head_number, d_model)

        # cross attention
        self.mha2 = MultiHeadAttention(head_number, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.mlp = FeedForwardNetwork(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, q, k, v, src_mask, tgt_mask):
        # 注意第一个注意力层的qkv都是同一个
        x2 = self.norm1(q)
        y = self.mha1(x2, x2, x2, tgt_mask)
        y = self.dropout1(y)

        # 注意残差连接是和norm之前的输入相加，norm之后的不在一个数量级
        y = y + q

        y2 = self.norm2(y)

        # 第二个自注意力层的k和v是encoder的输出
        y2 = self.mha2(y2, k, v, src_mask)
        y2 = self.dropout2(y2)
        y2 = y + y2

        y3 = self.norm3(y2)
        y3 = self.mlp(y3)
        y3 = self.dropout3(y3)
        y3 = y2 + y3

        return y3

# 解码器
# 解码器就是N个解码器层堆叠起来
class Decoder(nn.Module):
    def __init__(self, stack=6, head_number=8, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.decoder_stack = []
        for i in range(stack):
            self.decoder_stack.append(DecoderLayer(head_number, d_model, d_ff, dropout=dropout))
        self.decoder_stack = nn.ModuleList(self.decoder_stack)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, output_from_encoder, src_mask, tgt_mask):
        for decoder_layer in self.decoder_stack:
            x = decoder_layer(x, output_from_encoder, output_from_encoder, src_mask, tgt_mask)
        x = self.norm(x)
        return x

class EasyTransformer(BaseModel):
    def __init__(self,
                 src_voc_size,
                 target_voc_size,
                 stack_number=6,
                 d_model=512,
                 h=8,
                 d_ff=2048):
        super().__init__()
        self.input_embedding_layer = Embedder(src_voc_size, d_model)
        self.input_pe = PositionalEncoding(d_model)

        self.output_embedding_layer = Embedder(target_voc_size, d_model)
        self.output_pe = PositionalEncoding(d_model)

        self.encoder = Encoder(stack_number, h, d_model, d_ff)
        self.decoder = Decoder(stack_number, h, d_model, d_ff)
        self.final_output = nn.Linear(d_model, target_voc_size)

    def encode(self, src, src_mask):
        x = self.input_embedding_layer(src)
        x = self.input_pe(x)
        output_from_encoder = self.encoder(x, src_mask)
        return output_from_encoder

    def decode(self, output_from_encoder, shifted_right, src_mask, tgt_mask):
        shifted_right = self.output_embedding_layer(shifted_right)
        shifted_right = self.output_pe(shifted_right)

        decoder_output = self.decoder(shifted_right, output_from_encoder, src_mask, tgt_mask)

        output = self.final_output(decoder_output)
        return output

    def forward(self, x, shifted_right, src_mask, tgt_mask):
        x = self.input_embedding_layer(x)
        x = self.input_pe(x)

        output_from_encoder = self.encoder(x, src_mask)

        shifted_right = self.output_embedding_layer(shifted_right)
        shifted_right = self.output_pe(shifted_right)

        decoder_output = self.decoder(shifted_right, output_from_encoder, src_mask, tgt_mask)

        output = log_softmax(self.final_output(decoder_output), dim=-1)

        return output


class EasyTransformer(BaseModel):
    def __init__(self, n_input, n_hiddens, n_output, act_func='relu', dropout=None):
        """
        DNN network. The input size should be [batch, n_input]
        Args:
            n_input: dim of input
            n_hiddens: dim of hidden layers. e.g. [64, 128, 64]
            n_output: dim of output
            act_func: 'relu' | 'tanh' | 'LeakyReLU' | 'sigmoid'
            dropout: dropout rate, default:None
        """
        super(EasyTransformer, self).__init__()
        net = nn.Transformer()

        channel_list = [n_input] + n_hiddens
        layers = []
        for i in range(len(channel_list)-1):
            mini_layer = [nn.Linear(in_features=channel_list[i], out_features=channel_list[i+1]),
                       activation_layer(act_name=act_func)]
            if dropout is not None:
                mini_layer += [nn.Dropout(dropout)]
            layers += mini_layer

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features=channel_list[-1], out_features=n_output)