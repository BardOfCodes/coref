import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .flash_attention import FlashAttention
from .features_extractor_3d import Vox3DCNN
from .features_extractor_2d import Vox2DCNN
from einops import rearrange


class BaseVPINet(nn.Module):

    def __init__(self, config):
        super(BaseVPINet, self).__init__()

        self.set_config(config)
        self.set_feature_extractor(config)

        self.pos_encoding = LearnablePositionalEncoding(
            self.hidden_dim, dropout=self.dropout, max_len=self.max_length)
        self.token_embedding = nn.Embedding(
            self.prog_token_count, self.hidden_dim)
        if self.flash_attention:
            self.attn_layers = nn.ModuleList([FlashAttnLayer(
                self.num_heads, self.hidden_dim, self.head_dim, self.hidden_dim, dropout=self.dropout) for _ in range(self.num_dec_layers)])
        else:
            if self.old_attn_mode:
                self.attn_layers = nn.ModuleList([OldAttnLayer(
                    self.num_heads, self.hidden_dim, self.hidden_dim, dropout=self.dropout) for _ in range(self.num_dec_layers)])
                self.generate_attn_mask = self.generate_old_attn_mask
            else:
                self.attn_layers = nn.ModuleList([AttnLayer(
                    self.num_heads, self.hidden_dim, self.hidden_dim, dropout=self.dropout) for _ in range(self.num_dec_layers)])
                self.generate_attn_mask = self.generate_new_attn_mask
        self.attn_to_output = SEQ_MLP(self.post_attn_sizes, dropout_rate=0.0)

        self.cmd_logsmax = nn.LogSoftmax(dim=1)

        attn_mask = self.generate_attn_mask()
        self.register_buffer("attn_mask", attn_mask)

        self.apply(self.initialize_weights)
        # Compiled functions
        # self.unrolled_inference_forward = th.compile(self._unrolled_inference_forward)
        self.unrolled_inference_forward = self._unrolled_inference_forward

    def set_feature_extractor(self, config):
        if config.n_dims == 3:
            cnn_class = Vox3DCNN
        elif config.n_dims == 2:
            cnn_class = Vox2DCNN
        self.cnn_extractor = cnn_class(self.hidden_dim, dropout=self.dropout,
                                       first_stride=self.cnn_first_stride,
                                       out_len=self.visual_seq_len)

    def set_config(self, config):

        # Parameters:
        self.cnn_first_stride = config.cnn_first_stride
        self.visual_seq_len = config.visual_seq_len  # 8
        self.prog_seq_len = config.prog_seq_len  # 128  + 1# seq_len
        self.max_length = self.visual_seq_len + self.prog_seq_len  # start token
        self.num_dec_layers = config.num_dec_layers  # 8 # num_layers
        self.num_heads = config.num_heads  # 16# num_heads
        self.head_dim = config.head_dim  # 64 # dim_head
        # 75 # len(ex.TOKENS) + start symbol
        self.prog_token_count = config.prog_token_count
        self.hidden_dim = config.hidden_dim  # 128 # hidden_dim
        self.post_attn_sizes = config.post_attn_sizes
        self.device = config.device
        self.dropout = config.dropout
        self.infer_start_pos = config.visual_seq_len
        self.flash_attention = config.use_flash_attention
        self.old_attn_mode = config.old_attn_mode

    def initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight.data, 0, 0.02)
        # for attention layers
        elif isinstance(m, nn.MultiheadAttention):
            std = (m.embed_dim**-0.5)
            nn.init.normal_(m.in_proj_weight.data, std=std)
            nn.init.normal_(m.out_proj.weight.data, std=std)
            if m.in_proj_bias is not None:
                nn.init.constant_(m.in_proj_bias.data, 0)
            if m.out_proj.bias is not None:
                nn.init.constant_(m.out_proj.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def generate_old_attn_mask(self):
        sz = self.max_length
        mask = (th.triu(th.ones(sz, sz)) == 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0)).T
        mask[:self.infer_start_pos, :self.infer_start_pos] = 0.
        return mask

    def generate_new_attn_mask(self):
        sz = self.max_length
        mask = (th.triu(th.ones(sz, sz)) == 1)
        mask[:self.infer_start_pos, :self.infer_start_pos] = False
        return mask

    def forward_train(self, input_batch):

        x_in = input_batch["occs"]
        actions_in = input_batch["actions"]

        cnn_features = self.cnn_extractor.forward(x_in)
        token_embeddings = self.token_embedding(actions_in)

        out = self.pos_encoding(
            th.cat((cnn_features, token_embeddings), dim=1))

        for attn_layer in self.attn_layers:
            out = attn_layer(out, self.attn_mask, None)
        # should be self.prog_seq_len
        seq_out = out[:, self.visual_seq_len:-1, :]

        output = self.stack_all_vectors(seq_out)

        cmd_distr = self.attn_to_output(output)
        cmd_logsoft = self.cmd_logsmax(cmd_distr)
        # cmd_distr = th.softmax(cmd_distr, dim = 1)
        return cmd_logsoft

    def forward_beam_init(self, input_batch, beam_size):

        input_occ = input_batch["occs"]
        batch_size = input_occ.shape[0]
        start_token = self.token_embedding.num_embeddings - 2

        cnn_features = self.cnn_extractor.forward(input_occ)
        token_seq = th.zeros(batch_size, self.prog_seq_len,
                             device=self.device).long()
        token_seq[:, 0] = start_token

        embedding_seq = th.zeros(batch_size, self.max_length,
                                 self.token_embedding.embedding_dim,
                                 device=self.device)

        token_embedding = self.token_embedding(token_seq) * 0
        features = th.cat((cnn_features, token_embedding), dim=1)
        embedding_seq[:, :] = self.pos_encoding(features)

        embedding_seq = embedding_seq.unsqueeze(
            1).expand(-1, beam_size, -1, -1)

        return embedding_seq

    def _unrolled_inference_forward(self, out, index):

        attn_mask = self.attn_mask[:self.infer_start_pos +
                                   index + 1, :self.infer_start_pos + index + 1]
        for attn_layer in self.attn_layers:
            out = attn_layer(out, attn_mask, None)
        # should be self.prog_seq_len
        seq_out = out[:, -1, :]
        cmd_output = self.attn_to_output(seq_out)
        return cmd_output

    def stack_all_vectors(self, vectors):
        output = vectors.reshape(-1, vectors.shape[-1])
        return output


class SEQ_MLP(nn.Module):
    def __init__(self, size_seq, dropout_rate):
        super(SEQ_MLP, self).__init__()
        layers = []
        n_layers = len(size_seq) - 1
        for i in range(n_layers):
            layer = nn.Linear(size_seq[i], size_seq[i+1])
            layers.append(layer)
            if i != (n_layers - 1):
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        output = self.layers(x)
        return output


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=256):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)
        pos_arange = th.arange(max_len).unsqueeze(0)
        self.register_buffer("pos_arange", pos_arange)

    def forward(self, x):
        pe = self.pe(self.pos_arange.repeat(x.shape[0], 1))
        x = x + pe[:, :x.size(1)]
        x = self.dropout(x)
        return x

    def get_singular_position(self, x, position):
        position = th.tensor(position).to(x.device)
        pe = self.pe(position)
        x = x + pe
        return self.dropout(x)


class AttnLayer(nn.Module):

    def __init__(self, num_head, inp_dim, dim_feedforward, dropout):
        super(AttnLayer, self).__init__()
        self.num_heads = num_head
        self.inp_dim = inp_dim
        self.dim_feedforward = dim_feedforward

        self.self_attn = nn.MultiheadAttention(self.inp_dim, self.num_heads, dropout=dropout,
                                               batch_first=True)
        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(self.inp_dim, self.dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.dim_feedforward, self.inp_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(self.inp_dim)
        self.norm2 = nn.LayerNorm(self.inp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, key_padding_mask):

        attn_out = self.self_attn(x, x, x, attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=False)[0]
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x


class OldAttnLayer(nn.Module):
    def __init__(self, nh, hd, hd_2, dropout):
        super(OldAttnLayer, self).__init__()
        self.num_heads = nh
        self.hidden_dim = hd

        self.self_attn = nn.MultiheadAttention(self.hidden_dim, self.num_heads)

        self.l1 = nn.Linear(hd, hd)
        self.l2 = nn.Linear(hd, hd)

        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout)

        self.n1 = nn.LayerNorm(hd)
        self.n2 = nn.LayerNorm(hd)

    def forward(self, src, attn_mask, key_padding_mask):

        src = src.transpose(0, 1)

        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]

        src = src + self.d1(src2)
        src = self.n1(src)
        src2 = self.l2(self.d2(F.leaky_relu(self.l1(self.n2(src)), .2)))
        src = src + self.d2(src2)
        src = self.n2(src)
        return src.transpose(0, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(th.ones(dim))
        self.register_buffer("beta", th.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class FlashAttnLayer(nn.Module):

    def __init__(self,
                 num_head=8,
                 inp_dim=256,  # 256
                 dim_head=64,
                 dim_feedforward=256,
                 ff_mult=2,
                 dropout=0.1):
        super(FlashAttnLayer, self).__init__()

        self.num_heads = num_head
        self.inp_dim = inp_dim
        self.dim_feedforward = dim_feedforward

        self.norm = LayerNorm(inp_dim)

        attn_inner_dim = dim_head * num_head
        ff_inner_dim = dim_head * ff_mult
        self.fused_dims = (attn_inner_dim, attn_inner_dim,
                           attn_inner_dim, 2 * ff_inner_dim)

        self.attend = FlashAttention(dropout=dropout)

        self.fused_attn_ff_proj = nn.Linear(
            inp_dim, sum(self.fused_dims), bias=False)

        self.attn_out = nn.Linear(attn_inner_dim, inp_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # parallel feedforward tail

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_inner_dim, inp_dim, bias=False)
        )

    def forward(self, x, attn_mask, key_padding_mask):

        n, device, h = x.shape[1], x.device, self.num_heads

        # pre layernorm
        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        # attention function, either regular or flash
        out = self.attend(q, k, v, mask=attn_mask)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        attn_out = self.attn_out(out)
        ff_out = self.ff_out(ff)

        return attn_out + ff_out
