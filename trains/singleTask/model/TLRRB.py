"""
here is the mian backbone for TLRRB
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from einops import rearrange

class TLRRB(nn.Module):
    def __init__(self, args):
        super(TLRRB, self).__init__()
        self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)

        self.text_linear = nn.Linear(args.feature_dims[0], 128)
        self.video_linear = nn.Linear(args.feature_dims[2], 128)
        self.audio_linear = nn.Linear(args.feature_dims[1], 128)

        self.conv_text = nn.Conv1d(
            in_channels=768,
            out_channels=128,
            kernel_size=1,
            padding=0
        )
        self.conv_video = nn.Conv1d(
            in_channels=args.feature_dims[2],
            out_channels=128,
            kernel_size=1,
            padding=0
        )
        self.conv_audio = nn.Conv1d(
            in_channels=args.feature_dims[1],
            out_channels=128,
            kernel_size=1,
            padding=0
        )
        self.transformer_v = TransformerEncoder(embed_dim=128,
                                  num_heads=8,
                                  layers=2,
                                  attn_dropout=0,
                                  relu_dropout=0,
                                  res_dropout=0,
                                  embed_dropout=0,
                                  attn_mask=True)

        self.transformer_a = TransformerEncoder(embed_dim=128,
                                  num_heads=8,
                                  layers=2,
                                  attn_dropout=0,
                                  relu_dropout=0,
                                  res_dropout=0,
                                  embed_dropout=0,
                                  attn_mask=True)

        self.LayerNorm_video = nn.LayerNorm(128, eps=1e-12)
        self.LayerNorm_audio = nn.LayerNorm(128, eps=1e-12)
        self.LayerNorm_text = nn.LayerNorm(128, eps=1e-12)

        self.LayerNorm_video_trans = nn.LayerNorm(128, eps=1e-12)
        self.LayerNorm_audio_trans = nn.LayerNorm(128, eps=1e-12)

        self.attn_v = MultiHeadAttention(128,8, 0.1)
        self.attn_a = MultiHeadAttention(128,8, 0.1)

        self.Layernorm_cross = nn.LayerNorm(128, eps=1e-12)

        self.text_table = build_table(128)

        self.LayerNorm_video_MMD = nn.LayerNorm(128, eps=1e-12)
        self.LayerNorm_audio_MMD = nn.LayerNorm(128, eps=1e-12)
        self.LayerNorm_text_MMD = nn.LayerNorm(128, eps=1e-12)

        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear_final = nn.Linear(256, 1)

        self.CL = CL_sentiment()

        self.MMD_v = MMD_loss(128)
        self.MMD_a = MMD_loss(128)

        self.activate = nn.Tanh()

    def forward(self, text_bert, audio, video, labels_CL):
        size_of_batch = text_bert.size(0)
        t_len = text_bert.size(2)

        text_mask = text_bert[:, 1, :]

        video_mask = (video.abs().sum(dim=-1) > 0).long()
        video_mask[:, 0] = 1

        audio_mask = (audio.abs().sum(dim=-1) > 0).long()
        audio_mask[:, 0] = 1

        text_cl = self.text_model(text_bert)

        text = F.dropout(text_cl, p=0.1)

        text = rearrange(text, 'b m d-> b d m')
        text = self.conv_text(text)
        text = rearrange(text, 'b d m -> b m d')

        video = rearrange(video, 'b m d-> b d m')
        video = self.conv_video(video)
        video = rearrange(video, 'b d m -> b m d')

        audio = rearrange(audio, 'b m d-> b d m')
        audio = self.conv_audio(audio)
        audio = rearrange(audio, 'b d m -> b m d')

        video_trans = self.transformer_v(video)
        audio_trans = self.transformer_a(audio)

        video_att = F.dropout(video_trans, p=0.1)
        audio_att = F.dropout(audio_trans, p=0.1)

        text_v, _ = self.attn_v(text, video_att, video_att)
        text_a, _ = self.attn_a(text, audio_att, audio_att)

        text_fusion = self.Layernorm_cross(text + text_v + text_a)

        text_table = self.text_table(text_fusion, t_len)

        text_mean = text_table.mean(dim=(1, 2))

        text_mmd = text
        video_mmd = self.LayerNorm_video_MMD(video_trans)
        audio_mmd = self.LayerNorm_audio_MMD(audio_trans)
        #
        loss_v = 0.001 * self.MMD_v(text_mmd, video_mmd, text_mask, video_mask)
        loss_a = 0.001 * self.MMD_a(text_mmd, audio_mmd, text_mask, audio_mask)
        #
        loss_MMD = loss_v + loss_a

        labels_CL = torch.round(labels_CL)
        feature_t_cl = torch.unsqueeze(text_cl.reshape(text.size(0), -1), dim=1)
        loss_CL = 0.05 * self.CL(feature_t_cl, labels_CL)

        output_feature = text_mean
        output = self.linear2(F.dropout(F.relu(self.linear1(output_feature), inplace=True), p=0.5))
        output = output + output_feature

        output = self.linear_final(output)

        res = {
            'output_logit': output,
            'loss_CL': loss_CL,
            'loss_MMD': loss_MMD,
        }
        return res

class build_table(torch.nn.Module):
    def __init__(self, hidden_dim):
        torch.nn.Module.__init__(self)
        self.conv_0 = nn.Conv2d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim * 2,
            kernel_size=(1, 1),
            padding=0
        )
        self.conv_1 = nn.Conv2d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim * 2,
            kernel_size=(3, 3),
            padding=1
        )
        self.conv_2 = nn.Conv2d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim * 2,
            kernel_size=(1, 1),
            padding=0
        )
        self.conv_3 = nn.Conv2d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim * 2,
            kernel_size=(3, 3),
            padding=1
        )
        self.norm0 = nn.LayerNorm(hidden_dim * 2, eps=1e-12)
        self.norm1 = nn.LayerNorm(hidden_dim * 2, eps=1e-12)
        self.norm2 = nn.LayerNorm(hidden_dim * 2, eps=1e-12)
        self.norm3 = nn.LayerNorm(hidden_dim * 2, eps=1e-12)
        self.norm4 = nn.LayerNorm(hidden_dim * 2, eps=1e-12)

    def forward(self, seq, length):
        seq_table = seq.unsqueeze(2).expand([-1, -1, length, -1])
        seq_table_T = seq_table.transpose(1, 2)
        table = torch.cat((seq_table, seq_table_T), dim=3)
        table0 = rearrange(table, 'b m n d -> b d m n')
        n = table0.size(-1)

        D1_0 = self.conv_1(table0)
        D1_0 = rearrange(D1_0, 'b d m n-> b (m n) d ', n=n)
        D1_0 = self.norm0(D1_0)
        D1_0 = torch.relu(D1_0)
        D1_0 = rearrange(D1_0, 'b (m n) d -> b d m n', n=n)

        D1_1 = self.conv_2(D1_0)
        D1_1 = rearrange(D1_1, 'b d m n-> b (m n) d ', n=n)
        D1_1 = self.norm1(D1_1)
        D1_1 = torch.relu(D1_1)
        D1_1 = rearrange(D1_1, 'b (m n) d -> b m n d', n=n)

        return D1_1

class CL_sentiment(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all',
                 temperature=0.07):
        super(CL_sentiment, self).__init__()
        self.temperature = 0.07
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = F.normalize(features, dim=2)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask)-mask) * logits_mask
        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)

        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        return loss

class MMD_loss(nn.Module):
    def __init__(self, hidden_dim):
        super(MMD_loss, self).__init__()
        self.fix_sigma = None
        self.hidden_dim = hidden_dim
        return

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma=None):

        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

        L2_distance_square = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance_square) / (n_samples ** 2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        # print(bandwidth_list)

        kernel_val = [torch.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)  # /len(kernel_val)

    def forward(self, source, target, source_mask, target_mask, kernel_mul=4, kernel_num=8, fix_sigma=None):

        source = source.reshape(-1, self.hidden_dim)
        source_mask = source_mask.reshape(-1)
        # print("source ", source.shape)
        # print("source ", source_mask.shape)
        source = source[source_mask.bool()]
        # print("source", source.shape)

        target = target.view(size=(-1, self.hidden_dim))
        target_mask = target_mask.reshape(-1)
        # print("target ", target.shape)
        # print("target ", target_mask.shape)
        target = target[target_mask.bool()]
        # print("target", target.shape)

        source_num = int(source.size()[0])
        target_num = int(target.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                       fix_sigma=fix_sigma)
        XX = torch.mean(kernels[:source_num, :source_num])
        YY = torch.mean(kernels[source_num:, source_num:])
        XY = torch.mean(kernels[:source_num, source_num:])
        YX = torch.mean(kernels[source_num:, :source_num])
        loss = XX + YY - XY - YX
        return loss

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, drop):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.drop = torch.nn.Dropout(drop)

    def forward(self, q, k, v, key_padding_mask=None):
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        Q = self.q_proj(q).view(B, Lq, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(k).view(B, Lk, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(v).view(B, Lk, self.num_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Lq, -1)

        return self.drop(self.out_proj(out)), attn