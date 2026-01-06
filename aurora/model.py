import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    def __init__(self, in_features, num_frequencies=6, d_out=64):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.d_out = d_out
        self.linear = None

    def forward(self, x):
        embeds = [x]
        for i in range(1, self.num_frequencies+1):
            embeds.append(torch.sin((2.0**i) * x))
            embeds.append(torch.cos((2.0**i) * x))
        feat = torch.cat(embeds, dim=-1)

        if self.linear is None:
            self.linear = nn.Linear(feat.shape[-1], self.d_out, bias=True).to(feat.device)
        return self.linear(feat)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.src_embedding = FourierEmbedding(input_dim, d_out=d_model)
        self.tgt_embedding = FourierEmbedding(output_dim, d_out=d_model)

        self.pos_encoder = nn.Parameter(torch.randn(100, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.out = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        src = src + self.pos_encoder[:src.size(1)].unsqueeze(0)
        tgt = tgt + self.pos_encoder[:tgt.size(1)].unsqueeze(0)

        memory = self.encoder(src)
        output = self.decoder(tgt, memory)

        return self.out(output)


class TransformerEncoderOnly(nn.Module):
    def __init__(self, input_dim=5, output_dim=3, d_model=128,
                 nhead=4, num_layers=2, dropout=0.1, out_steps=2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, int(d_model/2)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(int(d_model/2), out_steps * output_dim)
        )
        self.out_steps = out_steps
        self.output_dim = output_dim

    def forward(self, x):
        x = self.input_proj(x)           # (B, T_in, d_model)
        memory = self.encoder(x)         # (B, T_in, d_model)
        pooled = memory.mean(dim=1)      # 평균 pooling (global context)
        out = self.fc_out(pooled)        # (B, out_steps * output_dim)
        return out.view(-1, self.out_steps, self.output_dim)