from __future__ import annotations

import math

import torch
from torch import nn


class ConvFeatureExtractor(nn.Module):
    """Small 1D CNN front-end for local EEG waveform/rhythm features."""

    def __init__(self, in_channels: int = 2, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),
            nn.Conv1d(64, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),
        )

    def forward(self, x_channels_first: torch.Tensor) -> torch.Tensor:
        # Input: (batch, channels, time). Output: (batch, features, reduced_time).
        return self.net(x_channels_first)


class LSTMOnly(nn.Module):
    """Ablation model: recurrent temporal modeling without CNN local features."""

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 2,
        d_model: int = 96,
        lstm_hidden: int = 48,
        use_context: bool = False,
    ):
        super().__init__()
        self.raw_pool = nn.AvgPool1d(kernel_size=30, stride=30)
        self.lstm_input = nn.Linear(in_channels, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        fused_dim = 2 * lstm_hidden
        self.context_lstm = (
            nn.LSTM(
                input_size=fused_dim,
                hidden_size=fused_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            if use_context
            else None
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def encode_epoch(self, x: torch.Tensor) -> torch.Tensor:
        x_cf = x.transpose(1, 2)
        lstm_seq = self.raw_pool(x_cf).transpose(1, 2)
        lstm_seq = self.lstm_input(lstm_seq)
        _, (h_n, _) = self.lstm(lstm_seq)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if self.context_lstm is None:
                raise ValueError("LSTMOnly was created without context support")
            batch, context, time_steps, channels = x.shape
            epoch_vec = self.encode_epoch(x.reshape(batch * context, time_steps, channels))
            epoch_vec = epoch_vec.reshape(batch, context, -1)
            _, (h_n, _) = self.context_lstm(epoch_vec)
            fused = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            fused = self.encode_epoch(x)
        return self.classifier(fused)


class CNNLSTM(nn.Module):
    """
    CNN-LSTM baseline close to the cited paper's feature-fusion idea.

    The paper describes CNN and LSTM extracting features in parallel, followed by
    feature fusion for sleep-stage classification. This implementation keeps that
    structure while using a compact architecture suitable for the course dataset:
      - CNN branch learns local EEG waveform features.
      - LSTM branch sees 100 short raw-signal patches from the 30s epoch.
      - The two representations are concatenated before classification.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 2,
        d_model: int = 96,
        lstm_hidden: int = 48,
        use_context: bool = False,
    ):
        super().__init__()
        self.cnn = ConvFeatureExtractor(in_channels=in_channels, out_channels=128)
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        self.cnn_proj = nn.Linear(128, d_model)

        self.raw_pool = nn.AvgPool1d(kernel_size=30, stride=30)
        self.lstm_input = nn.Linear(in_channels, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        fused_dim = d_model + 2 * lstm_hidden
        self.context_lstm = (
            nn.LSTM(
                input_size=fused_dim,
                hidden_size=fused_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            if use_context
            else None
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def encode_epoch(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time=3000, channels=2)
        x_cf = x.transpose(1, 2)

        cnn_features = self.cnn(x_cf)
        cnn_vec = self.cnn_pool(cnn_features).squeeze(-1)
        cnn_vec = self.cnn_proj(cnn_vec)

        # Downsample raw signal into 100 time steps for the recurrent branch.
        lstm_seq = self.raw_pool(x_cf).transpose(1, 2)
        lstm_seq = self.lstm_input(lstm_seq)
        _, (h_n, _) = self.lstm(lstm_seq)
        # Last layer, forward and backward hidden states.
        lstm_vec = torch.cat([h_n[-2], h_n[-1]], dim=1)

        fused = torch.cat([cnn_vec, lstm_vec], dim=1)
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if self.context_lstm is None:
                raise ValueError("CNNLSTM was created without context support")
            batch, context, time_steps, channels = x.shape
            epoch_vec = self.encode_epoch(x.reshape(batch * context, time_steps, channels))
            epoch_vec = epoch_vec.reshape(batch, context, -1)
            _, (h_n, _) = self.context_lstm(epoch_vec)
            fused = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            fused = self.encode_epoch(x)
        return self.classifier(fused)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PureTransformer(nn.Module):
    """
    Ablation model: Transformer over raw EEG patches, without a CNN front-end.

    This intentionally removes convolutional local feature extraction so that the
    experiment can test whether CNN features are important before attention.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 2,
        d_model: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        patch_size: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_size * in_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=1024)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            batch, context, time_steps, channels = x.shape
            x = x.reshape(batch, context * time_steps, channels)
        batch, time_steps, channels = x.shape
        usable = (time_steps // self.patch_size) * self.patch_size
        x = x[:, :usable, :]
        x = x.reshape(batch, usable // self.patch_size, self.patch_size * channels)
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos(tokens)
        encoded = self.encoder(tokens)
        return self.classifier(encoded[:, 0])


class CNNTransformer(nn.Module):
    """
    Proposed model: CNN front-end + Transformer encoder.

    The CNN learns local EEG features and downsamples the 3000-step signal before
    self-attention models longer-range relationships over higher-level tokens.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 2,
        d_model: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = ConvFeatureExtractor(in_channels=in_channels, out_channels=128)
        self.proj = nn.Conv1d(128, d_model, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x_cf = x.transpose(1, 2)
        features = self.cnn(x_cf)
        return self.proj(features).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            batch, context, time_steps, channels = x.shape
            tokens = self.encode_tokens(x.reshape(batch * context, time_steps, channels))
            tokens = tokens.mean(dim=1).reshape(batch, context, tokens.size(2))
        else:
            tokens = self.encode_tokens(x)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos(tokens)
        encoded = self.encoder(tokens)
        return self.classifier(encoded[:, 0])


def build_model(
    model_name: str,
    context_size: int = 1,
    cnn_transformer_layers: int = 2,
    in_channels: int = 2,
) -> nn.Module:
    use_context = context_size > 1
    if model_name == "lstm_only":
        return LSTMOnly(in_channels=in_channels, use_context=use_context)
    if model_name == "cnn_lstm":
        return CNNLSTM(in_channels=in_channels, use_context=use_context)
    if model_name == "pure_transformer":
        return PureTransformer(in_channels=in_channels)
    if model_name == "cnn_transformer":
        return CNNTransformer(in_channels=in_channels, num_layers=cnn_transformer_layers)
    raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
