# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:47:55 2023
@author: zhangxin
"""

import random
from .modules.seanet import SEANetEncoder, SEANetDecoder
from .quantization import ResidualVectorQuantizer
import torch.nn as nn
from einops import rearrange
import torch
import numpy as np
from functools import reduce

class SpeechTokenizer(nn.Module):
    def __init__(self, config):
        """

        Parameters
        ----------
        config : json
            Model Config.

        """
        super().__init__()
        self.config = config
        self.encoder = SEANetEncoder(
            n_filters=config.get("n_filters"),
            dimension=config.get("dimension"),
            ratios=config.get("strides"),
            lstm=config.get("lstm_layers"),
            bidirectional=config.get("bidirectional"),
            dilation_base=config.get("dilation_base"),
            residual_kernel_size=config.get("residual_kernel_size"),
            n_residual_layers=config.get("n_residual_layers"),
            activation=config.get("activation"),
        )
        self.sample_rate = config.get("sample_rate")
        self.n_q = config.get("n_q")
        self.downsample_rate = np.prod(config.get("strides"))
        if config.get("dimension") != config.get("semantic_dimension"):
            self.transform = nn.Linear(
                config.get("dimension"), config.get("semantic_dimension")
            )
        else:
            self.transform = nn.Identity()
        self.quantizer = ResidualVectorQuantizer(
            dimension=config.get("dimension"),
            n_q=config.get("n_q"),
            bins=config.get("codebook_size"),
        )
        self.decoder = SEANetDecoder(
            n_filters=config.get("n_filters"),
            dimension=config.get("dimension"),
            ratios=config.get("strides"),
            lstm=config.get("lstm_layers"),
            bidirectional=False,
            dilation_base=config.get("dilation_base"),
            residual_kernel_size=config.get("residual_kernel_size"),
            n_residual_layers=config.get("n_residual_layers"),
            activation=config.get("activation"),
        )

    @classmethod
    def load_from_checkpoint(cls, config_path: str, ckpt_path: str):
        """

        Parameters
        ----------
        config_path : str
            Path of model configuration file.
        ckpt_path : str
            Path of model  checkpoint.

        Returns
        -------
        model : SpeechTokenizer
            SpeechTokenizer model.

        """
        import json

        with open(config_path) as f:
            cfg = json.load(f)
        model = cls(cfg)
        params = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(params)
        return model

    def forward(
        self,
        x: torch.tensor,
    ):
        """

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.
        layers : list[int], optional
            Layers of RVQ should return quantized result. The default is the first layer.
        embedder : nn.Module, optional
            The embedder module for watermarking.
        message : torch.Tensor, optional 
            The message to embed.
        residual_coef : float, optional
            The coefficient for residual connection. The default is 1.0.

        Returns
        -------
        o : torch.tensor
            Output wavs. Shape: (batch, channels, timesteps).
        commit_loss : torch.tensor
            Commitment loss from residual vector quantizers.
        feature : torch.tensor
            Output of RVQ's first layer. Shape: (batch, timesteps, dimension)

        """
        e = self.encoder(x)
        quantized_full, _, _, quantized_list = self.quantizer(
            e, n_q=self.n_q, layers=[0, 1, 2, 3, 4, 5, 6, 7], st=0
        )
        o = self.decoder(quantized_full)
        return o

    def encode(self, x: torch.tensor):
        e = self.encoder(x)
        quantized_full, _, _, quantized_list = self.quantizer(
            e, n_q=self.n_q, layers=[0, 1, 2, 3, 4, 5, 6, 7], st=0
        )
        return quantized_full
    
    def decode(self, quantized_full: torch.tensor):
        o = self.decoder(quantized_full)
        return o
