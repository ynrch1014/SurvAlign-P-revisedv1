import os
import torch
from types import SimpleNamespace
from torch.nn.parallel import DistributedDataParallel as DDP

from models import WatermarkModel, AudioFusionModel
from speechtokenizer import SpeechTokenizer


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        if hasattr(self.model, name):
            return getattr(self.model, name)
        elif hasattr(self.model, "module") and hasattr(self.model.module, name):
            return getattr(self.model.module, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class WatermarkBase:
    def __init__(self, cfg: SimpleNamespace):
        self.cfg = cfg
        self.local_rank = getattr(cfg, "local_rank", None)
        self.device = getattr(cfg, "device", "cpu")
        self.sample_rate = getattr(cfg, "sample_rate", 16000)
        self.nbits = getattr(cfg, "nbits", 16)

        # Load SpeechTokenizer (used as VAE here)
        config_path = os.path.join(
            "speechtokenizer", "pretrained_model", "speechtokenizer_hubert_avg_config.json"
        )
        ckpt_path = os.path.join("speechtokenizer", "pretrained_model", "SpeechTokenizer.pt")
        self.vae = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path).to(self.device)
        for _, param in self.vae.named_parameters():
            param.requires_grad = False

        # Watermark model and fusion model
        self.model = WatermarkModel(cfg).to(self.device)
        self.fusion_model = AudioFusionModel(
            n_fft=256, hop_length=64, win_length=256, hidden_dim=64, nbits=self.nbits
        ).to(self.device)

        if self.local_rank is not None:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=False)
            self.fusion_model = DDP(self.fusion_model, device_ids=[self.local_rank], find_unused_parameters=False)

        self.model = ModelWrapper(self.model)
        self.fusion_model = ModelWrapper(self.fusion_model)

    def load_model(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path} ...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        # load watermark model
        new_wm_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
        self.model.load_state_dict(new_wm_dict, strict=True)

        # load fusion model if present
        new_fusion_dict = {k.replace("module.", ""): v for k, v in checkpoint["fusion_state_dict"].items()}
        self.fusion_model.load_state_dict(new_fusion_dict, strict=True)

        print("Model weights loaded.")


