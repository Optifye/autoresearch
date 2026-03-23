"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
------------------------------------------------------------------------------

modelcustom API requirements:

API requirements for Encoder module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :returns: (Tensor) Representations of video clip (shape=[batch_size x num_encoder_tokens x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.

API requirements for Predictor module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip tokens (shape=[batch_size x num_encoder_tokens x feature_dim])
        :param anticipation_time: (Tensor) Seconds into the future to predict for each sample in batch
            (shape=[batch_size])
        :returns: (Tensor) Representations of future frames (shape=[batch_size x num_output_tokens x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.
"""

import inspect
import logging
import os

import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.masks.utils import apply_masks
from src.models.utils.pos_embs import get_1d_sincos_pos_embed

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ALLOWED_ENCODERS = {
    "vit_giant_xformers": {"embed_dim": 1408, "num_heads": 22, "depth": 40},
    "vit_giant": {"embed_dim": 1408, "num_heads": 22, "depth": 40},
    "vit_large": {"embed_dim": 1024, "num_heads": 16, "depth": 24},
}


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    logger.info(f"Loading pretrained model from {checkpoint}")
    mmap_supported = "mmap" in inspect.signature(torch.load).parameters
    mmap_enabled = os.getenv("VJEPA_TORCH_MMAP", "1") not in {"0", "false", "False"}
    load_kwargs = {"map_location": "cpu"}
    if mmap_supported and mmap_enabled:
        load_kwargs["mmap"] = True
    weights_only = os.getenv("VJEPA_TORCH_WEIGHTS_ONLY", "1") not in {"0", "false", "False"}
    if "weights_only" in inspect.signature(torch.load).parameters and weights_only:
        load_kwargs["weights_only"] = True
    logger.info(
        "Loading pretrained model from %s (mmap=%s weights_only=%s)",
        checkpoint,
        load_kwargs.get("mmap", False),
        load_kwargs.get("weights_only", False),
    )
    checkpoint = torch.load(checkpoint, **load_kwargs)

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name") or "vit_giant_xformers"
    if enc_model_name not in ALLOWED_ENCODERS:
        raise ValueError(
            f"Unsupported encoder model_name '{enc_model_name}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_ENCODERS))}."
        )
    enc_kwargs["model_name"] = enc_model_name

    model = vit.__dict__[enc_model_name](img_size=resolution, num_frames=frames_per_clip, **enc_kwargs)
    embed_dim = getattr(model, "embed_dim", None)
    num_heads = getattr(model, "num_heads", None)
    depth = len(getattr(model, "blocks", [])) if hasattr(model, "blocks") else None
    expected = ALLOWED_ENCODERS[enc_model_name]
    if embed_dim != expected["embed_dim"]:
        raise ValueError(
            f"Encoder instantiated with embed_dim={embed_dim}; expected {expected['embed_dim']} for {enc_model_name}."
        )
    if num_heads != expected["num_heads"]:
        raise ValueError(
            f"Encoder instantiated with num_heads={num_heads}; expected {expected['num_heads']} for {enc_model_name}."
        )
    if depth not in {None, expected["depth"]}:
        raise ValueError(
            f"Encoder instantiated with depth={depth}; expected {expected['depth']} blocks for {enc_model_name}."
        )

    pretrained_dict = checkpoint[enc_ckp_key]
    # --
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    missing_keys = []
    shape_mismatch = []
    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            missing_keys.append(k)
        elif pretrained_dict[k].shape != v.shape:
            shape_mismatch.append((k, pretrained_dict[k].shape, v.shape))
            pretrained_dict[k] = v
    if missing_keys:
        raise ValueError(
            f"Checkpoint missing {len(missing_keys)} parameters (e.g., {', '.join(missing_keys[:5])}); "
            "this indicates an architecture mismatch."
        )
    if shape_mismatch:
        samples = ", ".join(
            f"{k}: {ckpt_shape}->{model_shape}" for k, ckpt_shape, model_shape in shape_mismatch[:5]
        )
        raise ValueError(
            f"Checkpoint has {len(shape_mismatch)} shape-mismatched tensors (e.g., {samples}); "
            "this indicates an architecture mismatch."
        )
    msg = model.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")
    print(model)

    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )
    del checkpoint
    return model


class ClipAggregation(nn.Module):
    """
    Process each clip independently and concatenate all tokens
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=128,
        use_pos_embed=False,
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads

        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):

        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, F, H, W = x[0][0].size()

        # Concatenate all spatial and temporal views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)

        outputs = self.model(x)

        def multiviews_postprocess(outputs):
            _, N, D = outputs.size()
            T = F // self.tubelet_size  # num temporal indices
            S = N // T  # num spatial tokens

            # Unroll outputs into a 2D array [spatial_views x temporal_views]
            eff_B = B * num_views_per_clip
            all_outputs = [[] for _ in range(num_views_per_clip)]
            for i in range(num_clips):
                o = outputs[i * eff_B : (i + 1) * eff_B]
                for j in range(num_views_per_clip):
                    all_outputs[j].append(o[j * B : (j + 1) * B])

            for i, outputs in enumerate(all_outputs):
                # Concatenate along temporal dimension
                outputs = [o.reshape(B, T, S, D) for o in outputs]
                outputs = torch.cat(outputs, dim=1).flatten(1, 2)
                # Compute positional embedding
                if (self.pos_embed is not None) and (clip_indices is not None):
                    _indices = [c[:, :: self.tubelet_size] for c in clip_indices]
                    pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, max_T, D]
                    pos_embed = apply_masks(pos_embed, _indices, concat=False)  # list(Tensor([B, T, D]))
                    pos_embed = torch.cat(pos_embed, dim=1)  # concatenate along temporal dimension
                    pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, S, 1)  # [B, T*num_clips, S, D]
                    pos_embed = pos_embed.flatten(1, 2)
                    outputs += pos_embed
                all_outputs[i] = outputs

            return all_outputs

        return multiviews_postprocess(outputs)
