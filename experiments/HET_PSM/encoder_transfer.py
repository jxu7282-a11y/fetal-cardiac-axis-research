"""Encoder-only transfer for HET-PSM.

Reuses the E3 building blocks (checkpoint de-prefixing, key classification and
the ``decoder.encoder is encoder`` structural guard) without editing E3.

Difference from ``nnUNetTrainer_E3EncoderOnly.transfer_encoder_weights``: E3
raises if any decoder tensor happens to *equal* the mid-trimester value.  That
heuristic misfires when both networks are built from the same RNG seed (the
decoder then starts out identical).  HET-PSM instead proves the decoder is
untouched by comparing a pre/post snapshot, and asserts that no side-head or
decoder parameter is ever written.
"""
from __future__ import annotations

import os
import sys

import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src", "nnunet_ext")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from nnUNetTrainer_E3EncoderOnly import (  # noqa: E402
    ENCODER_PREFIXES,
    _is_encoder_key,
    load_network_weights,
    unwrap_network,
)

__all__ = [
    "ENCODER_PREFIXES",
    "_is_encoder_key",
    "load_network_weights",
    "transfer_encoder_weights",
    "unwrap_network",
]


def transfer_encoder_weights(net, mid: dict) -> dict:
    """Copy only encoder tensors of ``mid`` into ``net``; return an audit dict."""
    if getattr(net, "encoder", None) is None or getattr(net, "decoder", None) is None:
        raise RuntimeError("network does not expose .encoder/.decoder as expected")
    if getattr(net.decoder, "encoder", None) is not net.encoder:
        raise RuntimeError(
            "net.decoder.encoder is not net.encoder; encoder/decoder split is "
            "ambiguous, refusing to transfer weights")

    sd_before = net.state_dict()
    enc_keys = sorted(k for k in sd_before if _is_encoder_key(k))
    dec_keys = sorted(k for k in sd_before if not _is_encoder_key(k))
    if not enc_keys:
        raise RuntimeError("no encoder state_dict entries found")

    fresh_dec = {k: sd_before[k].detach().cpu().clone() for k in dec_keys}

    uniq = {}
    for name, param in net.named_parameters():
        if _is_encoder_key(name):
            uniq.setdefault(id(param), (name, param))
    enc_params = list(uniq.values())

    missing = [n for n, _ in enc_params if n not in mid]
    mismatched = [
        {"name": n, "mid_shape": list(mid[n].shape), "net_shape": list(p.shape)}
        for n, p in enc_params
        if n in mid and tuple(mid[n].shape) != tuple(p.shape)]
    if missing or mismatched:
        raise RuntimeError(
            f"encoder transfer impossible: missing={missing[:5]} ({len(missing)}), "
            f"shape_mismatch={mismatched[:5]} ({len(mismatched)})")

    for name, param in enc_params:
        with torch.no_grad():
            param.copy_(mid[name].to(device=param.device, dtype=param.dtype))

    sd_after = net.state_dict()
    enc_not_equal = [k for k in enc_keys
                     if not torch.equal(sd_after[k].detach().cpu(),
                                        mid[k].detach().to(torch.float32).cpu())]
    if enc_not_equal:
        raise RuntimeError(
            f"{len(enc_not_equal)}/{len(enc_keys)} encoder entries != mid: "
            f"{enc_not_equal[:5]}")

    dec_changed = [k for k in dec_keys
                   if not torch.equal(fresh_dec[k], sd_after[k].detach().cpu())]
    if dec_changed:
        raise RuntimeError(
            f"{len(dec_changed)} decoder/seg entries changed during transfer: "
            f"{dec_changed[:5]}")

    side_written = [k for k in dec_changed if k.startswith("side_head.")]
    if side_written:
        raise RuntimeError(f"side head was written by the transfer: {side_written}")

    mid_side_keys = sorted(k for k in mid if k.startswith("side_head."))
    mid_dec_keys = sorted(k for k in mid
                          if not _is_encoder_key(k) and not k.startswith("side_head."))
    return {
        "encoder_state_dict_entries": len(enc_keys),
        "encoder_unique_parameter_objects": len(enc_params),
        "encoder_entries_not_equal_to_mid": [],
        "encoder_entries_in_mid_but_not_in_net": sorted(
            k for k in mid if _is_encoder_key(k) and k not in set(enc_keys)),
        "decoder_and_seg_state_dict_entries": len(dec_keys),
        "decoder_and_seg_entries_modified_by_transfer": 0,
        "side_head_entries_in_early_net": len(
            [k for k in dec_keys if k.startswith("side_head.")]),
        "side_head_entries_in_mid_checkpoint": len(mid_side_keys),
        "mid_decoder_entries_discarded": len(mid_dec_keys),
        "mid_decoder_tensor_count": int(sum(mid[k].numel() for k in mid_dec_keys)),
        "encoder_tensor_count": int(sum(p.numel() for _, p in enc_params)),
        "seg_layer_entries": len([k for k in dec_keys if k.startswith("decoder.seg_layers.")]),
        "transpconv_entries": len([k for k in dec_keys if "transpconvs" in k]),
        "encoder_loaded_names": sorted(n for n, _ in enc_params),
    }
