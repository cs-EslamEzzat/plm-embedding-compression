"""
embeddings.py
=============
Turns protein sequences into fixed-length numeric vectors (embeddings)
using a protein language model, then caches them to disk so the expensive
model inference runs only once.

Supported embedders:
  - "esm2"      : Meta's ESM-2 (via the `fair-esm` package)        -> 1280-d
  - "prott5"    : Rostlab ProtT5 (via HuggingFace `transformers`)  -> 1024-d
  - "synthetic" : random vectors, for the --smoke test only

Embeddings are mean-pooled over residues to give one vector per protein.
"""

from __future__ import annotations
import hashlib
import os
from typing import List

import numpy as np

try:
    from tqdm import tqdm
except ImportError:                      # progress bar is optional
    def tqdm(it, **kwargs):
        return it

from config import Config


def _cache_path(cfg: Config, seqs: List[str]) -> str:
    """A cache filename keyed by embedder + a hash of all sequences."""
    h = hashlib.md5(("||".join(seqs)).encode()).hexdigest()[:10]
    os.makedirs(cfg.cache_dir, exist_ok=True)
    return os.path.join(cfg.cache_dir, f"emb_{cfg.embedder}_{h}.npy")


# ----------------------------------------------------------------------
# Individual embedders
# ----------------------------------------------------------------------
def _embed_esm2(seqs: List[str], cfg: Config) -> np.ndarray:
    import torch
    import esm

    model, alphabet = esm.pretrained.__dict__[cfg.esm2_model]()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    repr_layer = model.num_layers

    out = []
    for i in tqdm(range(0, len(seqs), cfg.embed_batch_size), desc="ESM-2"):
        batch = seqs[i:i + cfg.embed_batch_size]
        data = [(f"p{j}", s[:cfg.max_seq_len]) for j, s in enumerate(batch)]
        _, _, toks = batch_converter(data)
        toks = toks.to(device)
        with torch.no_grad():
            res = model(toks, repr_layers=[repr_layer])
        reps = res["representations"][repr_layer]
        for k, s in enumerate(batch):
            L = min(len(s), cfg.max_seq_len)
            # positions 1..L are residues (0 is the BOS token)
            vec = reps[k, 1:L + 1].mean(0).cpu().numpy()
            out.append(vec)
    return np.vstack(out)


def _embed_prott5(seqs: List[str], cfg: Config) -> np.ndarray:
    import re
    import torch
    from transformers import T5EncoderModel, T5Tokenizer

    tok = T5Tokenizer.from_pretrained(cfg.prott5_model, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(cfg.prott5_model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    out = []
    for i in tqdm(range(0, len(seqs), cfg.embed_batch_size), desc="ProtT5"):
        batch = seqs[i:i + cfg.embed_batch_size]
        proc = [" ".join(list(re.sub(r"[UZOB]", "X", s[:cfg.max_seq_len])))
                for s in batch]
        enc = tok.batch_encode_plus(proc, add_special_tokens=True,
                                    padding="longest", return_tensors="pt")
        ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        with torch.no_grad():
            emb = model(input_ids=ids, attention_mask=att).last_hidden_state
        for k, s in enumerate(batch):
            L = min(len(s), cfg.max_seq_len)
            vec = emb[k, :L].mean(0).cpu().numpy()
            out.append(vec)
    return np.vstack(out)


def _embed_synthetic(seqs: List[str], cfg: Config, dim: int = 128) -> np.ndarray:
    """Deterministic pseudo-embeddings: cheap, just for plumbing tests."""
    rng = np.random.default_rng(cfg.seed)
    base = rng.normal(size=(dim,))
    out = np.zeros((len(seqs), dim))
    for i, s in enumerate(seqs):
        seed = int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
        r = np.random.default_rng(seed)
        out[i] = base + r.normal(scale=0.5, size=dim)
    return out


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def get_embeddings(seqs: List[str], cfg: Config,
                   use_cache: bool = True) -> np.ndarray:
    """Return an (n_proteins, D) embedding matrix, using the disk cache."""
    cache = _cache_path(cfg, seqs)
    if use_cache and os.path.exists(cache):
        if cfg.verbose:
            print(f"[emb] loading cached embeddings: {cache}")
        return np.load(cache)

    if cfg.embedder == "esm2":
        X = _embed_esm2(seqs, cfg)
    elif cfg.embedder == "prott5":
        X = _embed_prott5(seqs, cfg)
    elif cfg.embedder == "synthetic":
        X = _embed_synthetic(seqs, cfg)
    else:
        raise ValueError(f"Unknown embedder: {cfg.embedder}")

    if use_cache:
        np.save(cache, X)
        if cfg.verbose:
            print(f"[emb] cached embeddings -> {cache}  shape={X.shape}")
    return X
