"""
Train GPT-2 (124M) on FineWeb with Theta heavy-tailed noise injection.

Usage:
    # Single node, 4 GPUs
    torchrun --standalone --nproc_per_node=4 train_gpt2.py --config configs/theta_adam.yaml

    # Multi-node (2 nodes × 4 GPUs)
    torchrun --nnodes=2 --node_rank=$RANK --nproc_per_node=4 \
        --master_addr=$MASTER --master_port=29500 \
        train_gpt2.py --config configs/theta_muon_adam.yaml

Aligned with the original NanoGPT repo but written cleanly:
  - Standard GPT-2 architecture (RMSNorm, RoPE, ReLU², no bias)
  - FineWeb binary shards (same format as the speedrun)
  - Optimizer grid: {Adam, AdamW, Muon+AdamW}
  - Theta Rademacher noise injection at the loss level
  - Defaults: lr=6e-4, betas=(0.9,0.95), weight_decay=0.1, cosine LR decay
"""

import argparse
import glob
import math
import os
import sys
import threading
import time
import uuid
from itertools import cycle
from pathlib import Path

# Add Theta repo to path so `theta` package is importable via the symlink
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "Theta"))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", action="append", default=[],
                   help="YAML config file(s), merged left-to-right; CLI overrides all")

    # Model
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-embd", type=int, default=768)
    p.add_argument("--vocab-size", type=int, default=50304)  # GPT-2 50257, padded
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0)

    # Data
    p.add_argument("--train-files", type=str,
                   default="data/fineweb_train_*.bin")
    p.add_argument("--val-files", type=str,
                   default="data/fineweb_val_*.bin")
    p.add_argument("--val-tokens", type=int, default=10_485_760)

    # Training
    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--batch-tokens", type=int, default=491_520,
                   help="Total tokens per step across all GPUs")
    p.add_argument("--val-interval", type=int, default=250)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--ckpt-interval", type=int, default=1000)
    p.add_argument("--ckpt-dir", type=str, default="logs/results")
    p.add_argument("--ckpt-path", type=str, default=None,
                   help="Save rolling checkpoint to this exact path (overwrite)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume training from this checkpoint file")
    p.add_argument("--seed", type=int, default=1337)

    # Optimizer
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adam", "adamw", "muon_adam"])
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # LR schedule
    p.add_argument("--lr-decay", type=str, default="none",
                   choices=["none", "cosine", "linear"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--cooldown-frac", type=float, default=0.6,
                   help="Fraction of training spent in LR cooldown (for linear decay)")
    p.add_argument("--min-lr-frac", type=float, default=0.1,
                   help="Final LR as fraction of peak (for cosine and linear decay)")

    # Muon-specific (only when optimizer=muon_adam)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--muon-weight-decay", type=float, default=None,
                   help="Weight decay for Muon params (defaults to --weight-decay)")

    # Theta noise
    p.add_argument("--theta-enabled", action="store_true")
    p.add_argument("--theta-alpha", type=float, default=1.4)
    p.add_argument("--theta-scale", type=float, default=0.5)
    p.add_argument("--theta-tail-prob", type=float, default=0.9)
    p.add_argument("--theta-noise-stop-frac", type=float, default=0.9)
    p.add_argument("--theta-rademacher", action="store_true",
                   help="Use Rademacher signs on per-token loss (default: uniform (1+z) scaling)")
    p.add_argument("--theta-balanced-perturbation", action="store_true",
                   help="Balance Rademacher signs (exactly N/2 +1 and -1). "
                        "Only used when --theta-rademacher is set.")

    # Wandb
    p.add_argument("--wandb", action="store_true", help="enable wandb logging")
    p.add_argument("--wandb-project", type=str, default="Theta_NanoGPT")
    p.add_argument("--wandb-entity", type=str, default="vaseline555")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-group-name", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])

    args = p.parse_args()

    # Load YAML configs (merged left-to-right; CLI overrides all)
    if args.config:
        import yaml
        # First pass: merge all config files into a single dict
        merged_cfg = {}
        for config_path in args.config:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            merged_cfg.update(cfg)
        # Second pass: figure out which args were explicitly set on the CLI
        # so we can let CLI flags override config values
        cli_set = set()
        for action in p._actions:
            if action.option_strings:
                key = action.dest
                if getattr(args, key, None) != action.default:
                    cli_set.add(key)
        # Apply merged config values, skipping CLI-explicit overrides
        for k, v in merged_cfg.items():
            key = k.replace("-", "_")
            if hasattr(args, key) and key not in cli_set:
                # Coerce strings like "3e-4" to float
                default = p.get_default(key)
                if isinstance(default, float) and isinstance(v, str):
                    v = float(v)
                elif isinstance(default, int) and isinstance(v, str):
                    v = int(v)
                setattr(args, key, v)

    return args


# ---------------------------------------------------------------------------
# Data loading — FineWeb binary shards
# ---------------------------------------------------------------------------

BOS_ID = 50256

def load_shard(path: Path) -> Tensor:
    """Load a FineWeb binary shard. Returns uint16 token tensor."""
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "bad magic number"
    assert header[1] == 1, "unsupported version"
    n = int(header[2])
    with path.open("rb", buffering=0) as f:
        tokens = torch.empty(n, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        f.readinto(tokens.numpy())
    return tokens


def data_generator(file_pattern: str, seq_len: int, rank: int, world_size: int):
    """Yield (input, target) pairs of shape (seq_len,) in int64.

    Cycles through shards. Each rank gets a different slice.
    Simple contiguous chunking — no BOS alignment (keeps it simple).
    """
    files = sorted(glob.glob(file_pattern))
    assert files, f"no files matching {file_pattern}"

    for path in cycle(files):
        tokens = load_shard(Path(path))
        n = tokens.numel()
        # Each rank gets a contiguous region of the shard
        chunk = n // world_size
        start = rank * chunk
        local_tokens = tokens[start : start + chunk].to(torch.int64)

        pos = 0
        while pos + seq_len + 1 <= local_tokens.numel():
            x = local_tokens[pos : pos + seq_len]
            y = local_tokens[pos + 1 : pos + seq_len + 1]
            yield x, y
            pos += seq_len


# ---------------------------------------------------------------------------
# Model — clean GPT-2
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, eps=1e-6)


def precompute_rope(dim: int, max_len: int, base: float = 10000.0) -> Tensor:
    """Precompute RoPE complex exponentials."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary embeddings. x: (B, n_heads, T, head_dim)."""
    T = x.size(2)
    freqs = freqs[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_complex * freqs).flatten(-2)
    return out.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor, freqs: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, T, n_head, head_dim)
        q = q.transpose(1, 2)    # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # ReLU² activation (squared ReLU)
        h = F.relu(self.fc(x))
        h = h * h
        return self.dropout(self.proj(h))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: Tensor, freqs: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), freqs)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int, n_head: int,
                 n_embd: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        # Precompute RoPE frequencies
        head_dim = n_embd // n_head
        self.register_buffer(
            "rope_freqs", precompute_rope(head_dim, max_seq_len), persistent=False
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Tensor | None = None):
        """
        idx:     (B, T) int64 token IDs
        targets: (B, T) int64 targets

        Returns:
            If targets given: per-token losses (B, T), float32
            Otherwise:        logits (B, T, V), float32
        """
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_freqs)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
            ).view(idx.shape)
        return logits


# ---------------------------------------------------------------------------
# Muon optimizer — clean standalone implementation
# ---------------------------------------------------------------------------

@torch.no_grad()
def newton_schulz(M: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Quintic Newton-Schulz iteration for orthogonalization (Muon).

    Computes the nearest orthogonal matrix to M using the degree-5
    iteration from the modded-nanogpt Muon implementation. Coefficients
    (a, b, c) are chosen to maximise the slope at zero, which minimises
    the number of iterations needed for convergence.

    Input:  M of shape (m, n)
    Output: orthogonalized M of same shape, dtype matching M
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M.bfloat16()
    X = X / (X.norm() + eps)
    # Work with the wide orientation for numerical stability
    transposed = M.size(0) > M.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if transposed:
        X = X.T
    return X.to(M.dtype)


class Muon:
    """
    Muon optimizer for 2D weight matrices.

    Applies SGD with Nesterov momentum, then orthogonalizes the update
    via Newton-Schulz iteration (approximate Polar decomposition).

    Reference: https://kellerjordan.github.io/posts/muon/
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.state = {p: {"buf": torch.zeros_like(p, dtype=torch.float32)}
                      for p in self.params}

    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.float()
            buf = self.state[p]["buf"]

            # Nesterov momentum
            buf.mul_(self.momentum).add_(g)
            update = g + self.momentum * buf

            # Orthogonalize 2D params. The Newton-Schulz output is orthogonal
            # (per-element RMS ~ 1/sqrt(max_dim)); the reference Muon keeps that
            # magnitude and only applies aspect-ratio scaling max(1, rows/cols)**0.5
            # (== 1.0 for square matrices), which is why muon_lr=0.02 is calibrated.
            # (The previous max(rows,cols)**0.5 rescaled updates to unit RMS,
            #  ~sqrt(dim) ≈ 28x too large, so Muon over-stepped and stalled.)
            if update.ndim >= 2:
                if update.size(0) == 3 * update.size(1):
                    # Fused QKV: orthogonalize Q, K, V blocks independently
                    chunks = update.split(update.size(1))
                    update = torch.cat([newton_schulz(c) for c in chunks])
                    c0 = chunks[0]
                    scale = max(1.0, c0.size(0) / c0.size(1)) ** 0.5
                else:
                    update = newton_schulz(update)
                    scale = max(1.0, update.size(0) / update.size(1)) ** 0.5
            else:
                scale = 1.0

            # Weight decay
            if self.weight_decay > 0:
                p.data.add_(p.data, alpha=-self.lr * self.weight_decay)

            p.data.add_(update, alpha=-self.lr * scale)

    def state_dict(self):
        return {"state": {i: {k: v.clone() for k, v in self.state[p].items()}
                          for i, p in enumerate(self.params)}}

    def load_state_dict(self, sd):
        for idx, s in sd["state"].items():
            idx = int(idx)
            if idx < len(self.params):
                p = self.params[idx]
                for k, v in s.items():
                    self.state[p][k] = v.to(p.device)


class MuonAdam:
    """
    Combined optimizer: Muon for 2D projection matrices, AdamW for the rest.
    Aligned with the original Muon reference (2024-10-10).

    Param routing:
      - 2D weight tensors (not embeddings/lm_head): Muon
      - Everything else: AdamW
    """

    def __init__(self, named_params, lr: float = 3e-4,
                 muon_lr: float = 0.02, muon_momentum: float = 0.95,
                 betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 muon_weight_decay: float | None = None):
        muon_params = []
        adam_params = []
        for name, p in named_params:
            if not p.requires_grad:
                continue
            # 2D weights excluding embeddings and lm_head → Muon
            if p.ndim == 2 and "tok_emb" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                adam_params.append(p)

        muon_wd = muon_weight_decay if muon_weight_decay is not None else weight_decay
        self.muon = Muon(muon_params, lr=muon_lr, momentum=muon_momentum,
                         weight_decay=muon_wd)
        self.adam = torch.optim.AdamW(
            adam_params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay,
        )

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adam.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, sd):
        self.muon.load_state_dict(sd["muon"])
        self.adam.load_state_dict(sd["adam"])

    @property
    def param_groups(self):
        return self.adam.param_groups


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_multiplier(step: int, args) -> float:
    """Return LR multiplier. Constant by default."""
    # Warmup
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return (step + 1) / args.warmup_steps

    if args.lr_decay == "none":
        return 1.0

    # Cosine decay after warmup (original NanoGPT style: decays to min_lr_frac)
    if args.lr_decay == "cosine":
        t = step - args.warmup_steps
        T = args.train_steps - args.warmup_steps
        coeff = 0.5 * (1.0 + math.cos(math.pi * t / T))
        return args.min_lr_frac + coeff * (1.0 - args.min_lr_frac)

    # Linear cooldown (speedrun style): constant until cooldown, then linear to min_lr_frac
    if args.lr_decay == "linear":
        cd_start = int(args.train_steps * (1.0 - args.cooldown_frac))
        if step < cd_start:
            return 1.0
        t = (step - cd_start) / max(args.train_steps - cd_start, 1)
        return 1.0 * (1.0 - t) + args.min_lr_frac * t

    return 1.0


def set_lr(optimizer, lr_mult: float, args):
    """Set learning rate on optimizer."""
    if isinstance(optimizer, MuonAdam):
        for pg in optimizer.adam.param_groups:
            pg["lr"] = args.lr * lr_mult
        optimizer.muon.lr = args.muon_lr * lr_mult
    else:
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * lr_mult


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize distributed process group. Returns (rank, local_rank, world_size)."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank, local_rank, world_size = 0, 0, 1

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, args, rank, world_size, device):
    """Compute mean validation loss."""
    model.eval()
    gen = data_generator(args.val_files, args.max_seq_len, rank, world_size)
    tokens_seen = 0
    total_loss = 0.0
    n_batches = 0

    while tokens_seen < args.val_tokens // world_size:
        x, y = next(gen)
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        per_token = model(x, y)           # (1, T)
        total_loss += per_token.mean().item()
        n_batches += 1
        tokens_seen += x.numel()

    avg = total_loss / max(n_batches, 1)

    # Average across ranks
    if world_size > 1:
        t = torch.tensor([avg], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        avg = t.item()

    model.train()
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Resolve relative paths to script location (not CWD)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    for attr in ("ckpt_dir", "train_files", "val_files"):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.join(_script_dir, val))
    for attr in ("ckpt_path", "resume"):
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            setattr(args, attr, os.path.join(_script_dir, val))

    # ---- Resume: peek at checkpoint for step and wandb run id ----
    start_step = 0
    wandb_run_id = None
    resume_ckpt = None
    if args.resume and os.path.isfile(args.resume):
        resume_ckpt = torch.load(args.resume, map_location="cpu",
                                 weights_only=False)
        start_step = resume_ckpt["step"]
        wandb_run_id = resume_ckpt.get("wandb_run_id")
        if is_main:
            print(f"Resuming from {args.resume} at step {start_step}")
    elif args.resume and is_main:
        print(f"No checkpoint at {args.resume}, starting fresh")

    if is_main:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        print(f"{'='*60}")
        print(f"GPT-2 Training — {world_size} GPU(s)")
        print(f"{'='*60}")
        print(f"Model:     n_layer={args.n_layer}, n_head={args.n_head}, "
              f"n_embd={args.n_embd}")
        print(f"Optimizer: {args.optimizer}, lr={args.lr}, wd={args.weight_decay}")
        print(f"LR decay:  {args.lr_decay}")
        print(f"Theta:     enabled={args.theta_enabled}", end="")
        if args.theta_enabled:
            print(f", alpha={args.theta_alpha}, scale={args.theta_scale}, "
                  f"tail_prob={args.theta_tail_prob}, "
                  f"noise_stop_frac={args.theta_noise_stop_frac}")
        else:
            print()
        print(f"Steps:     {args.train_steps}, batch_tokens={args.batch_tokens}")
        print(f"{'='*60}")

    # ---- Wandb ----
    if args.wandb and is_main:
        wandb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir
        import wandb
        # Default run name: sweep_base_adam or sweep_theta_muon_adam, etc.
        prefix = "theta" if args.theta_enabled else "base"
        default_run_name = f"sweep_{prefix}_{args.optimizer}"
        # Default group name: optimizer type (for dashboard grouping)
        default_group_name = args.optimizer
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or default_run_name,
            group=args.wandb_group_name or default_group_name,
            mode=args.wandb_mode,
            dir=wandb_dir,
            config=vars(args),
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
        )
        wandb_run_id = wandb.run.id

    # ---- Model ----
    model = GPT2(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device).bfloat16()

    # Wrap in DDP
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
        )
    raw_model = model.module if world_size > 1 else model

    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        print(f"Parameters: {n_params:,}")

    # ---- Optimizer ----
    if args.optimizer == "adam":
        all_params = [p for p in raw_model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            all_params, lr=args.lr,
            betas=tuple(args.betas), eps=args.eps,
        )
    elif args.optimizer == "adamw":
        decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=args.lr,
            betas=tuple(args.betas), eps=args.eps,
        )
    elif args.optimizer == "muon_adam":
        optimizer = MuonAdam(
            raw_model.named_parameters(),
            lr=args.lr, muon_lr=args.muon_lr,
            muon_momentum=args.muon_momentum,
            betas=tuple(args.betas), eps=args.eps,
            weight_decay=args.weight_decay,
            muon_weight_decay=args.muon_weight_decay,
        )
    else:
        raise ValueError(f"unknown optimizer: {args.optimizer}")

    # ---- Restore checkpoint state ----
    if resume_ckpt is not None:
        raw_model.load_state_dict(resume_ckpt["model"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        if is_main:
            print(f"Restored model & optimizer from step {start_step}")
        del resume_ckpt

    # ---- Theta noise ----
    if args.theta_enabled:
        from theta._theta import sample_lomax, lomax_cdf
        noise_stop_step = int(args.train_steps * args.theta_noise_stop_frac)
        if is_main:
            print(f"Theta noise active for steps 0..{noise_stop_step - 1}")

    # ---- Gradient accumulation ----
    tokens_per_gpu = args.batch_tokens // world_size
    micro_batch_tokens = args.max_seq_len  # 1 sequence per micro-batch
    grad_accum_steps = tokens_per_gpu // micro_batch_tokens
    assert grad_accum_steps >= 1, (
        f"batch_tokens={args.batch_tokens} too small for "
        f"seq_len={args.max_seq_len} × {world_size} GPUs"
    )
    grad_scale = 1.0 / grad_accum_steps
    if is_main:
        print(f"Grad accum: {grad_accum_steps} steps, "
              f"{micro_batch_tokens} tokens/micro-batch")

    # ---- Data ----
    train_gen = data_generator(
        args.train_files, args.max_seq_len, rank, world_size
    )

    # ---- Training loop ----
    model.train()
    t0 = time.perf_counter()
    theta_active_count = 0

    for step in range(start_step, args.train_steps + 1):
        # ---- Validation ----
        if step % args.val_interval == 0 or step == args.train_steps:
            val_loss = validate(raw_model, args, rank, world_size, device)
            if is_main:
                elapsed = time.perf_counter() - t0
                print(f"step={step:>5d} | val_loss={val_loss:.4f} | "
                      f"time={elapsed:.1f}s")
                if args.wandb:
                    wandb.log({"val/loss": val_loss, "val/time": elapsed}, step=step)

        if step == args.train_steps:
            break

        # ---- LR schedule ----
        lr_mult = get_lr_multiplier(step, args)
        set_lr(optimizer, lr_mult, args)

        # ---- Sample Theta noise for this step ----
        z = 0.0
        noise_active = False
        if args.theta_enabled and step < noise_stop_step:
            # Deterministic per-step seed so all ranks get the same Z
            rng = torch.Generator()
            rng.manual_seed(args.seed + step)
            u = torch.rand((), generator=rng).item()
            z_val = args.theta_scale * ((1.0 - u) ** (-1.0 / args.theta_alpha) - 1.0)
            cdf = lomax_cdf(z_val, args.theta_alpha, args.theta_scale)
            noise_active = (cdf >= args.theta_tail_prob)
            if noise_active:
                z = z_val
                theta_active_count += 1

        # ---- Forward / backward ----
        optimizer.zero_grad(set_to_none=True)

        for accum_idx in range(grad_accum_steps):
            x, y = next(train_gen)
            x = x.unsqueeze(0).to(device)  # (1, T)
            y = y.unsqueeze(0).to(device)

            # Sync gradients only on last accumulation step
            if world_size > 1:
                model.require_backward_grad_sync = (
                    accum_idx == grad_accum_steps - 1
                )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                per_token_loss = model(x, y)  # (1, T)

                if noise_active:
                    if args.theta_rademacher:
                        num_tokens = per_token_loss.numel()
                        sign_rng = torch.Generator(device=device)
                        sign_rng.manual_seed(args.seed + step * 1000 + accum_idx)
                        if args.theta_balanced_perturbation:
                            half = num_tokens // 2
                            signs = torch.ones(num_tokens, device=device)
                            signs[half:] = -1.0
                            signs = signs[torch.randperm(num_tokens, device=device,
                                                         generator=sign_rng)]
                        else:
                            signs = 2.0 * torch.bernoulli(
                                torch.full((num_tokens,), 0.5, device=device),
                                generator=sign_rng) - 1.0
                        loss = ((1.0 + z * signs) * per_token_loss.view(-1)).mean()
                    else:
                        loss = (1.0 + z) * per_token_loss.mean()
                else:
                    loss = per_token_loss.mean()

                loss = loss * grad_scale

            loss.backward()

        # ---- Gradient clipping ----
        grad_norm = 0.0
        if args.grad_clip > 0:
            if isinstance(optimizer, MuonAdam):
                # Only clip Adam params — Muon handles its own update norm
                # via orthogonalization (Newton-Schulz)
                adam_params = [p for pg in optimizer.adam.param_groups
                               for p in pg["params"]]
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    adam_params, args.grad_clip
                ).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), args.grad_clip
                ).item()

        # ---- Optimizer step ----
        optimizer.step()

        # ---- Logging ----
        if is_main and step % args.log_interval == 0:
            elapsed = time.perf_counter() - t0
            lr_now = args.lr * lr_mult
            train_loss = loss.item() / grad_scale
            msg = (f"step={step:>5d} | loss={train_loss:.4f} | "
                   f"lr={lr_now:.2e} | time={elapsed:.1f}s")
            if args.theta_enabled:
                frac = theta_active_count / (step + 1)
                msg += f" | theta_z={z:.3f} active={int(noise_active)} frac={frac:.2f}"
            print(msg)

            if args.wandb:
                log_dict = {
                    "train/loss": train_loss,
                    "train/lr": lr_now,
                    "train/time": elapsed,
                    "train/grad_norm": grad_norm,
                }
                if args.theta_enabled:
                    log_dict["theta/z"] = z
                    log_dict["theta/active"] = int(noise_active)
                    log_dict["theta/active_frac"] = frac
                    log_dict["noise/active"] = int(noise_active)
                wandb.log(log_dict, step=step)

        # ---- Checkpointing ----
        if (is_main and args.ckpt_interval > 0 and
                step > 0 and step % args.ckpt_interval == 0):
            ckpt = {
                "step": step,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "wandb_run_id": wandb_run_id,
            }
            if args.ckpt_path:
                path = args.ckpt_path
            else:
                path = os.path.join(args.ckpt_dir, f"ckpt_step{step}.pt")
            torch.save(ckpt, path)
            print(f"  saved {path}")

    # ---- Final checkpoint ----
    if is_main:
        ckpt = {
            "step": args.train_steps,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "wandb_run_id": wandb_run_id,
        }
        if args.ckpt_path:
            path = args.ckpt_path
        else:
            path = os.path.join(args.ckpt_dir, f"ckpt_step{args.train_steps}.pt")
        torch.save(ckpt, path)
        print(f"  saved {path}")

    # ---- Cleanup ----
    if is_main:
        total = time.perf_counter() - t0
        print(f"\nDone. Total time: {total:.1f}s")
        if args.wandb:
            wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
