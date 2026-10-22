"""
Training loop — our own. Device-aware: CUDA (Colab GPU) / MPS / CPU.

- Memmapped uint16 token shards (no RAM blowup).
- AdamW, cosine LR schedule with warmup, gradient clipping, grad accumulation.
- Mixed precision (bf16/fp16 autocast + GradScaler) on CUDA for speed.
- Resumable checkpoints (model + optimizer + step) — essential for Colab's
  ephemeral/time-limited sessions. Checkpoint dir can point at Google Drive.
"""
from __future__ import annotations
import sys, json, time, math, argparse, os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from model.gpt import GPT, GPTConfig

torch.manual_seed(1337)


def pick_device(explicit=None):
    if explicit and explicit != "auto":
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, datasets, block_size, batch_size, device, ctx, iters=50):
    model.eval()
    out = {}
    for split, data in datasets.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_at(step, warmup, max_steps, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, (max_steps - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "small.json"))
    ap.add_argument("--dataset", default="gh_all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--ckpt-dir", default="", help="override checkpoint dir (e.g. a Drive path)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    device = pick_device(args.device)
    if device == "cpu":
        torch.set_num_threads(cfg.get("cpu_threads", os.cpu_count()))

    # autocast context + grad scaler (CUDA only; bf16 if supported else fp16)
    use_amp = (device == "cuda")
    if use_amp:
        bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16 else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    else:
        import contextlib
        ctx = contextlib.nullcontext()
        scaler = None

    tokd = ROOT / "corpus" / "tokenized" / args.dataset
    meta = json.load(open(tokd / "meta.json"))
    train_data = np.memmap(tokd / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(tokd / "val.bin", dtype=np.uint16, mode="r")
    datasets = {"train": train_data, "val": val_data}

    mcfg = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
        dropout=cfg["dropout"], bias=cfg.get("bias", True),
    )
    model = GPT(mcfg).to(device)
    print(f"Device: {device} | AMP: {use_amp}"
          + (f" ({'bf16' if use_amp and bf16 else 'fp16'})" if use_amp else ""))
    print(f"Model params: {model.num_params()/1e6:.2f}M | vocab {mcfg.vocab_size} | "
          f"ctx {mcfg.block_size} | L{mcfg.n_layer} H{mcfg.n_head} D{mcfg.n_embd}")
    print(f"Train tokens: {meta['n_train']:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            betas=(0.9, 0.95), weight_decay=cfg["weight_decay"])

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (ROOT / "checkpoints" / args.dataset)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ckpt.pt"
    log_path = ROOT / "logs" / f"{args.dataset}.log"
    log_path.parent.mkdir(exist_ok=True)

    start_step = 0
    best_val = float("inf")
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        best_val = ck.get("best_val", best_val)
        if scaler and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        print(f"Resumed from step {start_step} (best_val {best_val:.4f})")

    B = cfg["batch_size"]; T = cfg["block_size"]
    accum = cfg.get("grad_accum", 1)
    max_steps = cfg["max_steps"]

    def save(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "best_val": best_val, "config": cfg,
                    "model_config": mcfg.__dict__,
                    "scaler": scaler.state_dict() if scaler else None}, ckpt_path)

    t0 = time.time()
    model.train()
    for step in range(start_step, max_steps):
        lr = lr_at(step, cfg["warmup_steps"], max_steps, cfg["lr"], cfg["min_lr"])
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(accum):
            x, y = get_batch(train_data, T, B, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += loss.item()
        if scaler:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            scaler.step(opt); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            opt.step()

        if step % cfg["log_every"] == 0:
            dt = time.time() - t0
            msg = f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | {dt:.1f}s"
            print(msg, flush=True)
            with open(log_path, "a") as lg:
                lg.write(msg + "\n")

        if step > 0 and step % cfg["eval_every"] == 0:
            losses = estimate_loss(model, datasets, T, B, device, ctx, cfg.get("eval_iters", 50))
            msg = f"  eval step {step}: train {losses['train']:.4f} val {losses['val']:.4f}"
            print(msg, flush=True)
            with open(log_path, "a") as lg:
                lg.write(msg + "\n")
            if losses["val"] < best_val:
                best_val = losses["val"]
            save(step)

    save(max_steps)
    print(f"Done. Saved {ckpt_path}")


if __name__ == "__main__":
    main()
