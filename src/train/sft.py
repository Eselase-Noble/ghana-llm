"""
Supervised fine-tuning (SFT) — turn the base LM into a prompt->answer assistant.

Chat format (literal marker strings, tokenized by our existing BPE):
    <|user|>\n{prompt}\n<|assistant|>\n{response}<eot>
Loss is applied ONLY to the response tokens (prompt is masked with -1), so the
model learns to produce answers, not to echo prompts.

Starts from a base checkpoint (the pretrained gh_v4 model) and writes an SFT
checkpoint. Device-aware (CUDA/MPS/CPU) + AMP on CUDA.
"""
from __future__ import annotations
import sys, json, time, math, argparse, os, random
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from model.gpt import GPT, GPTConfig
from tokenizer.bpe import BPETokenizer

USER_PRE = "<|user|>\n"
ASST_PRE = "\n<|assistant|>\n"


def pick_device(x):
    if x and x != "auto": return x
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def build_examples(sft_path, tok, block_size):
    eot = tok.special["<|endoftext|>"]
    examples = []
    skipped = 0
    for line in open(sft_path, encoding="utf-8"):
        e = json.loads(line)
        P = tok.encode(USER_PRE + e["prompt"] + ASST_PRE)
        R = tok.encode(e["response"]) + [eot]
        seq = P + R
        if len(seq) > block_size + 1:
            seq = seq[:block_size + 1]
        if len(seq) < 2 or len(P) >= len(seq):
            skipped += 1; continue
        x = seq[:-1]
        y = seq[1:]
        # mask prompt positions: we only train from the last prompt token onward
        mask_until = len(P) - 1
        y = [-1] * min(mask_until, len(y)) + y[mask_until:]
        examples.append((x, y))
    return examples, skipped


def make_batch(examples, idxs, pad_id, device):
    L = max(len(examples[i][0]) for i in idxs)
    B = len(idxs)
    X = torch.full((B, L), pad_id, dtype=torch.long)
    Y = torch.full((B, L), -1, dtype=torch.long)
    for b, i in enumerate(idxs):
        x, y = examples[i]
        X[b, :len(x)] = torch.tensor(x, dtype=torch.long)
        Y[b, :len(y)] = torch.tensor(y, dtype=torch.long)
    return X.to(device), Y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gh_v4", help="for the tokenizer")
    ap.add_argument("--base", default="", help="base checkpoint path (defaults to checkpoints/<dataset>/ckpt.pt)")
    ap.add_argument("--sft-data", default=str(ROOT / "corpus" / "instructions" / "sft.jsonl"))
    ap.add_argument("--out", default="", help="output SFT checkpoint path")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--resume", action="store_true",
                    help="continue from a partially-finished sft.pt (e.g. after a Colab drop)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = pick_device(args.device)
    tokd = ROOT / "corpus" / "tokenized" / args.dataset
    tok = BPETokenizer.load(tokd / "tokenizer")
    pad_id = tok.special["<|pad|>"]

    base_path = args.base or (ROOT / "checkpoints" / args.dataset / "ckpt.pt")
    ck = torch.load(base_path, map_location=device)
    mcfg = GPTConfig(**ck["model_config"])
    model = GPT(mcfg).to(device)
    model.load_state_dict(ck["model"])
    print(f"Device {device} | base {base_path} | block {mcfg.block_size} | params {model.num_params()/1e6:.1f}M")

    examples, skipped = build_examples(args.sft_data, tok, mcfg.block_size)
    print(f"SFT examples: {len(examples)} (skipped {skipped})")

    use_amp = device == "cuda"
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    else:
        import contextlib; ctx = contextlib.nullcontext(); scaler = None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    out_path = Path(args.out) if args.out else (ROOT / "checkpoints" / args.dataset / "sft.pt")
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    bak_path = out_path.with_name(out_path.name + ".bak")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(1337)
    order = list(range(len(examples)))
    steps_per_epoch = math.ceil(len(examples) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(50, total_steps // 20 + 1)
    step = 0
    start_epoch = 0
    if args.resume:
        # Pick up a half-finished SFT run: load model+opt and skip completed epochs.
        for cand in (out_path, bak_path):
            if not cand.exists():
                continue
            try:
                rc = torch.load(cand, map_location=device)
                model.load_state_dict(rc["model"])
                if rc.get("opt"):
                    opt.load_state_dict(rc["opt"])
                start_epoch = rc.get("epoch", 0)
                step = rc.get("step", start_epoch * steps_per_epoch)
                print(f"Resumed SFT from {cand.name}: {start_epoch}/{args.epochs} epochs done")
                break
            except Exception as e:
                print(f"WARN: {cand.name} unreadable ({e}); trying next candidate")

    def save_sft(epoch_done, step_done):
        # Atomic + backup, same contract as pretraining: a crash never corrupts the
        # only copy. Stores opt+step so --resume can continue mid-run.
        torch.save({"model": model.state_dict(), "model_config": mcfg.__dict__,
                    "chat_format": {"user_pre": USER_PRE, "asst_pre": ASST_PRE},
                    "opt": opt.state_dict(), "epoch": epoch_done, "step": step_done},
                   tmp_path)
        if out_path.exists():
            os.replace(out_path, bak_path)
        os.replace(tmp_path, out_path)

    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        rng.shuffle(order)  # advance rng each epoch so a resumed run keeps the schedule moving
        if epoch < start_epoch:
            continue
        for bi in range(0, len(order), args.batch_size):
            idxs = order[bi:bi + args.batch_size]
            X, Y = make_batch(examples, idxs, pad_id, device)
            lr = args.lr * min(1.0, (step + 1) / warmup)
            for g in opt.param_groups: g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with ctx:
                _, loss = model(X, Y)
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} "
                      f"lr {lr:.2e} {time.time()-t0:.0f}s", flush=True)
            step += 1
        save_sft(epoch + 1, step)
        print(f"saved {out_path} after epoch {epoch+1}", flush=True)
    print("SFT done.")


if __name__ == "__main__":
    main()
