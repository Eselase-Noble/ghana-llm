"""Generate text from a trained checkpoint. All local."""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from model.gpt import GPT, GPTConfig
from tokenizer.bpe import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gh_all")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--ckpt", default="", help="explicit checkpoint path (e.g. a Drive path)")
    args = ap.parse_args()

    if args.device != "auto":
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokd = ROOT / "corpus" / "tokenized" / args.dataset
    tok = BPETokenizer.load(tokd / "tokenizer")
    ckpt_path = args.ckpt or (ROOT / "checkpoints" / args.dataset / "ckpt.pt")
    ck = torch.load(ckpt_path, map_location=device)
    mc = ck["model_config"]
    model = GPT(GPTConfig(**mc)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    if args.prompt:
        ids = tok.encode(args.prompt)
    else:
        ids = [tok.special["<|endoftext|>"]]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    text = tok.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
