"""
Chat with the SFT model: prompt -> answer.

Formats the prompt with the chat markers, generates until <|endoftext|>,
and prints only the assistant's response. Single-shot (--prompt) or interactive.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from model.gpt import GPT, GPTConfig
from tokenizer.bpe import BPETokenizer


def pick_device(x):
    if x and x != "auto": return x
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"


@torch.no_grad()
def answer(model, tok, device, user_pre, asst_pre, prompt, max_new=200, temperature=0.7, top_k=40):
    eot = tok.special["<|endoftext|>"]
    ids = tok.encode(user_pre + prompt + asst_pre)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    block = model.cfg.block_size
    out = ids[:]
    for _ in range(max_new):
        cond = torch.tensor([out[-block:]], dtype=torch.long, device=device)
        logits, _ = model(cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        if nxt == eot:
            break
        out.append(nxt)
    return tok.decode(out[len(ids):]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gh_v4")
    ap.add_argument("--ckpt", default="", help="SFT checkpoint (default checkpoints/<dataset>/sft.pt)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = BPETokenizer.load(ROOT / "corpus" / "tokenized" / args.dataset / "tokenizer")
    ckpt = args.ckpt or (ROOT / "checkpoints" / args.dataset / "sft.pt")
    ck = torch.load(ckpt, map_location=device)
    model = GPT(GPTConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    fmt = ck.get("chat_format", {"user_pre": "<|user|>\n", "asst_pre": "\n<|assistant|>\n"})

    def reply(p):
        return answer(model, tok, device, fmt["user_pre"], fmt["asst_pre"], p,
                      args.max_new, args.temperature, args.top_k)

    if args.prompt:
        print(reply(args.prompt))
        return
    print("Ghana-LLM chat. Type a prompt (Ctrl-C to quit).")
    while True:
        try:
            p = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if p:
            print("bot>", reply(p))


if __name__ == "__main__":
    main()
