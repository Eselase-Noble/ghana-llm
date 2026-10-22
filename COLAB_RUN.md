# Ghana-LLM — Colab run notes

Copy each block into a **Colab cell** and run top-to-bottom.
Runtime first: **Runtime → Change runtime type → T4 GPU → Save.**

Repo: `github.com/Eselase-Noble/ghana-llm` (public — clones with no token)
Dataset: `gh_v4`

---

## 0) One-time, on your Mac (before Colab)
Bundle the corpus and upload it to your Drive (code has NO data in it):
```
cd /Users/macbookpro/Projects/sensitive/ghana-llm
bash scripts/package_data.sh          # makes ghana-llm-data.tar.gz
```
Upload `ghana-llm-data.tar.gz` to Google Drive at **`MyDrive/ghana-llm/`**
(you can leave it as the .tar.gz — cell 4 unpacks it).

---

## 1) Check the GPU
```python
import torch, subprocess
print(subprocess.run(["nvidia-smi","-L"], capture_output=True, text=True).stdout)
print("CUDA:", torch.cuda.is_available(), "| torch", torch.__version__)
assert torch.cuda.is_available(), "No GPU! Runtime -> Change runtime type -> T4 GPU"
```

## 2) Pull the CODE from GitHub (public repo — no token)
```python
import os, shutil
if os.path.isdir("ghana-llm"): shutil.rmtree("ghana-llm")
!git clone --depth 1 https://github.com/Eselase-Noble/ghana-llm.git
%cd ghana-llm
!pip install -q -r requirements-colab.txt
```

## 3) Mount Drive + make the data available
```python
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/ghana-llm'
import os, subprocess
# unpack the bundle on Drive the first time (safe to re-run)
if not os.path.isdir(f'{DRIVE}/corpus'):
    subprocess.run(['tar','xzf',f'{DRIVE}/ghana-llm-data.tar.gz','-C',DRIVE], check=True)
# link the corpus into the repo tree
if os.path.islink('corpus') or os.path.isdir('corpus'):
    shutil.rmtree('corpus', ignore_errors=True) if os.path.isdir('corpus') and not os.path.islink('corpus') else os.remove('corpus') if os.path.islink('corpus') else None
os.symlink(f'{DRIVE}/corpus', 'corpus')
# checkpoints persist on Drive (survive Colab timeouts)
CKPT_DIR = f'{DRIVE}/checkpoints/gh_v4'; os.makedirs(CKPT_DIR, exist_ok=True)
assert os.path.exists('corpus/tokenized/gh_v4/train.bin'), "data not found — upload the bundle to Drive first"
print("data + checkpoints ready")
```

## 4) Train on the GPU  (RESUMABLE — just re-run this cell after a Colab timeout)
```python
!python src/train/train.py \
    --config configs/gpu.json \
    --dataset gh_v4 \
    --device cuda \
    --ckpt-dir "$CKPT_DIR" \
    --resume
```
Bigger model later (needs a much larger corpus): swap `configs/gpu.json` →
`configs/base_gpu.json`.

## 5) Sample from the model
```python
!python src/generate/sample.py --dataset gh_v4 --device cuda \
    --ckpt "$CKPT_DIR/ckpt.pt" --prompt "Ghana" --tokens 150 --temperature 0.8 --top-k 50
```

---

## (Optional) Grow the corpus ON Colab, then retrain
Runs the scrapers directly on Colab, rebuilds + re-tokenizes, saves back to Drive.
```python
# gather more data (Wikipedia + Gutenberg books)
!python src/scrape/black_history.py          # Ghana + Black-history Wikipedia
!python src/scrape/books.py --limit-per-query 15   # public-domain books
# rebuild clean text + re-tokenize a new dataset
!python src/data/build_corpus.py
!python src/data/prepare.py --dataset gh_v4 --glob '*.txt' --vocab-size 16000 --retrain-tokenizer
# persist the new corpus back to Drive so it survives the session
!cp -r corpus/clean corpus/tokenized "$DRIVE/corpus_v4_backup" 2>/dev/null; echo done
```
Then train with `--dataset gh_v4`.

---

## Notes
- **Data never goes in git** — only code. Corpus/checkpoints live on your Drive.
- Colab free sessions reset after idle/time limits → checkpoints on Drive + the
  `--resume` flag mean you lose nothing; just re-run the train cell.
- Current corpus is still small (proof-of-concept). Growing data (books/Wikipedia)
  is what moves the model toward actually knowing facts.
