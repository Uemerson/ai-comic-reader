# AI Comic Reader

A **panel-first** reader for comics, manga and webtoons. It finds the panels
in every page, reads the text inside each panel, summarizes the chapter with
an LLM, and builds an HTML gallery with the panels that **match the summary**.

Everything in the codebase (identifiers, comments, docstrings, CLI help,
terminal messages and prompts) is written in **English**. Only the AI reply
and the generated HTML page follow the language chosen at runtime.

---

## Features

- **Walks the folder tree** of the chosen kind (`mangas/` or `webtoons/`) and
  finds every chapter folder automatically, at any depth
  (e.g. `mangas/one-piece/1`).
- **Panel-first pipeline** per reading kind:
  - *manga* → panel segmentation with **YOLO26s-seg** (ShadowB);
  - *webtoon* → whitespace/**gutter** splitting.
- **Text detection + OCR** with **RT-DETR-v2** and **PP-OCRv5**, run *inside*
  each panel so the text is guaranteed to belong to the right panel.
- **Correct reading order**:
  - manga: rows top→bottom, right→left;
  - webtoon: rows top→bottom, left→right.
- **Two-stage LLM summarization**:
  1. the LLM writes the chapter **summary**;
  2. the LLM picks **only the panels that show what the summary says**, in
     the order of the events, quoting the summary excerpt each panel
     illustrates and the reason.
- **No limit** on the number of main panels.
- **HTML output** with the summary + a zoomable gallery (arrow keys `←`/`→`
  to navigate, `Esc` to close) and clickable images in full screen.
- **Multilingual output**: `pt-br`, `english`, `korean`, `japanese`, `chinese`
  (default: `english`) — drives both the AI reply and the HTML UI.
- **Pluggable LLM backend**:
  - `local` — [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
    with a GGUF model (default: `Qwen/Qwen3-4B-GGUF`, Q4_K_M);
  - `opencode` — the [OpenCode](https://opencode.ai) CLI (`opencode run`),
    with cost/token reporting per session.
- **Automatic GPU usage** when CUDA is available (detection/OCR and, if
  `llama-cpp-python` was built with CUDA, the LLM too).

---

## How it works

```
chapter folder ──► panel detection ──► text detection (RT-DETR-v2)
      │                                          │
      │                                          ▼
      │                              OCR (PP-OCRv5, per panel)
      │                                          │
      ▼                                          ▼
 annotated images + cropped panels ────► chapter texts
                                                 │
                             ┌───────────────────┴───────────────────┐
                             ▼                                       ▼
                  1st LLM call: summary          2nd LLM call: panels that match
                             │                                       │
                             └───────────────┬───────────────────────┘
                                             ▼
                             summary.html + summary.txt (+ browser)
```

The output is written next to the chapter folder, with the `-ai` suffix
(e.g. `mangas/one-piece/1` → `mangas/one-piece/1-ai`), containing:

- the annotated page images;
- `panels/` with the cropped panel images;
- `_chapter_texts.txt` with the extracted texts;
- `summary.txt` and `summary.html`.

---

## Project structure

```
ai-comic-reader/
├── main.py            # the whole application (single file)
├── requirements.txt   # core dependencies
├── LICENSE            # MIT license
├── .pylintrc          # pylint configuration
├── .gitignore
├── mangas/            # manga chapters — git-ignored
└── webtoons/          # webtoon chapters — git-ignored
```

---

## Requirements

- Python 3.9+
- Core packages (see [`requirements.txt`](requirements.txt)): `torch`,
  `transformers`, `ultralytics`, `onnxruntime`, `opencv-python`, `pillow`,
  `numpy`, `huggingface_hub`
- Optional, for the local backend: `llama-cpp-python`
- Optional, for the OpenCode backend: the `opencode` binary on `PATH`
- Optional, for CUDA acceleration: install `onnxruntime-gpu` instead of
  `onnxruntime` and a CUDA build of PyTorch

---

## Install

```bash
# clone the repository
git clone git@github.com:Uemerson/ai-comic-reader.git
cd ai-comic-reader

# core dependencies
pip install -r requirements.txt

# optional: local GGUF LLM backend
pip install llama-cpp-python

# optional: OpenCode backend — have the `opencode` binary on your PATH
```

The detection/OCR models are downloaded automatically from Hugging Face on
the first run and cached under `~/.cache/comic_ocr` (override with the
`COMIC_OCR_MODELS_DIR` environment variable).

---

## Usage

```bash
# scan mangas/ (whole tree), summarize and open the result
python main.py

# scan webtoons/ instead
python main.py --kind webtoon

# AI reply + HTML in Brazilian Portuguese
python main.py --language pt-br

# summarize through the OpenCode CLI
python main.py --llm opencode \
    --opencode-model opencode-go/deepseek-v4.1-flash

# local GGUF backend
python main.py --llm local --model Qwen/Qwen3-8B-GGUF

# force the GPU / limit the images
python main.py --device cuda
python main.py --max 0
```

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--kind {manga,webtoon}` | `manga` | reading kind; defines which folder is scanned (`manga` → `mangas/`, `webtoon` → `webtoons/`) |
| `--language {pt-br,english,korean,japanese,chinese}` | `english` | language of the AI reply and of the generated HTML page |
| `--llm {local,opencode}` | `local` | LLM backend |
| `--model` | `Qwen/Qwen3-4B-GGUF` | GGUF repo on Hugging Face (local backend) |
| `--model-file` | *(auto)* | specific `.gguf` file (local backend) |
| `--opencode-model` | `opencode-go/deepseek-v4.1-flash` | OpenCode model (`provider/model[#variant]`) |
| `--opencode-bin` | `opencode` | OpenCode binary |
| `--opencode-agent` | *(none)* | OpenCode agent |
| `--device` | *(auto)* | `cpu` or `cuda`; otherwise detected automatically |
| `--pad` | `8` | panel padding in pixels |
| `--max` | `0` | `0` = every image; otherwise at most `N` |

---

## Notes

- With the `opencode` backend, the cost and token usage of the session are
  printed in the terminal, saved in `summary.txt` and shown in the HTML
  footer.
- The two LLM calls run in the **same** OpenCode session, so the reported
  cost/tokens cover both stages.

---

## License

This project is licensed under the **MIT License** — see the
[LICENSE](LICENSE) file for details.
