# Milestone 1: persistent local Qwen chat

The model loads from this project, stays resident across chat turns, and runs
without Hub access after setup. This is **inference only**: all parameters are
frozen and the automatic learning loop is still pending.

## Identity and storage

- Model: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), post-trained.
- Revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- Loader: `Qwen3_5ForCausalLM`, text only, BF16 on the RTX 5070 Ti.
- Loaded text parameters: **752,393,024**. No missing, mismatched, or unexpected
  checkpoint keys were reported by this loader.
- Twelve downloaded files total about **1.65 GiB** under `models/hub/`.
  [Download manifest](download-manifest.json) records per-file sizes and SHA-256.
- Python 3.11.16, PyTorch 2.11.0+cu128, Transformers 5.16.1, Hub 1.30.0.
  `pyproject.toml`, `.python-version`, and `uv.lock` pin the environment.

Model storage and temporary file paths are configured before model-library imports.
Downloaded file targets resolve inside this project. Chat uses `local_files_only`
and a revision-matched local manifest. No global model cache is needed.
Setup was also rerun successfully with Hub offline mode enabled from the parent
directory, reusing the downloaded files.

## Measurements

The successful [GPU checks](chat-checks.json) recorded:

| Measurement | Observed value |
| --- | ---: |
| Model loading, after imports, from cached files | 1.016 seconds |
| Allocated GPU memory after loading | 1,459.10 MiB |
| Peak allocated GPU memory during two-turn probe | 1,502.69 MiB |
| First reply, 33 generated tokens including end token | 1.271 seconds |
| Second reply, 2 generated tokens including end token | 0.141 seconds |

Timing includes prompt processing and generation with explicit CUDA synchronization.
These are a few observations on a shared desktop, not a throughput benchmark or a
latency guarantee. PyTorch allocated memory excludes other applications, CUDA
driver overhead, and allocator-reserved memory. The load time excludes imports
and download time. Full-attention layers explicitly use SDPA math; no FlashAttention
or optimized DeltaNet-kernel performance claim is made.

## Checks that passed

- Actual offline GPU loading with all text parameters in BF16 and frozen.
- Two messages reuse one model object, retaining previous conversation text.
- `/reset` removes previous context without loading another model.
- Old complete turns are omitted to fit the configured context budget.
- An oversized new message is rejected without changing stored conversation state.
- Parameter version counters remain unchanged and no gradients are created.
- The actual [chat.bat](../chat.bat) launcher runs from the parent directory,
  accepts two English messages, resets, accepts Japanese, and exits cleanly.
  Its transcript has exactly one start/load and three chat turns.
  See [launcher evidence and source hashes](launcher-checks.json).
- `setup.bat` can reuse the downloaded model with Hub access disabled.

The probe stated a project name, then asked for it. The model answered `Atlas`.
That is **conversation-context recall**, not evidence of weight learning. Another
probe produced a Japanese greeting. The saved samples also show limitations:
one reply ignored an instruction to answer only `OK`; another invented a project
description that the prompt never supplied. No broad quality claim is warranted.

## Reproduce

Double-click `chat.bat` for interactive use. From the repository root, run:

```powershell
.venv\Scripts\python.exe -X utf8 -m qwenbrain.check_chat
```

The check module sets Hub offline mode before importing the model stack and writes
`reports/chat-checks.json`. Private chat transcripts are in `data/conversations/`
and ignored by Git. The tracked check reports contain synthetic test messages only.

Next milestone: attach LoRA, mask loss onto each new user message's content,
and prove an adapter update changes weights while the original Qwen weights remain
unchanged. Only after that check should the automatic per-message loop be connected.
