# Repeated identity answer: reproduction and fix

The sequence `hi` followed by `im chris, who are you` produced a repeated identity
sentence until the 256-token output limit. With seed 42 and the original config,
the actual GPU chat path reproduced the user's output exactly, twice.
Its most frequent six-word span occurred 14 times.

Running only the second message from a fresh conversation ended normally. The
two-turn sequence is therefore retained as the regression scenario. Evidence:
[original failure](../../reports/repetition-before.json),
[single-turn minimization](../../reports/repetition-single-before.json).

## Diagnosis

Ranked hypotheses were missing repetition control, overly narrow temperature/top-p
sampling, and cached-decoding/end-token behavior. Adding **only** a presence
penalty of 2.0 resolved the observed loop. Temperature 0.7, top-p 0.8, top-k 20,
model weights, prompt formatting, cache implementation, and end-token handling
remain unchanged. The response now ends naturally at the assistant end token;
it is not cut short by a repetition detector or a lower output limit.

Qwen's [model-card sampling guidance](https://huggingface.co/Qwen/Qwen3.5-0.8B)
includes presence penalty 2.0 for non-thinking text generation. The installed
Transformers generation API does not provide that parameter directly, so
`qwenbrain/sampling.py` supplies a small logits processor. This fix adopts the
penalty, not the model card's entire sampling configuration.

At each decoding step, the processor subtracts 2.0 from each token logit if that
token already appeared in the **current generated reply**. Prompt/history tokens
are excluded. Each token is penalized once regardless of its repetition count.
Tokens remain possible; this is neither a hard n-gram ban nor a frequency penalty.

## Verification

```powershell
.venv\Scripts\python.exe -X utf8 -m qwenbrain.check_repetition --seeds 42 43 44 45 46
```

- [All five seeds passed](../../reports/repetition-checks.json): no six-word span repeated three
  times and no reply hit the output limit in the reported two-turn scenario.
- The seed-42 identity reply changed from **256 tokens with 14 repeated spans**
  to **55 tokens with no repeated six-word span**, ending naturally.
- Numerical processor checks cover prompt exclusion, positive/negative scores,
  repeated token IDs, independent batches, and leaving original scores unchanged.
- The [actual batch launcher](../../reports/repetition-launcher-checks.json) passed the user's
  two messages, `/reset`, a Japanese greeting, and clean exit. It loaded the model
  once, using local files with Hub offline mode enabled.

This is a verified fix for the reproduced sampling loop, not a guarantee that the
small model will never repeat or give incorrect answers. Presence penalties change
word choice and may affect other tasks. Weight learning remains OFF.

Existing chat processes retain the old settings. Run `/quit`, then reopen
`chat.bat` to use the fix; `/reset` alone does not reload configuration.
