# Visualize GaMS attention

Runs a chat model on prompts from `../test_chat_models/data.jsonl` (default: lines 4 and 5, the
word-translation prompt and the Slovenian-history prompt). It stores attention and logits data
and builds an interactive, self-contained HTML viewer in the style of BertViz.

```bash
sbatch run_visualization.sbatch                                   # cjvt/GaMS3-12B-Instruct, 1024 new tokens
sbatch run_visualization.sbatch /models/GaMS3.1-Tool-Test 512     # any local / HF model
```

Output: `outputs/<model>/attention_visualization.html` (open it in a browser) and `outputs/<model>/prompt_<i>/`:

| file | content |
|---|---|
| `meta.json` | tokens, pieces, ids, segments (special/template/user/response), generated text, per-head statistics |
| `attention_topk.npz` | `idx`/`val` `[layers, heads+1, L, top_k]` strongest keys per query (head `heads` is the mean over heads), `bos` `[layers, heads+1, L]` exact BOS mass |
| `logits_topk.npz` | top-k next-token ids/log-probs, entropy, log-prob of each actual token |
| `predictions.json` | the same top-k predictions, decoded |
| `attention_full.npy` | only with `--save_full_attention`: full fp16 `[layers, heads, L, L]` |

Steps (both run inside the job; the second needs only numpy):
1. `extract_attention.py`: greedy generation, then one teacher-forced forward pass with `attn_implementation="eager"` and `output_attentions=True`.
2. `build_visualization.py`: sparsifies (`--min_weight`, `--max_per_row`), gzips the data per layer and embeds it in `viewer_template.html`. Use `--layers 0 6 12 ...` to make the file smaller.

Viewer:
- **Lines**: BertViz-style. Hover a token on the right to see what it attends to; hover a token on the left to see which tokens attend to it. Head colours match the head buttons.
- **Text**: heat-map over the running text. Works better for long answers. It can also colour each token by its probability or by the model's entropy.
- **Matrix**: the full query × key heat-map for the selected layer and heads.
- **Head overview**: a layers × heads heat-map of a statistic, such as previous-token heads, BOS sink or answer→prompt attention. Click a cell to jump to that layer and head.
- **Scale** (linear / √ / log, √ by default) sets how weights map to colour intensity. The legend above each view shows the mapping. Within one layer and head, attention really is sparse; √ and log only make the small weights visible.
- **Hide BOS sink**: Gemma models put most of their attention on `<bos>`. This hides that token and renormalizes the remaining weights.

Note: attention rollout (Abnar & Zuidema 2020) was tried and deliberately left out. With causal attention, the first tokens can only attend to themselves, so the product over layers puts about 95% of the mass on the first token kept, whichever token is queried. For "which earlier tokens did this token use", gradient-based attribution or value-weighted methods (ALTI) fit decoders better.
