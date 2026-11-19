"""
Build a self-contained interactive HTML attention viewer from the files written by
extract_attention.py.

    python3 build_visualization.py --input_dir outputs/GaMS3-12B-Instruct

The HTML contains, per prompt and layer, a sparse version of the attention
(for every head + the head mean, and every query token, the strongest keys).
Layers are gzip-compressed and decoded lazily in the browser. Only numpy and
the standard library are needed, so this can run on any machine.
"""

import base64
import glob
import gzip
import json
import os
from argparse import ArgumentParser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with prompt_*/ sub-directories (i.e. <output_dir>/<model_name>).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: <input_dir>/attention_visualization.html).")
    parser.add_argument("--min_weight", type=float, default=0.01,
                        help="Drop attention weights below this value. The threshold is applied relative to the "
                             "non-BOS mass, so weights stay visible when the BOS sink is hidden in the viewer.")
    parser.add_argument("--max_per_row", type=int, default=20,
                        help="Keep at most this many keys per (layer, head, query).")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Subset of layer indices to include (default: all). Useful to shrink the HTML.")
    parser.add_argument("--max_predictions", type=int, default=10,
                        help="Number of next-token alternatives shown per position.")
    return parser.parse_args()


def encode_layer(idx, val, bos, min_weight, max_per_row):
    """
    idx, val: [H+1, L, K] top-k keys/weights (sorted desc), bos: [H+1, L] exact BOS mass.
    Returns a dict with the gzip+base64 blob and array lengths.
    Blob layout: offsets uint32[(H+1)*L+1] | bos uint16[(H+1)*L] | keys uint16[nnz] | weights uint8[nnz]
    Weights are stored as round(sqrt(w) * 255) for better resolution of small values.
    """
    K = min(max_per_row, idx.shape[-1])
    idx = idx[..., :K]
    v = val[..., :K].astype(np.float32)
    b = bos.astype(np.float32)
    thr = (min_weight * np.clip(1.0 - b, 0.0, 1.0))[..., None]
    mask = (v >= thr) & (v > 0)
    counts = mask.reshape(-1, K).sum(1)
    off = np.zeros(counts.size + 1, dtype=np.uint32)
    off[1:] = np.cumsum(counts)
    keys = idx[mask].astype(np.uint16)
    w = np.clip(np.round(np.sqrt(np.clip(v[mask], 0, 1)) * 255), 0, 255).astype(np.uint8)
    bos_q = np.clip(np.round(b.reshape(-1) * 65535), 0, 65535).astype(np.uint16)
    raw = off.tobytes() + bos_q.tobytes() + keys.tobytes() + w.tobytes()
    blob = gzip.compress(raw, compresslevel=9, mtime=0)
    return {
        "n_off": int(off.size), "n_bos": int(bos_q.size), "nnz": int(keys.size),
        "b64": base64.b64encode(blob).decode("ascii"),
    }, len(raw), len(blob)


def short_title(text, n=70):
    first = text.strip().split("\n")[0]
    return first if len(first) <= n else first[: n - 1] + "…"


def load_prompt(p_dir, args):
    with open(os.path.join(p_dir, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(p_dir, "predictions.json")) as f:
        predictions = json.load(f)
    att = np.load(os.path.join(p_dir, "attention_topk.npz"))
    lg = np.load(os.path.join(p_dir, "logits_topk.npz"))
    idx, val, bos = att["idx"], att["val"], att["bos"]

    n_layers = meta["n_layers"]
    layer_ids = list(range(n_layers)) if not args.layers else [l for l in args.layers if 0 <= l < n_layers]
    layers, raw_total, gz_total = [], 0, 0
    for l in layer_ids:
        enc, raw_n, gz_n = encode_layer(idx[l], val[l], bos[l], args.min_weight, args.max_per_row)
        layers.append(enc)
        raw_total += raw_n
        gz_total += gz_n

    target_prob = [None if np.isnan(x) else round(float(np.exp(x)), 5) for x in lg["target_logprob"]]
    head_stats = {k: [v[l] for l in layer_ids] for k, v in meta["head_stats"].items()}

    print(f"  {os.path.basename(p_dir)}: L={meta['seq_len']}, layers={len(layer_ids)}, "
          f"sparse data {raw_total / 1e6:.1f} MB -> gzip {gz_total / 1e6:.1f} MB")

    return {
        "title": short_title(meta["user_message"]),
        "source": meta["source"],
        "user_message": meta["user_message"],
        "response_text": meta["response_text"],
        "n_layers": n_layers,
        "n_heads": meta["n_heads"],
        "layer_types": meta["layer_types"],
        "layer_ids": layer_ids,
        "prompt_len": meta["prompt_len"],
        "seq_len": meta["seq_len"],
        "tokens": meta["tokens"],
        "pieces": meta["pieces"],
        "token_ids": meta["token_ids"],
        "segments": meta["segments"],
        "target_prob": target_prob,
        "entropy": [round(float(x), 4) for x in lg["entropy"]],
        "predictions": [p[: args.max_predictions] for p in predictions],
        "head_stats": head_stats,
        "layers": layers,
    }


def main():
    args = parse_args()
    p_dirs = sorted(glob.glob(os.path.join(args.input_dir, "prompt_*")),
                    key=lambda p: int(p.rsplit("_", 1)[-1]))
    if not p_dirs:
        raise SystemExit(f"No prompt_* directories in {args.input_dir}")

    print(f"Building viewer from {len(p_dirs)} prompt(s) ...")
    prompts = [load_prompt(p, args) for p in p_dirs]
    with open(os.path.join(p_dirs[0], "meta.json")) as f:
        model = json.load(f)["model"]

    data = {"model": model, "min_weight": args.min_weight, "prompts": prompts}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    with open(os.path.join(HERE, "viewer_template.html")) as f:
        template = f.read()
    html = template.replace("__DATA_JSON__", payload)

    out = args.output or os.path.join(args.input_dir, "attention_visualization.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"Wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
