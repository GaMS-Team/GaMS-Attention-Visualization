"""
Run a (GaMS) chat model on a few prompts and store everything needed to
visualise how the model attends to the context.

For every prompt the script
  1. applies the chat template and greedily generates the answer,
  2. runs one teacher-forced forward pass over prompt + answer with
     eager attention, so that the attention probabilities of every layer and
     head are available,
  3. stores
       meta.json            tokens, segments, generated text, model info, per-head statistics
       attention_topk.npz   for every (layer, head, query) the top-k attended keys + weights
                            (head index n_heads holds the mean over all heads),
                            plus the exact attention mass on the BOS token
       logits_topk.npz      top-k next-token log-probs per position, log-prob of the
                            actual token and the entropy of the predictive distribution
       predictions.json     the same top-k predictions, decoded to strings
       attention_full.npy   (optional, --save_full_attention) full fp16 attention
                            tensor of shape [layers, heads, L, L]

Typical use (inside a GPU job, see run_visualization.sbatch):
    python3 extract_attention.py --model_path cjvt/GaMS3-12B-Instruct --output_dir outputs
"""

import json
import math
import os
import re
from argparse import ArgumentParser

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_DATA_PATH = "/shared/workspace/povejmo/scripts/test_chat_models/data.jsonl"
# 0-based line indices in data.jsonl:
#   4 -> "Prevedi naslednji seznam angleških besed v slovenščino ..." + "apple, mountain, river, sky, book"
#   5 -> "Izberi en dogodek iz slovenske zgodovine in ga podrobno predstavi."
DEFAULT_PROMPT_INDICES = [4, 5]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="cjvt/GaMS3-12B-Instruct",
                        help="HF locator or local path of the model.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
                        help="jsonl file with {'instruction', 'input'} examples.")
    parser.add_argument("--prompt_indices", type=int, nargs="*", default=DEFAULT_PROMPT_INDICES,
                        help="0-based line indices of the examples in --data_path to use.")
    parser.add_argument("--custom_prompt", type=str, action="append", default=[],
                        help="Additional raw user prompt (can be given multiple times).")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Results go to <output_dir>/<model_name>/prompt_<i>/.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_k_attention", type=int, default=32,
                        help="Number of strongest keys stored for every (layer, head, query).")
    parser.add_argument("--top_k_logits", type=int, default=10,
                        help="Number of top next-token predictions stored per position.")
    parser.add_argument("--save_full_attention", action="store_true",
                        help="Also store the full fp16 attention tensor (large: ~2 GB for 1k tokens on a 12B model).")
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------

def load_prompts(data_path, indices, custom_prompts):
    prompts = []
    if indices:
        with open(data_path, "r") as f:
            examples = [json.loads(line) for line in f if line.strip()]
        for idx in indices:
            example = examples[idx]
            user_message = example["instruction"]
            if example.get("input", "") != "":
                user_message += "\n" + example["input"]
            prompts.append({"source": f"{os.path.basename(data_path)}:{idx}", "user_message": user_message})
    for i, text in enumerate(custom_prompts):
        prompts.append({"source": f"custom:{i}", "user_message": text})
    return prompts


# --------------------------------------------------------------------------------------
# Token bookkeeping
# --------------------------------------------------------------------------------------

def token_segments(tokenizer, prompt_text, prompt_ids, user_message, n_total):
    """Label every token as special / template / user / response."""
    special_ids = set(tokenizer.all_special_ids)
    for t in ("<start_of_turn>", "<end_of_turn>"):
        tid = tokenizer.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            special_ids.add(tid)

    n_prompt = len(prompt_ids)
    segments = ["template"] * n_prompt + ["response"] * (n_total - n_prompt)

    # Locate the user message inside the rendered prompt through offset mapping.
    char_start = prompt_text.find(user_message)
    if char_start >= 0:
        try:
            enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
            if list(enc["input_ids"]) == list(prompt_ids):
                char_end = char_start + len(user_message)
                for i, (s, e) in enumerate(enc["offset_mapping"]):
                    if e > char_start and s < char_end:
                        segments[i] = "user"
            else:
                print("WARNING: re-tokenised prompt differs from chat-template ids; user span not marked.")
        except Exception as exc:  # slow tokenizers have no offsets
            print(f"WARNING: could not compute offsets ({exc}); user span not marked.")
    return segments, special_ids


def decode_tokens(tokenizer, ids):
    texts = [tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in ids]
    pieces = tokenizer.convert_ids_to_tokens(ids)
    return texts, pieces


# --------------------------------------------------------------------------------------
# Attention processing
# --------------------------------------------------------------------------------------

@torch.no_grad()
def process_attention(attentions, segments, top_k, full_out=None):
    """
    attentions: tuple(n_layers) of [1, H, L, L] tensors (attention probabilities).
    Returns top-k indices/values, exact BOS mass and per-head statistics.
    """
    n_layers = len(attentions)
    n_heads, L = attentions[0].shape[1], attentions[0].shape[-1]
    k = min(top_k, L)
    device = attentions[0].device

    idx_out = np.zeros((n_layers, n_heads + 1, L, k), dtype=np.uint16)
    val_out = np.zeros((n_layers, n_heads + 1, L, k), dtype=np.float16)
    bos_out = np.zeros((n_layers, n_heads + 1, L), dtype=np.float16)

    seg = np.array(segments)
    user_mask = torch.tensor(seg == "user", device=device)
    resp_mask = torch.tensor(seg == "response", device=device)
    prompt_mask = ~resp_mask
    q_pos = torch.arange(L, device=device)
    dist = (q_pos[:, None] - q_pos[None, :]).clamp(min=0).float()  # [L, L]
    rows = q_pos[1:]  # skip query 0 (it can only attend to itself)

    stat_names = ["bos", "self", "previous", "distance", "entropy", "to_user", "resp_to_prompt", "resp_to_resp"]
    stats = {name: np.zeros((n_layers, n_heads + 1), dtype=np.float32) for name in stat_names}

    for li, att in enumerate(attentions):
        a = att[0].float()                                    # [H, L, L]
        if full_out is not None:
            full_out[li] = a.to(torch.float16).cpu().numpy()
        a = torch.cat([a, a.mean(0, keepdim=True)], dim=0)    # [H+1, L, L]

        vals, idx = torch.topk(a, k=k, dim=-1, sorted=True)
        idx_out[li] = idx.to(torch.int32).cpu().numpy().astype(np.uint16)
        val_out[li] = vals.to(torch.float16).cpu().numpy()
        bos_out[li] = a[..., 0].to(torch.float16).cpu().numpy()

        ar = a[:, 1:, :]                                      # queries 1..L-1
        stats["bos"][li] = ar[..., 0].mean(-1).cpu().numpy()
        stats["self"][li] = a[:, rows, rows].mean(-1).cpu().numpy()
        stats["previous"][li] = a[:, rows, rows - 1].mean(-1).cpu().numpy()
        stats["distance"][li] = (ar * dist[1:]).sum(-1).mean(-1).cpu().numpy()
        stats["entropy"][li] = (-(ar * torch.log(ar.clamp_min(1e-12))).sum(-1)).mean(-1).cpu().numpy()
        if user_mask.any():
            stats["to_user"][li] = (ar * user_mask.float()).sum(-1).mean(-1).cpu().numpy()
        if resp_mask.any():
            ra = a[:, resp_mask, :]                           # response queries
            stats["resp_to_prompt"][li] = (ra * prompt_mask.float()).sum(-1).mean(-1).cpu().numpy()
            stats["resp_to_resp"][li] = (ra * resp_mask.float()).sum(-1).mean(-1).cpu().numpy()

    return idx_out, val_out, bos_out, {k_: v.tolist() for k_, v in stats.items()}


@torch.no_grad()
def process_logits(logits, ids, top_k, chunk=128):
    """logits: [L, V]. Position t predicts token t+1."""
    L = logits.shape[0]
    top_ids = np.zeros((L, top_k), dtype=np.int32)
    top_lp = np.zeros((L, top_k), dtype=np.float32)
    entropy = np.zeros(L, dtype=np.float32)
    target_lp = np.full(L, np.nan, dtype=np.float32)   # log p(token_t | tokens_<t)
    ids_t = torch.tensor(ids, device=logits.device)

    for s in range(0, L, chunk):
        e = min(L, s + chunk)
        lp = torch.log_softmax(logits[s:e].float(), dim=-1)
        v, i = torch.topk(lp, k=top_k, dim=-1)
        top_ids[s:e] = i.cpu().numpy()
        top_lp[s:e] = v.cpu().numpy()
        entropy[s:e] = (-(lp.exp() * lp).sum(-1)).cpu().numpy()
        nxt = torch.arange(s, e, device=logits.device) + 1
        valid = nxt < L
        if valid.any():
            rows = torch.arange(e - s, device=logits.device)[valid]
            target_lp[(nxt[valid]).cpu().numpy()] = lp[rows, ids_t[nxt[valid]]].cpu().numpy()
    return top_ids, top_lp, entropy, target_lp


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    args = parse_args()
    prompts = load_prompts(args.data_path, args.prompt_indices, args.custom_prompt)
    if not prompts:
        raise SystemExit("No prompts selected.")

    model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_path.rstrip("/").split("/")[-1])
    out_root = os.path.join(args.output_dir, model_name)
    os.makedirs(out_root, exist_ok=True)

    print(f"Loading {args.model_path} (eager attention, bf16) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",   # required to get attention probabilities
        device_map="cuda",
    )
    model.eval()
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    layer_types = list(getattr(cfg, "layer_types", None) or ["full_attention"] * n_layers)

    eos_ids = model.generation_config.eos_token_id
    eos_ids = list(eos_ids) if isinstance(eos_ids, (list, tuple)) else [eos_ids]
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot != tokenizer.unk_token_id and eot not in eos_ids:
        eos_ids.append(eot)

    for p_i, prompt in enumerate(prompts):
        print("=" * 80)
        print(f"[{p_i}] {prompt['source']}\n{prompt['user_message']}")
        messages = [{"role": "user", "content": prompt["user_message"]}]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        prompt_ids = input_ids[0].tolist()

        # 1) greedy generation
        with torch.no_grad():
            gen = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 eos_token_id=eos_ids)
        full_ids = gen[0].tolist()
        # drop trailing padding (if any) after the first EOS
        for j in range(len(prompt_ids), len(full_ids)):
            if full_ids[j] in eos_ids:
                full_ids = full_ids[: j + 1]
                break
        response_ids = full_ids[len(prompt_ids):]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        L = len(full_ids)
        print(f"--- response ({len(response_ids)} tokens, total sequence {L}) ---\n{response_text}\n")

        # 2) teacher-forced forward pass with attentions
        with torch.no_grad():
            out = model(input_ids=torch.tensor([full_ids], device=model.device),
                        output_attentions=True, use_cache=False)
        if out.attentions is None or len(out.attentions) == 0:
            raise RuntimeError("Model did not return attentions; is attn_implementation='eager' supported?")

        texts, pieces = decode_tokens(tokenizer, full_ids)
        segments, special_ids = token_segments(tokenizer, prompt_text, prompt_ids, prompt["user_message"], L)
        for i, tid in enumerate(full_ids):
            if tid in special_ids:
                segments[i] = "special"

        p_dir = os.path.join(out_root, f"prompt_{p_i}")
        os.makedirs(p_dir, exist_ok=True)

        full_att = None
        if args.save_full_attention:
            full_att = np.lib.format.open_memmap(os.path.join(p_dir, "attention_full.npy"), mode="w+",
                                                 dtype=np.float16, shape=(n_layers, n_heads, L, L))
        idx, val, bos, head_stats = process_attention(out.attentions, segments, args.top_k_attention, full_att)
        if full_att is not None:
            full_att.flush()
            del full_att

        logits = out.logits[0]
        del out
        top_ids, top_lp, entropy, target_lp = process_logits(logits, full_ids, args.top_k_logits)
        del logits
        torch.cuda.empty_cache()

        np.savez_compressed(os.path.join(p_dir, "attention_topk.npz"), idx=idx, val=val, bos=bos)
        np.savez_compressed(os.path.join(p_dir, "logits_topk.npz"), top_ids=top_ids, top_logprobs=top_lp,
                            entropy=entropy, target_logprob=target_lp, token_ids=np.array(full_ids, dtype=np.int32))

        # decoded predictions (position t -> distribution over token t+1)
        vocab_cache = {}

        def tok_text(i):
            if i not in vocab_cache:
                vocab_cache[i] = tokenizer.decode([int(i)], skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=False)
            return vocab_cache[i]

        predictions = [[[tok_text(int(t)), round(float(math.exp(lp)), 5)] for t, lp in zip(top_ids[pos], top_lp[pos])]
                       for pos in range(L)]
        with open(os.path.join(p_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f, ensure_ascii=False)

        meta = {
            "model": args.model_path,
            "source": prompt["source"],
            "user_message": prompt["user_message"],
            "prompt_text": prompt_text,
            "response_text": response_text,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "layer_types": layer_types,
            "prompt_len": len(prompt_ids),
            "seq_len": L,
            "top_k_attention": int(idx.shape[-1]),
            "token_ids": full_ids,
            "tokens": texts,
            "pieces": pieces,
            "segments": segments,
            "head_stats": head_stats,
        }
        with open(os.path.join(p_dir, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print(f"Saved to {p_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
