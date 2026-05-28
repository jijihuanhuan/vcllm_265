import math
import torch
from tqdm import tqdm


def compute_perplexity(model, tokenizer, text, max_length=512, stride=128):
    """
    Sliding-window perplexity over tokenized text. Pass ``text`` as a list of strings;
    long documents should be one string so the full sequence is windowed (no tokenizer truncation).
    """
    model.eval()
    device = next(model.parameters()).device

    if isinstance(text, str):
        text = [text]

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for chunk in text:
            tokenized = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=False,
            )
            input_ids = tokenized.input_ids
            if input_ids.size(1) < 2:
                continue

            seq_len = input_ids.size(1)
            for i in tqdm(
                range(0, seq_len, stride),
                desc="PPL windows",
                leave=False,
                disable=seq_len < max_length * 2,
            ):
                end = min(i + max_length, seq_len)
                begin = max(0, end - max_length)
                if end - begin < 2:
                    continue

                inputs = input_ids[:, begin:end].to(device)
                labels = inputs.clone()
                labels[:, :-stride] = -100

                outputs = model(inputs, labels=labels)
                loss = outputs.loss

                n_tok = end - begin - stride
                if n_tok <= 0:
                    continue
                total_loss += loss.item() * n_tok
                total_tokens += n_tok

    if total_tokens == 0:
        return float("inf")

    ppl = math.exp(total_loss / total_tokens)
    return ppl


def load_wikitext_data(file_path, max_chars=None):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()
    if max_chars is not None:
        raw = raw[:max_chars]

    paragraphs = raw.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def load_wikitext2_test_full():
    """HuggingFace WikiText-2 raw test split (full)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    parts = [str(t).strip() for t in ds["text"] if t is not None and str(t).strip()]
    return parts


def evaluate_perplexity_on_wikitext(
    model,
    tokenizer,
    data_path="archive/test.txt",
    *,
    use_hf_wikitext2_test=True,
):
    """
    Default: full WikiText-2 **test** via ``datasets`` (HF).
    Fallback: local ``archive/test.txt`` if HF load fails.
    """
    if use_hf_wikitext2_test:
        try:
            print("Loading WikiText-2 (test) from HuggingFace datasets...")
            paragraphs = load_wikitext2_test_full()
            print(f"Loaded {len(paragraphs)} non-empty segments from wikitext-2-raw-v1 test")
        except Exception as e:
            print(f"HuggingFace WikiText-2 test load failed ({e}); using file fallback.")
            paragraphs = load_wikitext_data(data_path, max_chars=None)
            print(f"Loaded {len(paragraphs)} paragraphs from {data_path}")
    else:
        print(f"Loading WikiText data from {data_path}...")
        paragraphs = load_wikitext_data(data_path, max_chars=None)
        print(f"Loaded {len(paragraphs)} paragraphs")

    full_text = "\n\n".join(paragraphs)
    print(f"Joined corpus length: {len(full_text)} characters")

    print("Computing perplexity...")
    ppl = compute_perplexity(model, tokenizer, [full_text])

    print(f"Perplexity: {ppl:.2f}")
    return ppl
