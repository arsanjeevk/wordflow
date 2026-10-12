"""WordFlow tokenization and temperature/top-k decoding."""

from pathlib import Path

import tiktoken
import torch

from model import GPTModel


CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "verdict_model.pth"


def choose_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint=DEFAULT_CHECKPOINT, device="auto"):
    device = choose_device(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model = GPTModel(CONFIG)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, device


def generate(model, prompt, max_new_tokens=30, temperature=0.8, top_k=25,
             device="cpu", seed=None):
    """Return prompt plus generated text using temperature and top-k sampling."""
    if not prompt.strip():
        raise ValueError("Enter a prompt.")
    if not 1 <= max_new_tokens <= 100:
        raise ValueError("New tokens must be between 1 and 100.")
    if not 0 <= temperature <= 2:
        raise ValueError("Temperature must be between 0 and 2.")
    if top_k is not None and not 1 <= top_k <= CONFIG["vocab_size"]:
        raise ValueError("Top-k must be between 1 and 50257.")
    if seed is not None:
        torch.manual_seed(seed)

    tokenizer = tiktoken.get_encoding("gpt2")
    ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    if not ids:
        raise ValueError("Enter a prompt with at least one token.")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(idx[:, -CONFIG["context_length"]:])[:, -1, :]
            if top_k is not None:
                top_values, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < top_values[:, -1:], float("-inf"))
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            if next_id.item() == 50256:
                break
            idx = torch.cat((idx, next_id), dim=1)
    new_ids = idx[0, len(ids):].tolist()
    return {
        "text": tokenizer.decode(idx[0].tolist()),
        "continuation": tokenizer.decode(new_ids),
        "prompt_tokens": len(ids),
        "generated_tokens": len(new_ids),
    }
