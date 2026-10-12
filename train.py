"""Train the WordFlow GPT model on a UTF-8 text file."""

import argparse
import math
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F

from generation import CONFIG, DEFAULT_CHECKPOINT, ROOT, choose_device
from model import GPTModel, create_dataloader_v1


def loss_batch(model, batch, device):
    x, y = (tensor.to(device) for tensor in batch)
    logits = model(x)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())


@torch.no_grad()
def evaluate(model, loader, device, batches=5):
    model.eval()
    losses = [loss_batch(model, batch, device).item()
              for batch in islice(loader, batches)]
    return sum(losses) / len(losses) if losses else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "the-verdict.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume", action="store_true", help="Start with checkpoint weights")
    parser.add_argument("--save-optimizer", action="store_true",
                        help="Also save optimizer state for exact training continuation (large file)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.eval_every < 1:
        parser.error("epochs, batch-size, and eval-every must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")

    torch.manual_seed(123)
    device = choose_device(args.device)
    text = args.data.read_text(encoding="utf-8")
    split = int(len(text) * 0.9)
    train = create_dataloader_v1(text[:split], batch_size=args.batch_size,
        max_length=CONFIG["context_length"], stride=CONFIG["context_length"],
        shuffle=True, drop_last=True)
    val = create_dataloader_v1(text[split:], batch_size=args.batch_size,
        max_length=CONFIG["context_length"], stride=CONFIG["context_length"],
        shuffle=False, drop_last=False)
    if not train or not val:
        parser.error("Dataset is too short for 256-token training and validation windows")

    model = GPTModel(CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    if args.resume:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state))
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
    model.to(device)
    print(f"Training on {device}; {len(train)} train batches, {len(val)} validation batches", flush=True)
    weights_path = args.checkpoint
    if weights_path.stem.endswith("_resume"):
        weights_path = weights_path.with_name(
            weights_path.stem.removesuffix("_resume") + weights_path.suffix
        )
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_batch(model, batch, device)
            loss.backward()
            optimizer.step()
            step += 1
            if step % args.eval_every == 0:
                train_loss = evaluate(model, train, device)
                val_loss = evaluate(model, val, device)
                print(f"epoch {epoch} step {step}: train {train_loss:.3f} "
                      f"val {val_loss:.3f} perplexity {math.exp(val_loss):.1f}", flush=True)
                model.train()
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        if args.save_optimizer:
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       weights_path.with_name(weights_path.stem + "_resume.pth"))
        print(f"Saved {weights_path} after epoch {epoch}", flush=True)
    print("Training complete", flush=True)


if __name__ == "__main__":
    main()
