"""Command-line text generation with the bundled WordFlow model."""

import argparse

from generation import DEFAULT_CHECKPOINT, generate, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Every effort moves you")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=25, help="Use 0 for no top-k filter")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    model, device = load_model(args.checkpoint, args.device)
    result = generate(model, args.prompt, args.tokens, args.temperature,
                      args.top_k or None, device, args.seed)
    print(result["text"])


if __name__ == "__main__":
    main()
