# WordFlow

WordFlow is a local text-generation project. It trains a decoder-only Transformer on a text file, saves the learned weights, and serves a browser interface where you can give the model a prompt and generate a continuation. The interface exposes the decoding controls used during inference: output length, temperature, top-k, and an optional random seed.

The bundled checkpoint was trained on `the-verdict.txt`, so you can try the interface without training first. The project follows the GPT construction and pretraining approach in Sebastian Raschka's *Build a Large Language Model (From Scratch)*. `model.py` retains the attribution header from the book's companion code.

## What is included

```text
wordflow/
├── README.md                 Project guide
├── RESULTS.md                Training measurements
├── requirements.txt          Python dependencies
├── the-verdict.txt           Exact text used for the bundled training run
├── model.py                  GPT architecture and sliding-window data loader
├── train.py                  Training, validation, and checkpoint saving
├── generation.py             Checkpoint loading and token sampling
├── generate.py               Command-line generation
├── app.py                    Local HTTP server and generation API
├── checkpoints/
│   └── verdict_model.pth     Trained model weights
└── static/                   Browser interface (HTML, CSS, JavaScript)
```

The dataset is 20,479 UTF-8 bytes (5,145 GPT-2 tokens). The checkpoint is approximately 623 MiB. Both are kept in this folder, so the original notebook and PDF are not needed to run WordFlow.

The checkpoint is stored with Git LFS on GitHub. Install Git LFS before cloning if you want the trained weights downloaded automatically; after cloning, run `git lfs pull` if the checkpoint is only a small pointer file.

## Run the web interface

Use Python 3.10 or newer. From the `wordflow` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8000>. The server binds to your own computer (`127.0.0.1`). It loads the checkpoint when it starts, so restart the server after replacing or retraining the checkpoint. CPU inference works, although longer generations can take time. PyTorch can use a supported CUDA or MPS device automatically.

### Generation controls

| Control | Effect |
| --- | --- |
| Prompt | Starting text. The model reads at most the last 256 tokens at each step. |
| New tokens | Number of tokens to add, from 1 to 100 in the interface. Tokens are pieces of text, not necessarily whole words. |
| Temperature | At `0`, always choose the highest-scoring next token. Higher values sample more varied tokens and can make the result less coherent. |
| Top-k | Restricts sampling to the `k` highest-scoring candidate tokens at each step. A smaller value makes output more constrained. |
| Seed | Repeats a sampling sequence when the model, prompt, settings, and environment are the same. |

The **Copy text** button copies the prompt and generated continuation.

## Generate from a terminal

```bash
python generate.py "Every effort moves you" --tokens 30 --temperature 0.8 --top-k 25 --seed 123
```

Use `--temperature 0` for greedy decoding. Use `--top-k 0` to remove the top-k filter. Run `python generate.py --help` for the full command-line options.

## Train the model

To preserve the bundled checkpoint, send a new run to another path:

```bash
python train.py --data the-verdict.txt --epochs 10 --checkpoint checkpoints/my_model.pth
```

`train.py` splits the text into 90% training and 10% validation by character position, then tokenizes each part with GPT-2's tokenizer. It builds input and next-token target windows of 256 tokens, uses a batch size of 2, and optimizes cross-entropy loss with AdamW (learning rate `0.0004`, weight decay `0.1`). It prints sampled train and validation losses plus validation perplexity every five steps and saves weights after each epoch. The random seed is set to `123`.

For a different corpus, pass a UTF-8 file to `--data`. It needs enough text to produce at least one 256-token window in both splits. A GPU is strongly recommended for training this model; the included 10-epoch run can be reproduced on CPU but takes longer.

### Continue training

Starting from model weights resets AdamW's internal state:

```bash
python train.py --resume --epochs 1 --checkpoint checkpoints/my_model.pth
```

To save and restore both model and optimizer state, train with `--save-optimizer`. That creates a second, much larger file with `_resume.pth` in its name:

```bash
python train.py --epochs 1 --save-optimizer --checkpoint checkpoints/my_model.pth
python train.py --resume --epochs 1 --save-optimizer --checkpoint checkpoints/my_model_resume.pth
```

## Model and measured result

WordFlow uses a GPT-style decoder with 12 Transformer blocks, 12 attention heads, 768-dimensional embeddings, a 256-token context, and a 50,257-token GPT-2 vocabulary. This implementation has **163,206,144 trainable parameters** because the input and output embedding weights are separate. It generates one token at a time with a causal attention mask.

The bundled checkpoint was trained from scratch for 10 epochs on *The Verdict*. Before training, a measured training-batch loss was about `10.99`. The best observed validation loss during the run was `6.09` at epoch 7; the final epoch's validation loss was `6.29`. The final training loss was about `0.82`. See [RESULTS.md](RESULTS.md) for the run summary.

The gap between train and validation loss shows that the model overfits this small story. Its output is useful for demonstrating training and decoding, but it is not a general-purpose assistant and may repeat phrases from the corpus.

## Local API

The interface sends `POST /api/generate` with JSON. For example:

```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Every effort moves you","tokens":20,"temperature":0.8,"top_k":25,"seed":123}'
```

The response includes `text`, `continuation`, `prompt_tokens`, and `generated_tokens`. The server rejects empty prompts and out-of-range generation settings.

## Attribution

The GPT architecture and data-loader code in `model.py` comes from Sebastian Raschka's *LLMs-from-scratch* companion repository and retains its copyright header. Its Apache 2.0 licence is included in [UPSTREAM_LICENSE.txt](UPSTREAM_LICENSE.txt). The story text is included as the training corpus used by this project.
