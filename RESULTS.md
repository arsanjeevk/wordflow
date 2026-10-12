# WordFlow training results

| Item | Value |
| --- | --- |
| Corpus | `the-verdict.txt` (20,479 bytes; 5,145 GPT-2 tokens) |
| Model | 12 layers, 12 heads, 768-dimensional embeddings, 256-token context, 163,206,144 trainable parameters |
| Split | 90% train / 10% validation by character position |
| Optimizer | AdamW, learning rate 0.0004, weight decay 0.1 |
| Run | 10 epochs, batch size 2, seed 123 |
| Initial measured training-batch loss | 10.99 |
| Best observed validation loss | 6.09 at epoch 7 |
| Final observed train / validation loss | About 0.82 / 6.29 at epoch 10 |
| Saved checkpoint | `checkpoints/verdict_model.pth` (final epoch) |

Example greedy continuation from the saved checkpoint:

```text
Every effort moves you?"

"Yes--quite insensible to
```

The validation loss remained much higher than the training loss. The small corpus limits how broadly the model can generalise; generated text may resemble passages it has seen.
