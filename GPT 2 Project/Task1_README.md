# Task 1 — Building a GPT Language Model from Scratch

For this task I built a character-level GPT model entirely from scratch using PyTorch and trained it on the complete works of Shakespeare. The goal was to get the model to generate text that sounds like Shakespeare — and honestly, the results came out better than I expected for a model this small.

---

## What I Built

The model is a decoder-only Transformer (same architecture as GPT) that works at the character level — meaning it doesn't use word tokens, it predicts one character at a time. I implemented everything by hand: the multi-head self-attention, the feed-forward layers, positional embeddings, the training loop, and the generation logic. No pretrained weights, no shortcuts.

---

## How to Run

### Pretraining

Make sure `input.txt` (the Shakespeare dataset) is in the same folder as the scripts, then just run:

```bash
python train.py
```

It prints the train and validation loss every 250 steps. The best model checkpoint (lowest validation loss) gets saved automatically to `best.pt`. On a GPU it takes about 15–20 minutes. Early stopping is built in — if the validation loss doesn't improve for 5 consecutive evaluations, training stops on its own.

### Generating Text

Once training is done and `best.pt` exists:

```bash
python generate.py
```

The script will ask you to type a prompt and how many characters you want generated. It keeps running in a loop so you can try multiple prompts without restarting.

Example session:
```


Enter your prompt: Shall I compare thee
How many characters to generate? (default 500): 300

--- Output ---
Shall I compare thee to the king of all,
That thou dost love the world...
```

---

## Sample Output

Prompt used: `To be or not to be`

![Sample Generation Output](generation%20image.png)

The model picks up Shakespeare's vocabulary (thee, thou, thy, shalt, dream'd), the dramatic line structure, and even the punctuation patterns pretty well for something trained from scratch on a single text file.

---

## Training and Validation Loss

![Training and Validation Losses](training%20and%20validation%20losses.png)

Trained on a 90/10 train/val split of the Shakespeare corpus (vocab size: 65 unique characters). The val loss closely followed the train loss throughout — ending at train 1.21 and val 1.49, a gap of about 0.28. That's a healthy margin and shows the model generalised rather than just memorising the training text.

---

## Model Architecture and Hyperparameters

| Parameter | Value |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Layers | 6 |
| Attention heads | 8 |
| Embedding size | 384 |
| Context length | 256 characters |
| Total parameters | ~10.76M |
| Dropout | 0.2 |

| Training Parameter | Value |
|---|---|
| Batch size | 64 |
| Learning rate | 3e-4 with cosine decay to 3e-5 |
| Warmup steps | 200 |
| Max iterations | 5000 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Gradient clipping | 1.0 |
| Early stopping patience | 5 evaluations |

---

## Why These Hyperparameters

I didn't land on these settings on the first try. My first attempt used 8 layers and an embedding size of 512 (~25M parameters), and the model badly overfit — train loss dropped below 0.3 but val loss shot up to 2.3. Shakespeare is only about 1MB of text, so a 25M parameter model has way more capacity than it needs and just ends up memorising the training data.

Dropping to 6 layers and 384 embedding (~10.76M params) fixed that. The model now has enough capacity to learn the style without memorising specific passages.

I also bumped dropout from 0.1 to 0.2 to add a bit more regularisation on top of that. It helped close the train/val gap noticeably.

For the learning rate I used a cosine decay schedule with a short linear warmup. The warmup prevents large unstable gradient updates right at the start of training, and the cosine decay lets the model settle into a good minimum gradually rather than bouncing around a fixed LR. AdamW with β₂=0.95 instead of the default 0.999 makes the optimiser respond a bit faster to recent gradient information, which tends to work better for language model training.

Gradient clipping at 1.0 is just a safety net — transformers can occasionally throw out huge gradients early in training and clipping stops that from blowing up the weights.

Early stopping with patience 5 meant I didn't have to pick a fixed number of steps to train for. The run ended naturally around step 4750 once the val loss had plateaued at 1.49.

---

## Approach Summary

The overall approach was pretty straightforward — build the simplest possible GPT that could still produce decent output, get the training stable, then tune the model size to match the dataset. The character-level tokenisation keeps things simple (no need for a tokenizer library, the vocabulary is just the 65 unique characters in Shakespeare) and the results speak for themselves. The generated text has the right rhythm, vocabulary, and dramatic feel of Shakespeare even though the model never saw any labels or templates — it purely learned by predicting the next character over and over.
