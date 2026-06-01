import math
import torch
from config import Config
from data import TextDataset
from model import GPT


def get_lr(step, config):
    # linear warmup then cosine decay
    if step < config.warmup_iters:
        return config.learning_rate * step / config.warmup_iters
    progress = (step - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def evaluate(model, dataset, config):
    model.eval()
    losses = {}

    for split in ["train", "val"]:
        loss_list = []
        for _ in range(config.eval_iters):
            x, y = dataset.get_batch(split, config.batch_size)
            x, y = x.to(config.device), y.to(config.device)

            _, loss = model(x, y)
            loss_list.append(loss.item())

        losses[split] = sum(loss_list) / len(loss_list)

    model.train()
    return losses


def main():
    config = Config()
    print(f"device: {config.device}")

    dataset = TextDataset("input.txt", config.block_size)
    vocab_size = len(dataset.stoi)

    print("vocab size:", vocab_size)

    model = GPT(vocab_size, config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {total_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    best_val = float("inf")
    patience = 5
    counter = 0

    for step in range(config.max_iters):
        # update learning rate
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = dataset.get_batch("train", config.batch_size)
        x, y = x.to(config.device), y.to(config.device)

        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if step % config.eval_interval == 0:
            stats = evaluate(model, dataset, config)
            print(f"{step} | train {stats['train']:.4f} | val {stats['val']:.4f} | lr {lr:.2e}")

            if stats["val"] < best_val:
                best_val = stats["val"]
                torch.save(model.state_dict(), "best.pt")
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("early stopping")
                break


if __name__ == "__main__":
    main()
