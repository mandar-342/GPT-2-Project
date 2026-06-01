import torch
import torch.nn.functional as F
from config import Config
from data import TextDataset
from model import GPT


def sample_logits(logits, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.1, prev_tokens=None):
    logits = logits / temperature

    
    if prev_tokens is not None:
        for token in set(prev_tokens):
            logits[:, token] /= repetition_penalty

    # top-k filter
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')

    # top-p (nucleus) filter
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_idx_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_idx_to_remove] = -float('Inf')
        logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(model, context, max_new_tokens, config):
    generated = context

    for _ in range(max_new_tokens):
        idx_cond = generated[:, -config.block_size:]
        logits, _ = model(idx_cond)

        logits = logits[:, -1, :]

        next_token = sample_logits(
            logits,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15,
            prev_tokens=generated[0].tolist()[-200:]  
        )

        generated = torch.cat((generated, next_token), dim=1)

    return generated


def encode_prompt(prompt, stoi):
   
    return [stoi[c] for c in prompt if c in stoi]


def main():
    config = Config()

    dataset = TextDataset("input.txt", config.block_size)
    vocab_size = len(dataset.stoi)

    model = GPT(vocab_size, config)
    model.load_state_dict(torch.load("best.pt", map_location=config.device))
    model.to(config.device)
    model.eval()

    prompt = "To be or not to be"
    encoded = encode_prompt(prompt, dataset.stoi)

    if not encoded:
        print("Prompt has no known characters, starting from newline.")
        encoded = [dataset.stoi.get('\n', 0)]

    context = torch.tensor([encoded], device=config.device)

    out = generate(model, context, 500, config)[0].tolist()

    decoded = "".join(dataset.itos[i] for i in out)
    print(decoded)


if __name__ == "__main__":
    main()
