import torch

try:
    ckpt = torch.load('models/best.pt', map_location='cpu', weights_only=False)
    with open('models/eval_output2.txt', 'w') as f:
        if 'train_metrics' in ckpt:
            f.write(f"Train Metrics: {ckpt['train_metrics']}\n")
        if 'train_results' in ckpt:
            f.write(f"Train Results: {ckpt['train_results']}\n")
except Exception as e:
    with open('models/eval_output2.txt', 'w') as f:
        f.write(f"Error: {e}")
