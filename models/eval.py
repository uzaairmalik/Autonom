import torch
import os

try:
    ckpt_path = 'models/best.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        with open('models/eval_output.txt', 'w') as f:
            if 'epoch' in ckpt:
                f.write(f"Epoch: {ckpt['epoch']}\n")
            if 'best_fitness' in ckpt:
                f.write(f"Best Fitness: {ckpt['best_fitness']}\n")
            if 'model' in ckpt and hasattr(ckpt['model'], 'yaml'):
                f.write(f"Model Yaml: {ckpt['model'].yaml}\n")
            if 'metrics' in ckpt:
                f.write(f"Metrics: {ckpt['metrics']}\n")
            elif 'fitness' in ckpt:
                 f.write(f"Fitness: {ckpt['fitness']}\n")
            else:
                 f.write("No metrics found in checkpoint.\n")
                 f.write(str(list(ckpt.keys())))
    else:
        with open('models/eval_output.txt', 'w') as f:
            f.write("best.pt not found")
except Exception as e:
    with open('models/eval_output.txt', 'w') as f:
        f.write(f"Error: {e}")
