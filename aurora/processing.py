import os
import torch

def save_checkpoint(state, checkpoint_dir, max_checkpoints=5):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{state['epoch']}.pth")
    torch.save(state, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    checkpoints = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    while len(checkpoints) > max_checkpoints:
        oldest_checkpoint = checkpoints.pop(0)
        os.remove(os.path.join(checkpoint_dir, oldest_checkpoint))

