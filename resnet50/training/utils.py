import os
import time
import torch


def get_checkpoint_path(run_name) -> str:
    return f"/data/checkpoints/{run_name}.pt"


def save_checkpoint(
    run_name, model, optimizer, lr_scheduler, epoch, wandb_run_id
) -> None:
    print(f"Saving checkpoint for epoch {epoch}")
    os.makedirs("/data/checkpoints", exist_ok=True)
    start_time = time.time()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "wandb_run_id": wandb_run_id,
        },
        get_checkpoint_path(run_name) + ".tmp",
    )
    os.replace(get_checkpoint_path(run_name) + ".tmp", get_checkpoint_path(run_name))
    print(f"Checkpoint saved in {time.time() - start_time} seconds")


def load_checkpoint(run_name):
    if os.path.exists(get_checkpoint_path(run_name)):
        return torch.load(get_checkpoint_path(run_name), weights_only=False)
    return None


def accuracy(output, target) -> float:
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).float().mean()


class Metric(object):
    def __init__(self, name: str) -> None:
        self.name = name
        self.sum = 0.0
        self.n = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val
        self.n += n

    @property
    def avg(self) -> float:
        if self.n == 0:
            return 0.0
        return self.sum / self.n
