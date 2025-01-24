import os
from pathlib import Path
import modal
import modal.experimental
import training.config as config
from torchrun_util import torchrun

cuda_version = "12.4.0"
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

training_code_dir = Path(__file__).parent / "training"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("curl", "unzip", "vim", "git", "htop")
    .pip_install("torch", "torchvision", "nvidia-dali-cuda120")
    # needed until they update PyPI
    .pip_install("git+https://github.com/thecodingwizard/webdataset.git")
    .pip_install("wandb")
    .pip_install("tqdm")
    .pip_install("pydantic")  # used by training.torchrun utility
    .add_local_dir(training_code_dir, remote_path="/root/training")
)
app = modal.App(
    f"resnet50-{config.run_name}",
    image=image,
    secrets=[
        # Required for downloading the ImageNet dataset.
        modal.Secret.from_name("huggingface-secret"),
        # Required for connecting to Weights & Biases from within the Modal container.
        modal.Secret.from_name("wandb-secret-modal-labs"),
    ],
    volumes={
        "/data": modal.Volume.from_name("imagenet"),
    },
)


@app.function(
    gpu=modal.gpu.H100(count=config.gpus_per_node),
    timeout=60 * 60 * 6,
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
)
@modal.experimental.clustered(size=config.nodes)
def train_resnet():
    cluster_info = modal.experimental.get_cluster_info()

    torchrun.run(
        node_rank=cluster_info.rank,
        master_addr=cluster_info.container_ips[0],
        master_port=1234,
        nnodes=str(config.nodes),
        nproc_per_node=str(config.gpus_per_node),
        training_script="training/train.py",
        training_script_args=[],
    )


@app.local_entrypoint()
def main():
    if config.runtime == "runc":
        assert os.environ["MODAL_FUNCTION_RUNTIME"] == "runc"
    else:
        assert "MODAL_FUNCTION_RUNTIME" not in os.environ
    train_resnet.remote()
