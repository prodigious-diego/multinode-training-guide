from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch

import glob
import numpy as np


# You can also decode the image on the GPU if you're using runc.
@pipeline_def()
def _imagenet_pipeline(
    wds_data, wds_indexes, shard_id: int, num_shards: int, is_training: bool = True
):
    images, cls = fn.readers.webdataset(
        paths=wds_data,
        ext=["jpg", "cls"],
        missing_component_behavior="error",
        index_paths=wds_indexes,
        name="Reader",
        shard_id=shard_id,
        num_shards=num_shards,
    )
    if is_training:
        images = fn.decoders.image_random_crop(
            images,
            device="cpu",
            output_type=types.RGB,
            random_aspect_ratio=[0.8, 1.25],
            random_area=[0.1, 1.0],
            num_attempts=100,
        )
        images = fn.resize(
            images,
            device="cpu",
            resize_x=224,
            resize_y=224,
            interp_type=types.INTERP_TRIANGULAR,
        )
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images, device="cpu", output_type=types.RGB)
        images = fn.resize(
            images,
            device="cpu",
            size=256,
            mode="not_smaller",
            interp_type=types.INTERP_TRIANGULAR,
        )
        mirror = False

    output = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror,
    )

    # wds returns pure ascii bytes. 48 = '0' in ascii
    cls = fn.pad(cls, device="cpu", fill_value=48)
    return output, cls


def get_dataloader(
    name: str,
    batch_size: int,
    seed: int | None,
    device_id: int,
    shard_id: int,
    num_shards: int,
):
    assert name in ["train", "val"]
    return DALIGenericIterator(
        [
            _imagenet_pipeline(
                sorted(glob.glob(f"/data/{name}/*.tar")),
                sorted(glob.glob(f"/data/{name}/*.idx")),
                batch_size=batch_size,
                num_threads=4,
                seed=seed,
                device_id=device_id,
                shard_id=shard_id,
                num_shards=num_shards,
            )
        ],
        ["data", "label"],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )


def parse_wds_labels(labels) -> torch.Tensor:
    labels = labels.numpy()
    labels = [int(np.char.asarray(label - 48).tobytes()) for label in labels]
    return torch.LongTensor(labels).cuda(non_blocking=True)
