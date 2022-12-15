import os
import random
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed, deterministic=False):
    """https://github.com/open-mmlab/mmclassification/blob/master/mmcls/apis/train.py"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until a specified number of iterations have been sampled

    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
        https://github.com/Lightning-AI/lightning/issues/5145
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    raise StopIteration

    def __len__(self):
        return self.num_iterations - self.start_iter


def get_latest_checkpoint(ckpt_dir):
    ckpt_paths = list(Path(ckpt_dir).glob("*.ckpt"))
    if len(ckpt_paths) == 0:
        return None
    else:
        ckpt_paths.sort(key=os.path.getmtime)
        return ckpt_paths[-1]
