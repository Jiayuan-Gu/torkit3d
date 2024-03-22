import pytest
import numpy as np
import torch
from torkit3d.ops.sample_farthest_points import sample_farthest_points


def sample_farthest_points_np(
    points: np.ndarray, num_samples: int, transpose=False
) -> np.ndarray:
    """Sample farthest points (numpy version)

    Args:
        points: [B, N, 3] or [B, 3, N] if @transpose is True.
        num_samples: the number of samples.
        transpose: whether to transpose points.

    Returns:
        np.array: [batch_size, num_samples], sample indices
    """
    if transpose:
        points = np.transpose(points, [0, 2, 1])
    index = []
    for points_per_batch in points:
        index_per_batch = [0]
        cur_ind = 0
        dist2set = None
        for ind in range(1, num_samples):
            cur_xyz = points_per_batch[cur_ind]
            dist2cur = points_per_batch - cur_xyz[None, :]
            dist2cur = np.square(dist2cur).sum(1)
            if dist2set is None:
                dist2set = dist2cur
            else:
                dist2set = np.minimum(dist2cur, dist2set)
            cur_ind = np.argmax(dist2set)
            index_per_batch.append(cur_ind)
        index.append(index_per_batch)
    return np.asarray(index)


test_data = [
    (1, 31, 2, True),
    (2, 1024, 128, True),
    (3, 1025, 129, False),
    (3, 1025, 129, True),
    (16, 1024, 512, False),
    (32, 1024, 512, True),
    (32, 8192, 2048, False),
]


@pytest.mark.parametrize("batch_size, num_points, num_samples, transpose", test_data)
def test(batch_size, num_points, num_samples, transpose):
    np.random.seed(0)
    points_np = np.random.rand(batch_size, num_points, 3)
    if transpose:
        points_np = np.transpose(points_np, [0, 2, 1])

    index_np = sample_farthest_points_np(points_np, num_samples, transpose=transpose)
    points = torch.from_numpy(points_np).double().cuda()
    index = sample_farthest_points(points, num_samples, transpose=transpose)
    np.testing.assert_equal(index_np, index.cpu().numpy())


def profile(batch_size, num_points, num_samples, transpose=False):
    print(f"Profiling ({batch_size}, {num_points}, {num_samples})")
    torch.manual_seed(0)
    points = torch.randn(batch_size, num_points, 3).cuda()
    with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
        sample_farthest_points(points, num_samples)
    print(prof)


def main():
    profile(16, 8192, 2048, True)
    profile(4, 10000, 512, False)


if __name__ == "__main__":
    main()
