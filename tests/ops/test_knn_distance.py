import pytest
import torch
from torkit3d.ops.knn_points import knn_points
from torkit3d.ops.native import knn

test_data = [
    (1, 512, 1024, 3, True),
    (2, 512, 1024, 32, False),
    (3, 511, 1025, 31, True),
    (32, 2048, 8192, 64, False),
]


@pytest.mark.parametrize("b, n1, n2, k, transpose", test_data)
def test(b, n1, n2, k, transpose):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    query = torch.randn(b, n1, 3).cuda()
    key = torch.randn(b, n2, 3).cuda()
    # Need double precision to avoid close distance (leading to ambiguous indices)
    query = query.double()
    key = key.double()
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)

    dist0, idx0 = knn(query, key, k, transpose=transpose, sorted=True)
    dist1, idx1 = knn_points(
        query, key, k, transpose=transpose, sorted=True, sqrt_distance=True
    )
    torch.testing.assert_close(dist0, dist1, atol=5e-5, rtol=1e-4)
    torch.testing.assert_close(idx0, idx1)


def profile(b, n1, n2, k):
    print(f"Profiling for b={b}, n1={n1}, n2={n2}, k={k}")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    query = torch.randn(b, n1, 3).cuda()
    key = torch.randn(b, n2, 3).cuda()

    knn(query, key, k)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        knn(query, key, k)
    print(prof)

    knn_points(query, key, k, version=0)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        knn_points(query, key, k, version=0)
    print(prof)

    knn_points(query, key, k, version=1)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        knn_points(query, key, k, version=1)
    print(prof)


def main():
    profile(32, 2048, 8192, 32)
    profile(4, 512, 10000, 64)
    profile(4, 10000, 512, 3)


if __name__ == "__main__":
    main()
