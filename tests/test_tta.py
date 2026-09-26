import pytest

torch = pytest.importorskip("torch")
from inference.tta import hflip, shift  # noqa: E402


def test_shapes_and_identities():
    x = torch.randn(4, 3, 32, 32)
    assert hflip(x).shape == x.shape
    assert shift(x, 2).shape == x.shape
    assert torch.equal(hflip(hflip(x)), x)
    y = shift(shift(x, 2), -2)
    assert torch.allclose(y[..., 2:-2], x[..., 2:-2])
