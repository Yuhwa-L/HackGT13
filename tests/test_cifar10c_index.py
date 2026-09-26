from benchmark.load_cifar10c import base_and_severity_to_row, row_to_base_and_severity


def test_row_math():
    assert row_to_base_and_severity(0) == (0, 1)
    assert row_to_base_and_severity(9999) == (9999, 1)
    assert row_to_base_and_severity(10000) == (0, 2)
    assert row_to_base_and_severity(49999) == (9999, 5)


def test_roundtrip():
    for i in (0, 123, 10123, 49999):
        assert base_and_severity_to_row(*row_to_base_and_severity(i)) == i
