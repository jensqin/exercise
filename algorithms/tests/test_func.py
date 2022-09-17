from intro.binary_search import binary_search
from intro.sorting import merge_sort


def test_merge():
    # merge sort
    x = [2, 3, 6, 4, 7, 0, 8, 1]
    y = sorted(x)
    merge_sort(x)
    assert x == y


def test_binary_search():
    x = sorted([2, 3, 6, 4, 7, 0, 8, 1])
    t = 3
    assert binary_search(x, t) == 3
