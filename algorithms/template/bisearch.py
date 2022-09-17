import bisect


def binary_search_left(array, x):
    """binary search left

    >>> array = [-2, 3, 3, 10, 11, 99, 100]
    >>> binary_search_left(array, 3)
    1
    """
    ans = bisect.bisect_left(array, x)
    left = 0
    right = len(array) - 1
    while left <= right:
        mid = (left + right) // 2
        if array[mid] >= x:
            right = mid - 1
        else:
            left = mid + 1
    assert ans == left
    return left

def binary_search_right(array, x):
    """binary search right

    >>> array = [-2, 3, 3, 10, 11, 99, 100]
    >>> binary_search_right(array, 3)
    3
    """
    ans = bisect.bisect_right(array, x)
    left = 0
    right = len(array) - 1
    while left <= right:
        mid = (left + right) // 2
        if array[mid] > x:
            right = mid - 1
        else:
            left = mid + 1
    assert ans == left
    return left

if __name__ == "__main__":
    import doctest

    doctest.testmod()
