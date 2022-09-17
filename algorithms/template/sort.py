"""
sort
---------
This file contains an implementation of merge sort algorithm.
"""


def merge_list(left, right):
    left_len = len(left)
    right_len = len(right)
    left_curr = 0
    right_curr = 0
    ans = []
    while left_curr < left_len and right_curr < right_len:
        if left[left_curr] < right[right_curr]:
            ans.append(left[left_curr])
            left_curr += 1
        else:
            ans.append(right[right_curr])
            right_curr += 1
    while left_curr < left_len:
        ans.append(left[left_curr])
        left_curr += 1
    while right_curr < right_len:
        ans.append(right[right_curr])
        right_curr += 1
    return ans


def merge_sort(array):
    """Returns a list of sorted array elements using merge sort.

    >>> from random import shuffle
    >>> array = [-2, 3, -10, 11, 99, 100000, 100, -200]
    >>> shuffle(array)
    >>> merge_sort(array)
    [-200, -10, -2, 3, 11, 99, 100, 100000]
    """
    # left = 0
    # right = len(array)
    l = len(array)
    if l < 2:
        return array
    mid = l // 2
    left = array[:mid]
    right = array[mid:]
    return merge_list(merge_sort(left), merge_sort(right))

if __name__ == "__main__":
    import doctest

    doctest.testmod()
