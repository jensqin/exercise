def binary_search(input, target):
    assert input == sorted(input)
    left = 0
    right = len(input) - 1
    while left <= right:
        mid = (left + right) // 2
        if input[mid] == target:
            return mid
        elif input[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return -1

def bisect_left(x, target):
    l = 0
    r = len(x) - 1
    while l <= r:
        mid = (l + r) // 2
        if x[mid] >= target:
            r = mid - 1
        else:
            l = mid + 1
    return l

def biset_right(x, target):
    l = 0
    r = len(x) - 1
    while l <= r:
        mid = (l + r) // 2
        if x[mid] <= target:
            l = mid + 1
        else:
            r = mid - 1
    return l
    