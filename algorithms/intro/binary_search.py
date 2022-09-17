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
