# selection sort, O(n^2)
from re import S


def selection_sort(input, i=None):
    if i is None:
        i = len(input) - 1
    if i > 0:
        j = id_max(input, i)
        input[j], input[i] = input[i], input[j]
        selection_sort(input, i - 1)


def id_max(input, i):
    if i > 0:
        j = id_max(input, i - 1)
        if input[j] > input[i]:
            return j
    return i


# insertion sort, O(n^2)
def insertion_sort(input, i=None):
    if i is None:
        i = len(input) - 1
    if i > 0:
        insertion_sort(input, -1)
        insert_last(input, i)


def insert_last(input, i):
    if i > 0 and input[i] < input[i - 1]:
        input[i], input[i - 1] = input[i - 1], input[i]
        insert_last(input, i - 1)


# merge sort
def merge_sort(input, a=0, b=None):
    if b is None:
        b = len(input)

    def merge(l, r, inp, i, j, a, b):
        if a < b:
            if (j == 0) or (i > 0 and l[i - 1] > r[j - 1]):
                inp[b - 1] = l[i - 1]
                i = i - 1
            else:
                inp[b - 1] = r[j - 1]
                j = j - 1
            merge(l, r, inp, i, j, a, b - 1)

    if b - a > 1:
        c = (a + b + 1) // 2
        merge_sort(input, a, c)
        merge_sort(input, c, b)
        l, r = input[a:c], input[c:b]
        merge(l, r, input, len(l), len(r), a, b)


if __name__ == "__main__":
    x = [2, 3, 6, 4, 7, 0, 8, 1]
    merge_sort(x)
