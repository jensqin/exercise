# hashing
def colision_test(c):
    A = [47, 61, 36, 52, 56, 33, 92]
    k = [((10 * a + 4) % c) % 7 for a in A]
    return len(set(k)) == 7


if __name__ == "__main__":
    c = 7
    while not colision_test(c):
        c += 1
    print(c)
