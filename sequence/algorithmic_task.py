from collections import Counter


def sequence(a, b):
    return [i for i in a if not is_prime(Counter(b)[i])]


def is_prime(n):
    return all([(n % j) for j in range(2, int(n ** 0.5) + 1)]) and n > 1
