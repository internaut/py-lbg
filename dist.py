import math


def euclid_squared(a, b):
    return sum((x_a - x_b) ** 2 for x_a, x_b in zip(a, b))


def euclid(a, b):
    return math.sqrt(euclid_squared(a, b))
