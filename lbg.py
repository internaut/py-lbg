#
# Python 3!
#

import dist
from functools import reduce
from collections import defaultdict

_size_data = 0
_dim = 0


def generate_codebook(data, size_codebook, epsilon=0.00001):
    global _size_data, _dim

    _size_data = len(data)
    assert _size_data > 0

    _dim = len(data[0])
    assert _dim > 0

    codebook = []
    codebook_abs_weights = [_size_data]
    codebook_rel_weights = [1.0]

    # calculate initial codevector: average vector of whole input data
    c0 = avg_vec_of_vecs(data, _dim, _size_data)
    codebook.append(c0)

    while len(codebook) < size_codebook:
        avg_dist = avg_distortion_c0(c0, data)

        codebook, codebook_abs_weights, codebook_rel_weights = split_codebook(data, codebook,
                                                                              codebook_abs_weights,
                                                                              codebook_rel_weights,
                                                                              epsilon, avg_dist)

    return codebook, codebook_abs_weights, codebook_rel_weights


def split_codebook(data, codebook, abs_weights, rel_weights, epsilon, initial_avg_dist):
    # split codevectors
    new_codevectors = []
    for c in codebook:
        c1 = new_codevector(c, epsilon)
        c2 = new_codevector(c, -epsilon)
        new_codevectors.extend((c1, c2))

    codebook = new_codevectors
    len_codebook = len(codebook)
    abs_weights = [0] * len_codebook
    rel_weights = [0.0] * len_codebook

    # print('> splitting to size', len_codebook)

    avg_dist = 0
    err = epsilon + 1
    num_iter = 0
    while err > epsilon:
        # find closest codevectors for each vector in data
        closest_c_list = [None] * _size_data
        vecs_near_c = defaultdict(list)
        vec_idxs_near_c = defaultdict(list)
        for i, vec in enumerate(data):
            min_dist = None
            closest_c_index = None
            for i_c, c in enumerate(codebook):
                d = dist.euclid_squared(vec, c)
                if min_dist is None or d < min_dist:    # found new closest codevector
                    min_dist = d
                    closest_c_list[i] = c
                    closest_c_index = i_c
            vecs_near_c[closest_c_index].append(vec)
            vec_idxs_near_c[closest_c_index].append(i)

        # update codebook
        for i_c in range(len_codebook):
            vecs = vecs_near_c.get(i_c) or []
            num_vecs_near_c = len(vecs)
            if num_vecs_near_c > 0:
                new_c = avg_vec_of_vecs(vecs, _dim)
                codebook[i_c] = new_c
                for i in vec_idxs_near_c[i_c]:
                    closest_c_list[i] = new_c
                abs_weights[i_c] = num_vecs_near_c
                rel_weights[i_c] = num_vecs_near_c / _size_data

        # recalculate average distortion value
        prev_avg_dist = avg_dist if avg_dist > 0 else initial_avg_dist
        avg_dist = avg_distortion_c_list(closest_c_list, data)
        err = (prev_avg_dist - avg_dist) / prev_avg_dist
        # print(closest_c_list)
        # print('> iteration', num_iter, 'avg_dist', avg_dist, 'prev_avg_dist', prev_avg_dist, 'err', err)

        num_iter += 1

    return codebook, abs_weights, rel_weights


def avg_vec_of_vecs(vecs, dim=None, size=None):
    size = size or len(vecs)
    dim = dim or len(vecs[0])
    avg_vec = [0.0] * dim
    for vec in vecs:
        for i, x in enumerate(vec):
            avg_vec[i] += x / size

    return avg_vec


def new_codevector(c, e):
    return [x * (1.0 + e) for x in c]


def avg_distortion_c0(c0, data, size=None):
    size = size or _size_data
    return reduce(lambda s, d:  s + d / size,
                  (dist.euclid_squared(c0, vec)
                   for vec in data),
                  0.0)


def avg_distortion_c_list(c_list, data, size=None):
    size = size or _size_data
    return reduce(lambda s, d:  s + d / size,
                  (dist.euclid_squared(c_i, data[i])
                   for i, c_i in enumerate(c_list)),
                  0.0)
