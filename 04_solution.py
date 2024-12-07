from io import StringIO

import numpy as np


# part 1


IGNORE_IDX = -1


def count_hits(filter, array):
    def maybe_take(a):
        if any(filter.shape == np.ones_like(filter.shape)):  # 1d filter
            return a
        else:  # 2d filter
            return np.take(
                a.reshape(*a.shape[:-2], a.shape[-2] * a.shape[-1]),
                np.flatnonzero(filter != IGNORE_IDX),
                axis=-1,
            )

    return len(
        np.flatnonzero(
            maybe_take(
                np.lib.stride_tricks.sliding_window_view(array, filter.shape) == filter
            ).all(axis=-1)
        )
    )


def solve(grid):
    str2idx = {char: np.int8(idx) for idx, char in enumerate(np.unique(grid))}
    grid_ids = np.vectorize(lambda c: str2idx[c])(grid)
    h_filter = np.array([[str2idx[x] for x in "XMAS"]])
    d_filter = np.full((4, 4), IGNORE_IDX, dtype=np.int8)
    np.fill_diagonal(d_filter, [str2idx[x] for x in "XMAS"])

    xmas = 0
    xmas += count_hits(h_filter, grid_ids)  # h
    xmas += count_hits(np.flip(h_filter, axis=-1), grid_ids)  # h rev
    xmas += count_hits(h_filter, np.rot90(grid_ids))  # v
    xmas += count_hits(np.flip(h_filter, axis=-1), np.rot90(grid_ids))  # v rev
    xmas += count_hits(d_filter, grid_ids)  # d \
    xmas += count_hits(d_filter, np.rot90(grid_ids))  # d /
    xmas += count_hits(np.flip(d_filter, axis=(-1, -2)), grid_ids)  # d \
    xmas += count_hits(np.flip(d_filter, axis=(-1, -2)), np.rot90(grid_ids))  # d \ rev

    return xmas


test_data = (
    "MMMSXXMASM\n"
    "MSAMXMSMSA\n"
    "AMXSXMAAMM\n"
    "MSAMASMSMX\n"
    "XMASAMXAMM\n"
    "XXAMMXXAMA\n"
    "SMSMSASXSS\n"
    "SAXAMASAAA\n"
    "MAMMMXMMMM\n"
    "MXMXAXMASX"
)
test_grid = np.genfromtxt(StringIO(test_data), delimiter=1, dtype="<U19")
assert solve(test_grid) == 18

real_grid = np.genfromtxt("04_input.txt", delimiter=1, dtype="<U19")
print(solve(real_grid))  # 2297


# part 2


def solve2(grid):
    str2idx = {char: np.int8(idx) for idx, char in enumerate(np.unique(grid))}
    grid_ids = np.vectorize(lambda c: str2idx[c])(grid)

    diag_mask = np.eye(3).astype(bool)
    x_filter_1 = np.full((3, 3), IGNORE_IDX, dtype=np.int8)
    x_filter_1[diag_mask] = [str2idx[x] for x in "MAS"]
    x_filter_1[np.fliplr(diag_mask)] = [str2idx[x] for x in "MAS"]
    x_filter_2 = np.full((3, 3), IGNORE_IDX, dtype=np.int8)
    x_filter_2[diag_mask] = [str2idx[x] for x in "MAS"]
    x_filter_2[np.fliplr(diag_mask)] = [str2idx[x] for x in "SAM"]

    xmas = 0
    xmas += count_hits(x_filter_1, grid_ids)
    xmas += count_hits(np.flipud(x_filter_1), grid_ids)
    xmas += count_hits(x_filter_2, grid_ids)
    xmas += count_hits(np.fliplr(x_filter_2), grid_ids)

    return xmas


assert solve2(test_grid) == 9

print(solve2(real_grid))  # 1745
