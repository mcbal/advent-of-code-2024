from collections import defaultdict, deque
from io import StringIO

import numpy as np
from tqdm import tqdm


# part 1


def _flood_fill(grid, pos, label):
    prev_label = grid[pos]
    if prev_label == label:
        return
    q = deque()
    q.append(pos)
    grid[pos] = label
    while q:
        x, y = q.popleft()
        if x + 1 < grid.shape[0] and grid[x + 1][y] == prev_label:
            grid[x + 1][y] = label
            q.append((x + 1, y))
        if x - 1 >= 0 and grid[x - 1][y] == prev_label:
            grid[x - 1][y] = label
            q.append((x - 1, y))
        if y + 1 < grid.shape[1] and grid[x][y + 1] == prev_label:
            grid[x][y + 1] = label
            q.append((x, y + 1))
        if y - 1 >= 0 and grid[x][y - 1] == prev_label:
            grid[x][y - 1] = label
            q.append((x, y - 1))


def _relabel_connected_components(
    _grid_str,
):  # can be sped up a lot by using numbers instead of chars/strings here
    cc = defaultdict(list)
    for unique_label in tqdm(np.unique(_grid_str)):
        mask = _grid_str == unique_label
        maybe_connected_coords = list(zip(*np.where(mask)))

        test_pos = maybe_connected_coords[0]
        test_grid = np.vectorize(lambda c: "@" if c == unique_label else c)(_grid_str)

        while True:
            prev_test_grid = test_grid.copy()
            _flood_fill(test_grid, test_pos, unique_label)
            if np.all(mask == (test_grid == unique_label)):
                cc[str(unique_label)].append(np.where(test_grid != prev_test_grid))
                break
            else:
                cc[str(unique_label)].append(np.where(test_grid != prev_test_grid))
                test_pos = next(
                    _pos
                    for _pos in maybe_connected_coords
                    if _pos in list(zip(*np.where(test_grid == "@")))
                )
    for k, v in cc.items():
        for i, vv in enumerate(v):
            _grid_str[vv] = f"{k}{i}"
    return _grid_str


def _tokenize(_grid_str):
    _vocab = {label: idx for idx, label in enumerate(np.unique(_grid_str))}
    return (
        np.array(
            [_vocab[char] for row in _grid_str.tolist() for char in row], dtype=np.int16
        ).reshape(_grid_str.shape),
        _vocab,
    )


def solve(grid_str):
    grid_str = _relabel_connected_components(grid_str)
    grid, vocab = _tokenize(grid_str)

    def _area(_idx):
        return (grid == _idx).sum()

    def _perimeter(_idx):
        mask = (np.pad(grid, 1, constant_values=-1) == _idx).astype(bool)
        return (mask[:, 1:] != mask[:, :-1]).sum() + (mask[1:, :] != mask[:-1, :]).sum()

    return sum(_area(idx) * _perimeter(idx) for idx in vocab.values())


assert (
    solve(
        np.genfromtxt(
            StringIO(("AAAA\n" "BBCD\n" "BBCC\n" "EEEC\n")),
            comments=None,
            delimiter=1,
            dtype="<U19",
        )
    )
    == 140
)

assert (
    solve(
        np.genfromtxt(
            StringIO(("OOOOO\n" "OXOXO\n" "OOOOO\n" "OXOXO\n" "OOOOO\n")),
            comments=None,
            delimiter=1,
            dtype="<U19",
        )
    )
    == 772
)


assert (
    solve(
        np.genfromtxt(
            StringIO(
                (
                    "RRRRIICCFF\n"
                    "RRRRIICCCF\n"
                    "VVRRRCCFFF\n"
                    "VVRCCCJFFF\n"
                    "VVVVCJJCFE\n"
                    "VVIVCCJJEE\n"
                    "VVIIICJJEE\n"
                    "MIIIIIJJEE\n"
                    "MIIISIJEEE\n"
                    "MMMISSJEEE\n"
                )
            ),
            comments=None,
            delimiter=1,
            dtype="<U19",
        )
    )
    == 1930
)

print(
    solve(np.genfromtxt("12_input.txt", comments=None, delimiter=1, dtype="<U19"))
)  # 1452678


# part 2


def solve2(grid_str):
    grid_str = _relabel_connected_components(grid_str)
    grid, vocab = _tokenize(grid_str)

    def _area(_idx):
        return (grid == _idx).sum()

    def _sides(_idx):
        mask = (np.pad(grid, 1, constant_values=-1) == _idx).astype(int)

        def _fix_flat_diff(a):
            a_padded = np.pad(a, 1, constant_values=0)
            cond = np.abs(a_padded[1:] - a_padded[:-1]) == 2
            # shift because we update in place and positions obtained from np.where no longer match for multiple hits
            for shift, idx in enumerate(list(zip(*np.where(cond)))):
                a = np.concatenate(
                    (a[: idx[0] + shift], np.array([0]), a[idx[0] + shift :])
                )  # pull apart (..., -1, 1, ...) and (..., 1, -1, ...) regions in 1d array
            return np.abs(a)

        diff_h = mask[:, 1:] - mask[:, :-1]
        flat_diff_h = diff_h.flatten(order="F")
        flat_diff_h = _fix_flat_diff(flat_diff_h)

        diff_v = mask[1:, :] - mask[:-1, :]
        flat_diff_v = diff_v.flatten(order="C")
        flat_diff_v = _fix_flat_diff(flat_diff_v)

        def num_consecutive_pieces(a):
            return sum(
                aa.size > 1 for aa in np.split(a, np.where(np.diff(a) != 1)[0] + 1)
            )

        return num_consecutive_pieces(flat_diff_h.cumsum()) + num_consecutive_pieces(
            flat_diff_v.cumsum()
        )

    return sum(_area(idx) * _sides(idx) for idx in vocab.values())


assert (
    solve2(
        np.genfromtxt(
            StringIO(("AAAA\n" "BBCD\n" "BBCC\n" "EEEC\n")),
            comments=None,
            delimiter=1,
            dtype="<U19",
        )
    )
    == 80
)

assert (
    solve2(
        np.genfromtxt(
            StringIO(("OOOOO\n" "OXOXO\n" "OOOOO\n" "OXOXO\n" "OOOOO\n")),
            comments=None,
            delimiter=1,
            dtype="<U19",
        )
    )
    == 436
)

assert (
    solve2(
        np.genfromtxt(
            StringIO(("EEEEE\n" "EXXXX\n" "EEEEE\n" "EXXXX\n" "EEEEE\n")),
            comments=None,
            delimiter=1,
            dtype="<U19",
        )
    )
    == 236
)


assert (
    solve2(
        np.genfromtxt(
            StringIO(
                ("AAAAAA\n" "AAABBA\n" "AAABBA\n" "ABBAAA\n" "ABBAAA\n" "AAAAAA\n")
            ),
            comments=None,
            delimiter=1,
            dtype="<U19",
        )
    )
    == 368
)


assert (
    solve2(
        np.genfromtxt(
            StringIO(
                (
                    "RRRRIICCFF\n"
                    "RRRRIICCCF\n"
                    "VVRRRCCFFF\n"
                    "VVRCCCJFFF\n"
                    "VVVVCJJCFE\n"
                    "VVIVCCJJEE\n"
                    "VVIIICJJEE\n"
                    "MIIIIIJJEE\n"
                    "MIIISIJEEE\n"
                    "MMMISSJEEE\n"
                )
            ),
            comments=None,
            delimiter=1,
            dtype="<U19",
        )
    )
    == 1206
)


print(
    solve2(np.genfromtxt("12_input.txt", comments=None, delimiter=1, dtype="<U19"))
)  # 873584
