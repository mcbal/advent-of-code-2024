from collections import defaultdict
from io import StringIO
import numpy as np


# part 1


def _split(_pos, _moves):
    _new_pos_0, _new_pos_1 = [], []
    if _moves[0] != 0:
        _new_pos_0.append(_pos[0] + 1)
        _new_pos_1.append(_pos[1])
    if _moves[1] != 0:
        _new_pos_0.append(_pos[0] - 1)
        _new_pos_1.append(_pos[1])
    if _moves[2] != 0:
        _new_pos_0.append(_pos[0])
        _new_pos_1.append(_pos[1] + 1)
    if _moves[3] != 0:
        _new_pos_0.append(_pos[0])
        _new_pos_1.append(_pos[1] - 1)
    return (
        np.asarray(_new_pos_0, dtype=np.int8),
        np.asarray(_new_pos_1, dtype=np.int8),
    )


def solve_topo(grid):
    d_diff = np.diff(grid, n=1, axis=0, append=np.nan)  # (n-1, n) -> (n, n)
    u_diff = -np.diff(grid, n=1, axis=0, prepend=np.nan)  # (n-1, n) -> (n, n)
    r_diff = np.diff(grid, n=1, axis=1, append=np.nan)  # (n, n-1) -> (n, n)
    l_diff = -np.diff(grid, n=1, axis=1, prepend=np.nan)  # (n, n-1) -> (n, n)

    scores = defaultdict(lambda: defaultdict(int))

    def _forking_paths(pos, root=None, level=0):
        diffs = np.stack(
            (
                d_diff[np.minimum(pos[0], pos[0] + 1), pos[1]],
                u_diff[np.maximum(pos[0], pos[0] - 1), pos[1]],
                r_diff[pos[0], np.minimum(pos[1], pos[1] + 1)],
                l_diff[pos[0], np.maximum(pos[1], pos[1] - 1)],
            )
        )
        allowed_moves = [[1], [-1], [1], [-1]] * (diffs == 1)

        for idx, hiker in enumerate(zip(*pos)):
            next_pos = _split(hiker, allowed_moves[:, idx])

            _root = root if root is not None else tuple(map(int, hiker))

            if next_pos[0].size == 0:
                for peak in list(tuple(map(int, x)) for x in zip(*pos)):
                    if grid[peak] == 9:
                        scores[(_root, level, idx)][peak] += 1
            else:
                _forking_paths(next_pos, root=_root, level=level + 1)

    _forking_paths((grid == 0).nonzero(), root=None)  # start at trailheads

    return scores


def solve_scores(grid):
    scores = solve_topo(grid)

    grouped_scores = defaultdict(list)
    for k, v in scores.items():
        grouped_scores[k[:2]].extend(v)

    return sum(len(set(v)) for k, v in grouped_scores.items())


test_data = (
    "89010123\n"
    "78121874\n"
    "87430965\n"
    "96549874\n"
    "45678903\n"
    "32019012\n"
    "01329801\n"
    "10456732"
)

test_grid = np.genfromtxt(
    StringIO(test_data), comments=None, delimiter=1, dtype=np.int8
)
assert solve_scores(test_grid) == 36

real_grid = np.genfromtxt("10_input.txt", comments=None, delimiter=1, dtype=np.int8)
print(solve_scores(real_grid))  # 531


# part 2


def solve_ratings(grid):
    scores = solve_topo(grid)

    grouped_scores = defaultdict(list)
    for k, v in scores.items():
        grouped_scores[k[:2]].append(v)

    return sum(
        vv
        for vv in {
            k: sum(max(v, key=lambda x: len(x)).values())
            for k, v in grouped_scores.items()
        }.values()
    )


assert solve_ratings(test_grid) == 81

print(solve_ratings(real_grid))  # 1210
