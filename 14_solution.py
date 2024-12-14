from itertools import product
import numpy as np
from scipy import stats
from tqdm import tqdm


# part 1


def parse(data, y_shift=0):
    positions, velocities = [], []
    for line in data.split("\n"):
        if line != "":
            p, v = tuple(
                map(
                    lambda s: tuple(s.split(",")),
                    map(lambda s: s.split("=")[1], line.split()),
                )
            )
            positions.append([p[1], p[0]])
            velocities.append([v[1], v[0]])
    return {
        "positions": np.array(positions, dtype=np.int64).T,
        "velocities": np.array(velocities, dtype=np.int64).T,
    }


def solve(data, *, num_cols, num_rows, num_iter, return_positions=False):
    state = parse(data)

    def _wrap_edges(_p):
        for i, bound in enumerate((num_rows, num_cols)):
            _p[i, :][_p[i, :] < 0] = (bound) + (_p[i, :][_p[i, :] < 0] % bound)
            _p[i, :][_p[i, :] >= bound] = _p[i, :][_p[i, :] >= bound] % bound
        return _p

    def _multiply_quadrant_occupation(_positions):
        rows, cols = _positions[0, :], _positions[1, :]
        q1 = rows[(rows < num_rows // 2) & (cols < num_cols // 2)].size
        q2 = rows[(rows < num_rows // 2) & (cols > num_cols // 2)].size
        q3 = rows[(rows > num_rows // 2) & (cols < num_cols // 2)].size
        q4 = rows[(rows > num_rows // 2) & (cols > num_cols // 2)].size
        return q1 * q2 * q3 * q4

    positions = _wrap_edges(state["positions"] + num_iter * state["velocities"])
    if return_positions:
        return positions

    return _multiply_quadrant_occupation(positions)


test_data = (
    "p=0,4 v=3,-3\n"
    "p=6,3 v=-1,-3\n"
    "p=10,3 v=-1,2\n"
    "p=2,0 v=2,-1\n"
    "p=0,0 v=1,3\n"
    "p=3,0 v=-2,-2\n"
    "p=7,6 v=-1,-3\n"
    "p=3,0 v=-1,-2\n"
    "p=9,3 v=2,3\n"
    "p=7,3 v=-1,2\n"
    "p=2,4 v=2,-3\n"
    "p=9,5 v=-3,-3\n"
)

assert solve(test_data, num_cols=11, num_rows=7, num_iter=100) == 12

with open("14_input.txt") as fh:
    real_data = fh.read()

print(solve(real_data, num_cols=101, num_rows=103, num_iter=100))  # 229421808


# part 2


# modulo edge wrapping means rows/cols repeat every `num_rows`/`num_cols` iterations
# so we can take big summed steps to iterate over distinct configurations, e.g.,
# after i*num_rows iterations, row coords will be identical to those of initial state
# but col coords will be offset from initial state col coords by an integer m in [0, num_cols[.
# if we continue to iterate for j*num_cols iterations, we wil stay in the same offset
# col coords (because of modulo in cols direction) but the row coords s will now be
# offset by n in [0, num_rows[ from the row coords in the initial state. so we are
# effectively looping over offset combinations (m, n)


def solve2(data, *, num_cols, num_rows):

    def _entropy(labels, base=None):
        value, counts = np.unique(labels, return_counts=True)
        return stats.entropy(counts, base=base)

    min_entropy_state = ((0, 0), np.inf)
    for r, c in (
        pbar := tqdm(
            product(range(num_rows), range(num_cols)), total=num_rows * num_cols
        )
    ):
        positions = solve(
            real_data,
            num_cols=num_cols,
            num_rows=num_rows,
            num_iter=r * num_rows + c * num_cols,
            return_positions=True,
        )
        grid = np.zeros((num_rows, num_cols))
        grid[positions[0], positions[1]] = 1
        hist_cols, hist_rows = grid.sum(axis=0), grid.sum(axis=1)

        # we use sum of entropies of masks of bigger than mean entries for histograms
        # since we expect clustering structure for the xmas tree state, so lower entropy

        summed_hist_entropy = float(
            _entropy((hist_cols > hist_cols.mean()).flatten(), base=2)
            + _entropy((hist_rows > hist_rows.mean()).flatten(), base=2)
        )

        if summed_hist_entropy < min_entropy_state[1]:
            min_entropy_state = ((r, c), summed_hist_entropy)
            pbar.set_description(
                f"min_entropy {summed_hist_entropy} found @ iteration {r} {c} ({r * num_rows + c * num_cols})"
            )

    min_entropy_num_iter = (
        min_entropy_state[0][0] * num_rows + min_entropy_state[0][1] * num_cols
    )
    positions = solve(
        real_data,
        num_cols=num_cols,
        num_rows=num_rows,
        num_iter=min_entropy_num_iter,
        return_positions=True,
    )

    grid = np.zeros((num_rows, num_cols))
    grid[positions[0], positions[1]] = 1
    viz_grid = np.array2string(
        grid.astype(int),
        max_line_width=num_cols + 10,
        separator="",
        threshold=np.inf,
    ).replace("1", "\x1b[6;30;42m1\x1b[0m")
    print(viz_grid)

    return min_entropy_num_iter


print(solve2(real_data, num_cols=101, num_rows=103))  # 6577
