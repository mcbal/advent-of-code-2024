from io import StringIO
import numpy as np


def _parse(grid):
    antennas = {}
    for freq in np.unique(grid):
        if freq != ".":
            antennas[str(freq)] = np.array(list(zip(*np.where(grid == freq))))
    return antennas, grid.shape


def solve(grid):
    antennas, bounds = _parse(grid)
    antinodes = None
    for coords in antennas.values():
        diffs = coords[:, None, :] - coords
        unbounded_antinodes = (coords + 2 * diffs)[
            ~np.eye(coords.shape[0], dtype=bool)
        ].reshape(-1, 2)
        bounded_antinodes = unbounded_antinodes[
            (
                (0 <= unbounded_antinodes[:, 0])
                & (unbounded_antinodes[:, 0] < bounds[0])
                & (0 <= unbounded_antinodes[:, 1])
                & (unbounded_antinodes[:, 1] < bounds[1])
            )
        ]
        antinodes = np.unique(
            (
                np.concatenate(
                    (antinodes, bounded_antinodes),
                    axis=0,
                )
                if antinodes is not None
                else bounded_antinodes
            ),
            axis=0,
        )
    return len(antinodes)


test_data = (
    "............\n"
    "........0...\n"
    ".....0......\n"
    ".......0....\n"
    "....0.......\n"
    "......A.....\n"
    "............\n"
    "............\n"
    "........A...\n"
    ".........A..\n"
    "............\n"
    "............"
)
test_grid = np.genfromtxt(StringIO(test_data), delimiter=1, dtype="<U19")
print(solve(test_grid))
assert solve(test_grid) == 14

real_grid = np.genfromtxt("08_input.txt", delimiter=1, dtype="<U19")
print(solve(real_grid))  # 285


# part 2


def solve2(grid):
    antennas, bounds = _parse(grid)
    antinodes = None
    for dist in range(int((bounds[0] * bounds[1]) ** 0.5)):
        for coords in antennas.values():
            if dist == 0:
                ucoords, ucounts = np.unique(coords, axis=0, return_counts=True)
                bounded_antinodes = ucoords[ucounts > 1]
            else:
                diffs = coords[:, None, :] - coords
                unbounded_antinodes = (coords + dist * diffs)[
                    ~np.eye(coords.shape[0], dtype=bool)
                ].reshape(-1, 2)
                bounded_antinodes = unbounded_antinodes[
                    (
                        (0 <= unbounded_antinodes[:, 0])
                        & (unbounded_antinodes[:, 0] < bounds[0])
                        & (0 <= unbounded_antinodes[:, 1])
                        & (unbounded_antinodes[:, 1] < bounds[1])
                    )
                ]
            antinodes = (
                np.concatenate(
                    (antinodes, bounded_antinodes),
                    axis=0,
                )
                if antinodes is not None
                else bounded_antinodes
            )
        antinodes = np.unique(antinodes, axis=0)
    return len(antinodes)


print(solve2(test_grid))
assert solve2(test_grid) == 34

print(solve2(real_grid))  # 944
