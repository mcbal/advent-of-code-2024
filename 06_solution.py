import os
from functools import partial
from io import StringIO
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm


# part 1


BLANK = 0
OBSTACLE = 1
GUARD = 2

vocab = {
    ".": BLANK,
    "#": OBSTACLE,
    "^": GUARD,
}


def tokenize(array):
    return np.array(
        [vocab[char] for row in array.tolist() for char in row], dtype=np.uint8
    ).reshape(array.shape)


def _move(pos, num_rotations):
    if num_rotations == 0:
        return (pos[0] - 1, pos[1])
    elif num_rotations == 1:
        return (pos[0], pos[1] + 1)
    elif num_rotations == 2:
        return (pos[0] + 1, pos[1])
    elif num_rotations == 3:
        return (pos[0], pos[1] - 1)


def _out_of_bounds(xmax, ymax, pos):
    return (pos[0] < 0) | (pos[0] >= xmax) | (pos[1] < 0) | (pos[1] >= ymax)


def step_until_leave_or_cycle(pos, obstacles, bounds, return_cyclicity=False):
    out_of_bounds = partial(_out_of_bounds, *bounds)

    visited, num_rotations = {}, 0
    while (
        True
    ):  # will run forever if `return_cyclicity=False` and configuration leads to cycles...
        new_pos = _move(pos, num_rotations)
        if out_of_bounds(new_pos):
            break
        if (
            return_cyclicity
            and new_pos in visited
            and visited[new_pos] == num_rotations
        ):
            return True  # cycle: early exit if we already visited position in same orientation
        if new_pos in obstacles:  # turn right
            num_rotations = (num_rotations + 1) % 4
        else:  # move ahead to new location
            pos = new_pos
            visited[pos] = num_rotations
    return visited if not return_cyclicity else False


def solve(state):
    return len(
        step_until_leave_or_cycle(
            list(zip(*np.where(state == GUARD)))[0],
            list(zip(*np.where(state == OBSTACLE))),
            state.shape,
        )
    )


# part 2


def add_extra_obstacle(pos, obstacles, bounds, extra_obstacle):
    return step_until_leave_or_cycle(
        pos,
        obstacles + [extra_obstacle],
        bounds,
        return_cyclicity=True,
    )


def solve2(state):
    num_limit_cycles = 0

    pos = list(zip(*np.where(state == GUARD)))[0]
    obstacles = list(zip(*np.where(state == OBSTACLE)))
    bounds = state.shape

    possible_obstacles = step_until_leave_or_cycle(
        pos,
        list(zip(*np.where(state == OBSTACLE))),
        state.shape,
    )
    possible_obstacles.pop(pos)

    with Pool(os.cpu_count()) as p:
        num_limit_cycles = sum(
            list(
                tqdm(
                    p.imap(
                        partial(add_extra_obstacle, pos, obstacles, bounds),
                        possible_obstacles,
                    ),
                    total=len(possible_obstacles),
                )
            )
        )

    return num_limit_cycles


if __name__ == "__main__":
    test_data = (
        "....#.....\n"
        ".........#\n"
        "..........\n"
        "..#.......\n"
        ".......#..\n"
        "..........\n"
        ".#..^.....\n"
        "........#.\n"
        "#.........\n"
        "......#...\n"
    )

    assert (
        solve(
            tokenize(
                np.genfromtxt(
                    StringIO(test_data), comments=None, delimiter=1, dtype="<U19"
                ),
            )
        )
        == 41
    )

    print(
        solve(
            tokenize(
                np.genfromtxt("06_input.txt", comments=None, delimiter=1, dtype="<U19")
            )
        )
    )  # 4883

    assert (
        solve2(
            tokenize(
                (
                    np.genfromtxt(
                        StringIO(test_data), comments=None, delimiter=1, dtype="<U19"
                    )
                )
            )
        )
        == 6
    )

    print(
        solve2(
            tokenize(
                np.genfromtxt("06_input.txt", comments=None, delimiter=1, dtype="<U19")
            )
        )
    )  # 1655 (~3min on shitty old 8-core laptop)
