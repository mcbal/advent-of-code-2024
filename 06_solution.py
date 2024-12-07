import os
from copy import deepcopy
from functools import partial
from io import StringIO
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm


# part 1


BLANK = 0
OBSTACLE = 1
GUARD = 2
VISITED = 3

vocab = {
    ".": BLANK,
    "#": OBSTACLE,
    "^": GUARD,
    "X": VISITED,
}


def tokenize(array):
    return np.array(
        [vocab[char] for row in array.tolist() for char in row], dtype=np.uint8
    ).reshape(array.shape)


def _move(loc, num_rotations):
    if num_rotations == 0:
        return (lambda t: (t[0] - 1, t[1]))(loc)
    elif num_rotations == 1:
        return (lambda t: (t[0], t[1] + 1))(loc)
    elif num_rotations == 2:
        return (lambda t: (t[0] + 1, t[1]))(loc)
    elif num_rotations == 3:
        return (lambda t: (t[0], t[1] - 1))(loc)


def _out_of_bounds(xmax, ymax, loc):
    return (loc[0] < 0) | (loc[0] >= xmax) | (loc[1] < 0) | (loc[1] >= ymax)


def step_until_leave(state):
    mask = np.zeros_like(state, dtype=np.uint8)
    mask[np.where(state == GUARD)] = VISITED

    out_of_bounds = partial(_out_of_bounds, *state.shape)

    def _step_until_leave(_state, _mask):
        num_rotations = 0
        while True:
            guard_loc = np.where(_state == GUARD)
            new_guard_loc = _move(guard_loc, num_rotations)
            if out_of_bounds(new_guard_loc):
                break
            if _state[new_guard_loc] == OBSTACLE:  # turn right
                _state[guard_loc] = GUARD
                num_rotations = (num_rotations + 1) % 4
            else:  # move ahead to new location
                _state[guard_loc] = BLANK
                _state[new_guard_loc] = GUARD
                _mask[new_guard_loc] = VISITED + num_rotations
        return _state, _mask

    init_state, init_mask = deepcopy(state), deepcopy(mask)
    _, mask = _step_until_leave(init_state, init_mask)

    return mask


def solve(state):
    return np.count_nonzero(step_until_leave(state))


# part 2


def step_until_leave_or_cycle_or_maxit(position, state, mask, maxit):
    init_state, init_mask = deepcopy(state), deepcopy(mask)
    init_state[position[0], position[1]] = OBSTACLE

    out_of_bounds = partial(_out_of_bounds, *state.shape)

    def _step_until_leave_or_cycle_or_maxit(_state, _mask, _maxit):
        num_rotations, it = 0, 0
        while it < _maxit:
            guard_loc = np.where(_state == GUARD)
            new_guard_loc = _move(guard_loc, num_rotations)
            if out_of_bounds(new_guard_loc):
                return False
            if _state[new_guard_loc] == OBSTACLE:  # turn right
                _state[guard_loc] = GUARD
                num_rotations = (num_rotations + 1) % 4
            else:  # move ahead to new location
                _state[guard_loc] = BLANK
                _state[new_guard_loc] = GUARD
                if _mask[new_guard_loc] == VISITED + num_rotations:
                    return True  # cycle: early exit if we already visited position in same orientation
                _mask[new_guard_loc] = VISITED + num_rotations
            it += 1
        return False

    return _step_until_leave_or_cycle_or_maxit(init_state, init_mask, maxit)


def solve2(state):
    mask = np.zeros_like(state, dtype=np.uint8)
    mask[np.where(state == GUARD)] = VISITED

    def find_limit_cycles_brute_force(state, mask):
        num_limit_cycles = 0

        candidate_positions = step_until_leave(state)
        candidate_positions[np.where(GUARD == state)] = False
        list_of_candidate_positions = list(zip(*candidate_positions.nonzero()))

        with Pool(os.cpu_count()) as p:
            num_limit_cycles = sum(
                list(
                    tqdm(
                        p.imap(
                            partial(
                                step_until_leave_or_cycle_or_maxit,
                                state=state,
                                mask=mask,
                                maxit=1e4,
                            ),
                            list_of_candidate_positions,
                        ),
                        total=len(list_of_candidate_positions),
                    )
                )
            )

        return num_limit_cycles

    return find_limit_cycles_brute_force(state, mask)


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
    )

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
    )  # 1655 (6 minutes on shitty old 8-core laptop)
