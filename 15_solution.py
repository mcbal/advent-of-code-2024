from collections import deque
from functools import partial
from operator import add, sub
import numpy as np


# part 1


BLANK, ROBOT, WALL, BOX, UP, DOWN, LEFT, RIGHT = list(range(8))
token2idx = {
    ".": BLANK,
    "@": ROBOT,
    "#": WALL,
    "O": BOX,
    "^": UP,
    "v": DOWN,
    "<": LEFT,
    ">": RIGHT,
}
idx2token = {v: k for k, v in token2idx.items()}


def flatten_list_of_list(lol):
    return [el for l in lol for el in l]


def parse(data):
    grid_lines, instructions_lines = [], []
    for line in data.split("\n"):
        if line != "":
            if line.startswith("#"):
                grid_lines.append(line)
            else:
                instructions_lines.append(line)
    return (
        np.genfromtxt(
            grid_lines,
            comments=None,
            delimiter=1,
            dtype="<U19",
        ),
        np.genfromtxt(
            flatten_list_of_list(instructions_lines),
            comments=None,
            delimiter=1,
            dtype="<U19",
        ),
    )


def encode(array):
    return np.array(
        [token2idx[char] for row in array.tolist() for char in row], dtype=np.int8
    ).reshape(array.shape)


def decode(array):
    return np.array2string(
        np.array(
            [
                idx2token[number]
                for row in (array.tolist() if isinstance(array, np.ndarray) else array)
                for number in row
            ],
            dtype="<U19",
        ).reshape(array.shape if isinstance(array, np.ndarray) else len(array)),
        max_line_width=1e3,
        separator="",
        threshold=np.inf,
        formatter={"str_kind": lambda x: x},
    ).replace(idx2token[ROBOT], f"\x1b[6;30;42m{idx2token[ROBOT]}\x1b[0m")


def _move(pos, direction):
    if direction == UP:
        return (pos[0] - 1, pos[1])
    elif direction == DOWN:
        return (pos[0] + 1, pos[1])
    elif direction == LEFT:
        return (pos[0], pos[1] - 1)
    elif direction == RIGHT:
        return (pos[0], pos[1] + 1)


def simulate(state, instructions, debug=False):
    bounds = state.shape
    for instr in instructions:
        pos = np.where(state == ROBOT)
        next_pos = _move(pos, instr)

        if state[next_pos] == WALL:
            if debug:
                print(f"Move {decode([[instr]])}:\n{decode(state)}")
            continue
        elif state[next_pos] == BLANK:
            state[pos], state[next_pos] = BLANK, ROBOT
        elif state[next_pos] == BOX:
            direction = tuple(map(sub, next_pos, pos))
            direction_array = np.array(direction)

            def _push_(_pos, _next_pos, _state):
                _next_next_pos = tuple(map(add, _next_pos, direction))
                if _state[_next_next_pos] == WALL:
                    pass
                elif _state[_next_next_pos] == BLANK:
                    _state[_next_next_pos], _state[_next_pos], _state[_pos] = (
                        _state[_next_pos],
                        _state[_pos],
                        BLANK,
                    )
                elif _state[_next_next_pos] == BOX:
                    _next_blank_pos = None
                    for i in np.arange(
                        1,
                        bounds[0 if instr in (UP, DOWN) else 1]
                        - _next_pos[0 if instr in (UP, DOWN) else 1][0],
                    ):
                        lookahead_pos = tuple(map(add, _next_pos, i * direction_array))
                        if _state[lookahead_pos] == WALL:
                            break
                        elif _state[lookahead_pos] == BLANK:
                            _next_blank_pos = lookahead_pos
                            break
                    if (
                        _next_blank_pos is not None
                    ):  # shortcut: box in `next_pos` jumps to first blank spot
                        _state[_next_blank_pos] = BOX
                        _state[_next_pos] = ROBOT
                        _state[_pos] = BLANK
                return _state

            _push_(pos, next_pos, state)

        if debug:
            print(f"Move {decode([[instr]])}:\n{decode(state)}")
    return state


def solve(parsed_data, debug=False):
    state, instructions = map(encode, parsed_data)
    final_state = simulate(state, instructions, debug=debug)
    return (np.array([100, 1]) @ np.stack(np.where(final_state == BOX))).sum()


assert (
    solve(
        parse(
            (
                "########\n"
                "#..O.O.#\n"
                "##@.O..#\n"
                "#...O..#\n"
                "#.#.O..#\n"
                "#...O..#\n"
                "#......#\n"
                "########\n"
                "\n"
                "<^^>>>vv<v>>v<<\n"
            )
        ),
        debug=False,
    )
) == 2028


assert (
    solve(
        parse(
            (
                "##########\n"
                "#..O..O.O#\n"
                "#......O.#\n"
                "#.OO..O.O#\n"
                "#..O@..O.#\n"
                "#O#..O...#\n"
                "#O..O..O.#\n"
                "#.OO.O.OO#\n"
                "#....O...#\n"
                "##########\n"
                "\n"
                "<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^\n"
                "vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v\n"
                "><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<\n"
                "<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^\n"
                "^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><\n"
                "^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^\n"
                ">^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^\n"
                "<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>\n"
                "^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>\n"
                "v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^\n"
            )
        ),
        debug=False,
    )
) == 10092

with open("15_input.txt") as fh:
    real_data = fh.read()

print(solve(parse(real_data), debug=False))  # 1412971


# part 2


BLANK, ROBOT, WALL, BOX_LEFT, BOX_RIGHT, UP, DOWN, LEFT, RIGHT = list(range(9))
token2idx = {
    ".": BLANK,
    "@": ROBOT,
    "#": WALL,
    "[": BOX_LEFT,
    "]": BOX_RIGHT,
    "^": UP,
    "v": DOWN,
    "<": LEFT,
    ">": RIGHT,
}
idx2token = {v: k for k, v in token2idx.items()}


def parse2(data):
    grid_lines, instructions_lines = [], []
    for line in data.split("\n"):
        if line != "":
            if line.startswith("#"):
                grid_lines.append(
                    line.replace("#", "##")
                    .replace("O", "[]")
                    .replace(".", "..")
                    .replace("@", "@.")
                )
            else:
                instructions_lines.append(line)
    return (
        np.genfromtxt(
            grid_lines,
            comments=None,
            delimiter=1,
            dtype="<U19",
        ),
        np.genfromtxt(
            flatten_list_of_list(instructions_lines),
            comments=None,
            delimiter=1,
            dtype="<U19",
        ),
    )


def out_of_bounds(row_max, col_max, pos):
    return (pos[0] < 0) | (pos[0] >= row_max) | (pos[1] < 0) | (pos[1] >= col_max)


def simulate2(state, instructions, debug=False):
    if debug:
        print(f"Init state:\n{decode(state)}")

    bounds = state.shape
    _out_of_bounds = partial(out_of_bounds, *bounds)

    for instr in instructions:
        pos = np.where(state == ROBOT)
        next_pos = _move(pos, instr)

        # boink: hit a wall and continue
        if state[next_pos] == WALL:
            if debug:
                print(f"Move {decode([[instr]])}:\n{decode(state)}")
            continue

        # boop: move to open position
        elif state[next_pos] == BLANK:
            state[pos], state[next_pos] = BLANK, ROBOT

        # beep: bump into boxes
        elif state[next_pos] in (BOX_LEFT, BOX_RIGHT):
            direction_array = np.array(tuple(map(sub, next_pos, pos)))

            # left/right: 1d push to wall
            if instr in (LEFT, RIGHT):
                _next_next_pos = tuple(map(add, next_pos, direction_array))
                _next_next_next_pos = tuple(map(add, next_pos, 2 * direction_array))
                if (
                    _out_of_bounds(_next_next_next_pos) | state[_next_next_next_pos]
                    == WALL
                ):
                    pass
                elif state[_next_next_next_pos] in (BLANK, BOX_LEFT, BOX_RIGHT):
                    for i in np.arange(2, bounds[1] - next_pos[1][0]):
                        lookahead_pos = tuple(map(add, next_pos, i * direction_array))
                        if state[lookahead_pos] == WALL:
                            break
                        elif state[lookahead_pos] == BLANK:
                            for j in range(i, 0, -1):
                                state[
                                    tuple(map(add, next_pos, j * direction_array))
                                ] = state[
                                    tuple(
                                        map(
                                            add,
                                            next_pos,
                                            (j - 1) * direction_array,
                                        )
                                    )
                                ]
                            state[next_pos] = ROBOT
                            state[pos] = BLANK
                            break

            # up/down: 2d push connected frontier of boxes (~bfs on binary tree)
            elif instr in (
                UP,
                DOWN,
            ):
                q = deque()
                q.append(next_pos)
                visited = {}  # use tuples for hashing (no np.ndarray)
                leaves = []

                while len(q) > 0:
                    node = q.popleft()
                    if not visited.get(tuple(zip(*node))):
                        visited[tuple(zip(*node))] = True
                        if _out_of_bounds(node) | state[node] == WALL:
                            leaves = []  # invalidate frontier
                            break
                        elif state[node] == BLANK:
                            leaves.append(node)  # build up frontier
                        elif state[node] == BOX_LEFT:
                            q.append((node[0], node[1] + 1))  # right half
                            q.append(
                                tuple(map(add, node, direction_array))
                            )  # push ahead

                        elif state[node] == BOX_RIGHT:
                            q.append((node[0], node[1] - 1))  # left half
                            q.append(
                                tuple(map(add, node, direction_array))
                            )  # push ahead

                if len(leaves) > 0:
                    reverse_visited = sorted(
                        list(k[0] for k in visited.keys()),
                        key=lambda t: t[0],
                        reverse=(instr == DOWN),
                    )
                    reverse_direction = tuple(-t[0] for t in map(sub, next_pos, pos))
                    # shift boxes starting from frontier working backwards like a caterpillar
                    for node in reverse_visited:
                        prev_node = tuple(map(add, node, reverse_direction))
                        if prev_node in reverse_visited:
                            state[node] = state[prev_node]
                        elif node[0] != pos[0]:
                            # clears other half of first pushed box (other half gets overwritten
                            # with ROBOT below). prev_node is not part of visited because it's at
                            # `pos` level next to ROBOT and we only start bfs'ing from `next_pos`
                            state[node] = BLANK
                    state[next_pos] = ROBOT
                    state[pos] = BLANK

        if debug:
            print(f"Move {decode([[instr]])}:\n{decode(state)}")

    return state


def solve2(parsed_data, debug=False):
    state, instructions = map(encode, parsed_data)
    final_state = simulate2(state, instructions, debug=debug)
    return (np.array([100, 1]) @ np.stack(np.where(final_state == BOX_LEFT))).sum()


assert (
    solve2(
        parse2(
            (
                "##########\n"
                "#..O..O.O#\n"
                "#......O.#\n"
                "#.OO..O.O#\n"
                "#..O@..O.#\n"
                "#O#..O...#\n"
                "#O..O..O.#\n"
                "#.OO.O.OO#\n"
                "#....O...#\n"
                "##########\n"
                "\n"
                "<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^\n"
                "vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v\n"
                "><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<\n"
                "<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^\n"
                "^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><\n"
                "^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^\n"
                ">^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^\n"
                "<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>\n"
                "^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>\n"
                "v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^\n"
            )
        ),
        debug=False,
    )
    == 9021
)


print(solve2(parse2(real_data), debug=False))  # 1429299
