import heapq
import numpy as np


# part 1


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


RIGHT, UP, LEFT, DOWN, BLANK, WALL, START, END, TILE = list(range(9))
token2idx = {
    ">": RIGHT,
    "^": UP,
    "<": LEFT,
    "v": DOWN,
    ".": BLANK,
    "#": WALL,
    "S": START,
    "E": END,
    "O": TILE,
}
idx2token = {v: k for k, v in token2idx.items()}


def parse(data):
    grid_lines = []
    for line in data.split("\n"):
        if line != "":
            if line.startswith("#"):
                grid_lines.append(line)
    return np.genfromtxt(
        grid_lines,
        comments=None,
        delimiter=1,
        dtype="<U19",
    )


def encode(array):
    return np.array(
        [token2idx[char] for row in array.tolist() for char in row], dtype=np.int8
    ).reshape(array.shape)


def decode(array):
    return np.array2string(
        np.array(
            [idx2token[number] for row in array.tolist() for number in row],
            dtype="<U19",
        ).reshape(array.shape),
        max_line_width=1e3,
        separator="",
        threshold=np.inf,
        formatter={"str_kind": lambda x: x},
    )


def visualize_path(state, came_from, start_node, end_node):
    current = end_node
    state_copy = state.copy()
    while current != came_from[start_node]:
        state_copy[current[0][0], current[0][1]] = current[1]
        current = came_from[current]
    print(decode(state_copy))


def move(pos, direction):
    if direction == UP:
        return (pos[0] - 1, pos[1])
    elif direction == DOWN:
        return (pos[0] + 1, pos[1])
    elif direction == LEFT:
        return (pos[0], pos[1] - 1)
    elif direction == RIGHT:
        return (pos[0], pos[1] + 1)


def next_legal_moves(pos_dir, state):
    pos, orientation = pos_dir
    next_legal_moves = []
    for proposal_orientation in (
        orientation,
        (orientation + 1) % 4,
        (orientation - 1) % 4,
    ):
        proposal_pos = move(pos, proposal_orientation)
        if state[proposal_pos] != WALL:
            next_legal_moves.append((proposal_pos, proposal_orientation))
    return next_legal_moves


def cost(current, next):
    return 1 + 1000 * (current[1] != next[1])


def find_lowest_score_path(state, start=START, end=END, penalty_path=None, debug=False):
    start, end = tuple(map(int, list(zip(*np.where(state == START)))[0])), tuple(
        map(int, list(zip(*np.where(state == END)))[0])
    )
    start_node = (start, RIGHT)

    frontier = PriorityQueue()
    frontier.put(start_node, 0)

    came_from = {}
    came_from[start_node] = None

    cost_so_far = {}
    cost_so_far[start_node] = 0

    while not frontier.empty():
        node = frontier.get()

        if node[0] == end:
            end_node = node
            break

        for next_node in next_legal_moves(node, state):
            new_cost = cost_so_far[node] + cost(node, next_node)
            if penalty_path is not None and next_node[0] in penalty_path:
                new_cost += 1  # increase cost for penalty path so priority queue favors other equivalent path
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                frontier.put(next_node, new_cost)
                came_from[next_node] = node

    if debug:
        visualize_path(state, came_from, start_node, end_node)

    if penalty_path is not None:
        # `cost_so_far` has become invalid since we used it to penalize nodes, so we need
        # to undo potential penalty costs. easiest to just recompute cost along reverse path
        # (number of rotations / steps is the same forward and backward)
        node = end_node
        penalty_corrected_cost = 0
        while node != came_from[(start, RIGHT)]:
            next_node = came_from[node]
            if next_node != came_from[start_node]:
                penalty_corrected_cost += cost(node, next_node)
            node = next_node

        return penalty_corrected_cost, came_from
    else:
        return cost_so_far[end_node], came_from


def solve(parsed_data, debug=False):
    return find_lowest_score_path(encode(parsed_data), debug=debug)[0]


assert (
    solve(
        parse(
            (
                "###############\n"
                "#.......#....E#\n"
                "#.#.###.#.###.#\n"
                "#.....#.#...#.#\n"
                "#.###.#####.#.#\n"
                "#.#.#.......#.#\n"
                "#.#.#####.###.#\n"
                "#...........#.#\n"
                "###.#.#####.#.#\n"
                "#...#.....#.#.#\n"
                "#.#.#.###.#.#.#\n"
                "#.....#...#.#.#\n"
                "#.###.#.#.#.#.#\n"
                "#S..#.....#...#\n"
                "###############\n"
            )
        ),
        debug=False,
    )
) == 7036


assert (
    solve(
        parse(
            (
                "#################\n"
                "#...#...#...#..E#\n"
                "#.#.#.#.#.#.#.#.#\n"
                "#.#.#.#...#...#.#\n"
                "#.#.#.#.###.#.#.#\n"
                "#...#.#.#.....#.#\n"
                "#.#.#.#.#.#####.#\n"
                "#.#...#.#.#.....#\n"
                "#.#.#####.#.###.#\n"
                "#.#.#.......#...#\n"
                "#.#.###.#####.###\n"
                "#.#.#...#.....#.#\n"
                "#.#.#.#####.###.#\n"
                "#.#.#.........#.#\n"
                "#.#.#.#########.#\n"
                "#S#.............#\n"
                "#################\n"
            )
        ),
        debug=False,
    )
) == 11048

with open("16_input.txt") as fh:
    real_data = fh.read()

print(solve(parse(real_data)))  # 122492


# part 2


def positions_in_path(end, came_from, start):
    node = next(n for n in came_from if n[0] == end)
    pos = []
    while node != came_from[start]:
        pos.append(node[0])
        node = came_from[node]
    return pos


def solve2(parsed_data):
    state = encode(parsed_data)
    target_score, came_from = find_lowest_score_path(state)

    start, end = tuple(map(int, list(zip(*np.where(state == START)))[0])), tuple(
        map(int, list(zip(*np.where(state == END)))[0])
    )

    best_tiles = set()
    best_tiles.update(positions_in_path(end, came_from, (start, RIGHT)))

    prev_len = len(best_tiles)
    while True:
        # add penalty for node positions along known best tiles which forces path finding to explore other paths
        penalty_corrected_score, came_from = find_lowest_score_path(
            state, penalty_path=best_tiles
        )
        if penalty_corrected_score == target_score:
            # found equivalent path, which probably shares most tiles so only keep unique ones in set
            best_tiles.update(positions_in_path(end, came_from, (start, RIGHT)))
            if len(best_tiles) == prev_len:
                # no more tiles to detect using penalty method
                break
        else:
            break
        prev_len = len(best_tiles)

    return len(best_tiles)


assert (
    solve2(
        parse(
            (
                "###############\n"
                "#.......#....E#\n"
                "#.#.###.#.###.#\n"
                "#.....#.#...#.#\n"
                "#.###.#####.#.#\n"
                "#.#.#.......#.#\n"
                "#.#.#####.###.#\n"
                "#...........#.#\n"
                "###.#.#####.#.#\n"
                "#...#.....#.#.#\n"
                "#.#.#.###.#.#.#\n"
                "#.....#...#.#.#\n"
                "#.###.#.#.#.#.#\n"
                "#S..#.....#...#\n"
                "###############\n"
            )
        )
    )
) == 45


assert (
    solve2(
        parse(
            (
                "#################\n"
                "#...#...#...#..E#\n"
                "#.#.#.#.#.#.#.#.#\n"
                "#.#.#.#...#...#.#\n"
                "#.#.#.#.###.#.#.#\n"
                "#...#.#.#.....#.#\n"
                "#.#.#.#.#.#####.#\n"
                "#.#...#.#.#.....#\n"
                "#.#.#####.#.###.#\n"
                "#.#.#.......#...#\n"
                "#.#.###.#####.###\n"
                "#.#.#...#.....#.#\n"
                "#.#.#.#####.###.#\n"
                "#.#.#.........#.#\n"
                "#.#.#.#########.#\n"
                "#S#.............#\n"
                "#################\n"
            )
        )
    )
) == 64


print(solve2(parse(real_data)))  # 520
