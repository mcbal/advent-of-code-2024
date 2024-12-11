from functools import partial, reduce
from itertools import chain, product, zip_longest
from operator import add, mul
from tqdm import tqdm


# part 1


def solve(equations, operators):
    def operator_match_possible(eq):

        target, numbers = eq[0], eq[1]

        for ops in product(*[operators] * (len(numbers) - 1)):

            def _op_reduce(x, y):  # trick from https://stackoverflow.com/a/70227259
                return x(y) if callable(x) else partial(y, x)

            if target == (
                reduce(
                    _op_reduce,
                    [
                        num_op
                        for num_op in chain(*zip_longest(numbers, ops))
                        if num_op is not None
                    ],
                )
            ):
                return True

        return False

    return sum(eq[0] for eq in tqdm(equations) if operator_match_possible(eq))


def parse(data):
    def _parse(_line):
        target, numbers = _line.split(":")
        return (int(target), list(map(int, numbers.strip().split())))

    return [_parse(line) for line in data.strip().split("\n")]


test_data = (
    "190: 10 19\n"
    "3267: 81 40 27\n"
    "83: 17 5\n"
    "156: 15 6\n"
    "7290: 6 8 6 15\n"
    "161011: 16 10 13\n"
    "192: 17 8 14\n"
    "21037: 9 7 18 13\n"
    "292: 11 6 16 20\n"
)
assert solve(parse(test_data), operators=[add, mul]) == 3749

with open("07_input.txt", mode="r") as fh:
    data = fh.read()

print(solve(parse(data), operators=[add, mul]))  # 1620690235709


# part 2


assert (
    solve(parse(test_data), operators=[add, mul, lambda x, y: int(f"{x}{y}")]) == 11387
)

print(
    solve(parse(data), operators=[add, mul, lambda x, y: int(f"{x}{y}")])
)  # 145397611075341
