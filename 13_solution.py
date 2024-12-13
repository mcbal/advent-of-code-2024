import re
from itertools import batched
import numpy as np


# part 1


np.set_printoptions(suppress=True, formatter={"float_kind": "{:f}".format})


def parse(data, y_shift=0):
    ols_problems = []
    for lines in batched(data.split("\n"), 4):
        X = np.zeros((2, 2))
        X[0, 0] = float(re.search(r"X\+(.*?),", lines[0]).group(1))
        X[1, 0] = float(re.search(r"Y\+(.*?)$", lines[0]).group(1))
        X[0, 1] = float(re.search(r"X\+(.*?),", lines[1]).group(1))
        X[1, 1] = float(re.search(r"Y\+(.*?)$", lines[1]).group(1))
        y = np.array(
            [
                float(re.search(r"X\=(.*?),", lines[2]).group(1)),
                float(re.search(r"Y\=(.*?)$", lines[2]).group(1)),
            ]
        )
        ols_problems.append((X.astype(np.float64), (y + y_shift).astype(np.float64)))
    return ols_problems


def solve(data, y_shift=0, atol=1e-4, check_matrices=False):
    ols = parse(data, y_shift=y_shift)

    min_tokens = []
    for X, y in ols:
        if check_matrices:
            print(
                f"rank: {np.linalg.matrix_rank(X)} :: condition number: {np.linalg.cond(X)}"
            )
        # solve linear matrix equation (avoids numerical precision issues when computing matrix inverse of X)
        beta = np.linalg.solve(X, y)
        if np.all(np.isclose(np.round(beta % 1), beta % 1, atol=atol)):
            min_tokens.extend((np.round(beta).astype(np.int64) * np.array([3, 1])))

    return sum(min_tokens)


test_data = (
    "Button A: X+94, Y+34\n"
    "Button B: X+22, Y+67\n"
    "Prize: X=8400, Y=5400\n"
    "\n"
    "Button A: X+26, Y+66\n"
    "Button B: X+67, Y+21\n"
    "Prize: X=12748, Y=12176\n"
    "\n"
    "Button A: X+17, Y+86\n"
    "Button B: X+84, Y+37\n"
    "Prize: X=7870, Y=6450\n"
    "\n"
    "Button A: X+69, Y+23\n"
    "Button B: X+27, Y+71\n"
    "Prize: X=18641, Y=10279\n"
)

assert solve(test_data) == 480

with open("13_input.txt") as fh:
    real_data = fh.read()

print(solve(real_data))  # 32067


# part 2

assert solve(test_data, y_shift=10000000000000) == 875318608908

print(solve(real_data, y_shift=10000000000000))  # 92871736253789
