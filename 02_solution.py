import numpy as np


# part 1


def check(report: list[int]) -> bool:
    cond_order = np.all(
        (np.argsort(report) == np.arange(len(report)))
        | (np.argsort(report[::-1]) == np.arange(len(report)))
    )
    abs_diff = np.abs(np.diff(report, n=1))
    cond_diff = np.where((1 <= abs_diff) & (abs_diff <= 3), 0, 1).sum() == 0
    return cond_order & cond_diff


with open("02_input.txt", mode="r") as fh:
    data = [list(map(int, line.split())) for line in fh.readlines()]

print(sum(check(report) for report in data))  # 564


# part 2


def check_tolerate_single_bad_level(report: list[int]) -> bool:
    if check(report):
        return True
    for skip_idx in range(len(report)):
        modified_report = [el for idx, el in enumerate(report) if idx != skip_idx]
        if check(modified_report):
            return True
    return False


print(sum(check_tolerate_single_bad_level(report) for report in data))  # 604
