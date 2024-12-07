import re
from functools import reduce


# part 1


def solve(mem):
    return sum(
        reduce(lambda x, y: x * y, map(int, mul_str_repr.group(1, 2)))
        for mul_str_repr in re.finditer(r"mul\((\d{1,3}),(\d{1,3})\)", mem)
    )


test_data = "xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))"
assert solve(test_data) == 161

with open("03_input.txt") as fh:
    real_data = fh.read().strip()

print(solve(real_data))  # 169021493


# part 2


def _enabled_instructions(s):
    idx, enabled = 0, True
    for _s in re.finditer(r"do\(\)|don\'t\(\)", s):
        if _s.group() == "do()":
            if not enabled:
                # state change: update idx
                idx = _s.end()
                enabled = True
        elif _s.group() == "don't()":
            if enabled:
                # state change: yield piece from last idx update until current position
                yield _s.string[idx : _s.end()]
                enabled = False
    if enabled:
        yield s[idx:]  # make sure to return final enabled piece


test_data_conditionals = (
    "xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))"
)
assert solve("".join(list(_enabled_instructions(test_data_conditionals)))) == 48

print(solve("".join(list(_enabled_instructions(real_data)))))  # 111762583
