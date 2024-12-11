from collections import defaultdict


# part 1


def _apply_list_rules(stone: int):
    if stone == 0:
        return 1
    elif len(f"{stone}") % 2 == 0:
        digits = f"{stone}"
        return [int(digits[: len(digits) // 2]), int(digits[len(digits) // 2 :])]
    else:
        return 2024 * stone


def solve(str_repr: str, *, blinks: int):
    list_repr: list[int] = list(map(int, str_repr.split()))
    for _ in range(blinks):
        list_repr = list(map(_apply_list_rules, list_repr))
        list_repr = [
            el for l in list_repr for el in (l if isinstance(l, list) else [l])
        ]
    return len(list_repr)


assert solve("0 1 10 99 999", blinks=1) == 7
assert solve("125 17", blinks=6) == 22
assert solve("125 17", blinks=25) == 55312

with open("11_input.txt") as fh:
    data = fh.read().strip()

print(solve(data, blinks=25))  # 194782


# part 2 (address slowdown of update/insertion using list representation)

# Since we only care about the final number of stones the order of the stones actually *does
# not matter* so we can use buckets labeled by numbers to store counts. This will avoid the
# slowdown since all repeating elements will just be a count associated to a bucket with
# that number. We will use a python dictionary to implement the hash map.


def _apply_dict_rules(dict_repr):
    for k, v in dict_repr.copy().items():
        if k == 0:
            dict_repr[0] -= v
            dict_repr[1] += v
        elif len(f"{k}") % 2 == 0:
            digits = f"{k}"
            left = int(digits[: len(digits) // 2])
            right = int(digits[len(digits) // 2 :])
            dict_repr[k] -= v
            dict_repr[left] += v
            dict_repr[right] += v
        else:
            dict_repr[k] -= v
            dict_repr[2024 * k] += v
    for k, v in dict_repr.copy().items():
        if (
            v == 0
        ):  # get rid of zero counts or else they will trigger rule #2 and start spreading
            del dict_repr[k]
    return dict_repr


def solve2(str_repr: str, *, blinks: int):
    dict_repr = defaultdict(int, ((k, 1) for k in list(map(int, str_repr.split()))))
    for _ in range(blinks):
        dict_repr = _apply_dict_rules(dict_repr)
    return sum((dict_repr).values())


assert solve2("0 1 10 99 999", blinks=1) == 7
assert solve2("125 17", blinks=6) == 22
assert solve2("125 17", blinks=25) == 55312

print(solve2(data, blinks=75))  # 233007586663131
