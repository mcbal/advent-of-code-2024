from collections import defaultdict
from itertools import pairwise


# part 1


def build_graph(rules):
    adj = defaultdict(list)
    for rule in rules:
        edge_from, edge_to = rule
        adj[edge_from].append(edge_to)
    return adj


def check(update, adj):
    for current_node, next_node in pairwise(update):
        if next_node not in adj[current_node]:
            return False
    return True


def solve(rules, updates):
    adj = build_graph(rules)
    valid_updates = []
    for update in updates:
        if check(update, adj):
            valid_updates.append(update)
    return sum(vu[len(vu) // 2] for vu in valid_updates)


def parse(data):
    rules, updates = [], []
    for line in data.split("\n"):
        if "|" in line:
            rules.append(list(map(int, line.split("|"))))
        elif "," in line:
            updates.append(list(map(int, line.split(","))))
    return rules, updates


test_data = (
    "47|53\n"
    "97|13\n"
    "97|61\n"
    "97|47\n"
    "75|29\n"
    "61|13\n"
    "75|53\n"
    "29|13\n"
    "97|29\n"
    "53|29\n"
    "61|53\n"
    "97|53\n"
    "61|29\n"
    "47|13\n"
    "75|47\n"
    "97|75\n"
    "47|61\n"
    "75|61\n"
    "47|29\n"
    "75|13\n"
    "53|13\n\n"
    "75,47,61,53,29\n"
    "97,61,53,29,13\n"
    "75,29,13\n"
    "75,97,47,61,53\n"
    "61,13,29\n"
    "97,13,75,29,47"
)
assert solve(*parse(test_data)) == 143

with open("05_input.txt") as fh:
    real_data = fh.read()

print(solve(*parse(real_data)))  # 5329


# part 2


def correct(update, adj):
    """Every element only appears once so we can just brute force swap until checks pass."""
    while not check(update, adj):
        for idx, (current_node, next_node) in enumerate(pairwise(update)):
            if next_node not in adj[current_node]:
                update[idx], update[idx + 1] = update[idx + 1], update[idx]
    return update


def solve2(rules, updates):
    adj = build_graph(rules)
    invalid_updates = []
    for update in updates:
        if not check(update, adj):
            invalid_updates.append(correct(update, adj))
    return sum(vu[len(vu) // 2] for vu in invalid_updates)


assert solve2(*parse(test_data)) == 123

print(solve2(*parse(real_data)))  # 5833
