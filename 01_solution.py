import numpy as np


# part 1


sorted_data = np.sort(np.genfromtxt("01_input.txt", dtype=np.int64), axis=0)
total_distance = np.sum(np.absolute(sorted_data[:, 0] - sorted_data[:, 1]))
print(total_distance)  # 2192892


# part 2


counts_map = {
    key: value for key, value in zip(*np.unique(sorted_data[:, 1], return_counts=True))
}
similarity_score = sum(
    number * counts_map.get(number, 0) for number in sorted_data[:, 0]
)
print(similarity_score)  # 22962826
