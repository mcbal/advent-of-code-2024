from copy import deepcopy
from itertools import chain, zip_longest
from tqdm import tqdm


def _map2repr(disk_map):
    return [
        tuple(
            map(
                lambda x: int(x) if x is not None else 0,
                (file_id_number, num_file_block, num_free_block),
            )
        )
        for file_id_number, (num_file_block, num_free_block) in enumerate(
            zip_longest(disk_map[::2], disk_map[1::2])
        )
    ]


def _repr2str(t):
    """Useful for debugging."""
    return "".join(
        [
            f"({file_id_number})" * num_file_block + "." * num_free_block
            for (file_id_number, num_file_block, num_free_block) in t
        ]
    )


def fs_checksum(disk_map):
    list_repr = _map2repr(disk_map)  # (file_id_number, num_file_block, num_free_block)

    last_file = max(list_repr, key=lambda t: t[0])  # largest file id_number
    file_idx = (
        last_file[0],
        last_file[1],
    )  # can ignore spacing for this part since it's all pushed to the right

    def _fetch_file_blocks(file_idx, batch_size, min_idx):
        batch = []
        fetched_file_blocks = 0
        while (remainder := (batch_size - fetched_file_blocks)) > 0:
            if file_idx[0] == min_idx:
                break
            full_capacity = list_repr[file_idx[0]][1]
            next_num_file_block = min(file_idx[1], full_capacity)
            if (remainder - next_num_file_block) < 0:
                fetched_file_blocks += remainder
                batch.append((file_idx[0], remainder, 0))
                file_idx = (file_idx[0], next_num_file_block - remainder)
            else:
                fetched_file_blocks += next_num_file_block
                batch.append((file_idx[0], next_num_file_block, 0))
                next_full_capacity = list_repr[file_idx[0] - 1][1]
                file_idx = (file_idx[0] - 1, next_full_capacity)
        return file_idx, batch

    fragmented_list_repr = []
    for file_id_number, num_file_block, num_free_block in list_repr:
        if file_id_number + 1 > file_idx[0]:
            file_idx, batch = _fetch_file_blocks(
                file_idx, file_idx[1], file_id_number - 1
            )
            fragmented_list_repr.extend(batch)
            break

        fragmented_list_repr.append((file_id_number, num_file_block, 0))
        file_idx, batch = _fetch_file_blocks(file_idx, num_free_block, file_id_number)
        fragmented_list_repr.extend(batch)

    return sum(
        pos * tt
        for pos, tt in enumerate(
            chain.from_iterable(([t[0]] * t[1] for t in fragmented_list_repr))
        )
    )


assert fs_checksum("2333133121414131402") == 1928

with open("09_input.txt") as fh:
    real_disk_map = fh.read().strip()

print(fs_checksum(real_disk_map))  # 6607511583593


# part 2


def fs_checksum_part_2(disk_map):
    list_repr = _map2repr(disk_map)  # (file_id_number, num_file_block, num_free_block)

    for source in tqdm(deepcopy(list_repr)[::-1]):  # backward pass until end
        for dest_idx, dest in enumerate(list_repr):  # forward pass until break
            if source[1] <= dest[2]:
                # find source in list_repr by matching on (file_id_number, num_file_block) since
                # spacing (num_free_block) can have changed as we update list_repr in place
                matching_source_idx, matching_source = next(
                    (it, t) for (it, t) in enumerate(list_repr) if t[:2] == source[:2]
                )
                # now check if free blocks of dest is strictly to the left of matching source
                if dest_idx < matching_source_idx:
                    # remove source that needs to be moved left
                    list_repr.pop(matching_source_idx)
                    # update file before removed source file to absorb length and spacing
                    list_repr[matching_source_idx - 1] = (
                        list_repr[matching_source_idx - 1][0],
                        list_repr[matching_source_idx - 1][1],
                        list_repr[matching_source_idx - 1][2]
                        + matching_source[1]
                        + matching_source[2],
                    )
                    # add removed source file at target destination in correct position with correct spacing
                    spacing = (
                        dest[2] - matching_source[1]
                        if (matching_source_idx - dest_idx) > 1
                        else dest[2] + matching_source[2]
                    )  # this was tricky
                    list_repr.insert(
                        dest_idx + 1,
                        (
                            matching_source[0],
                            matching_source[1],
                            spacing,
                        ),
                    )
                    # remove spacing of target destination since moved file tuple at
                    # position (dest_idx + 1) now contains the appropriate spacing
                    list_repr[dest_idx] = (dest[0], dest[1], 0)

                break  # an important break statement

    pos, checksum = 0, 0
    for tt in list_repr:
        for _ in range(tt[1]):
            checksum += pos * tt[0]
            pos += 1
        pos += tt[2]  # move position ahead with spacing length

    return checksum


assert fs_checksum_part_2("2333133121414131402") == 2858

assert fs_checksum_part_2("1010101010101010101010") == 385
assert fs_checksum_part_2("354631466260") == 1325
assert fs_checksum_part_2("111000000000000000001") == 12
assert fs_checksum_part_2("20101011331210411119") == 826  # this was tricky

print(fs_checksum_part_2(real_disk_map))  # 6636608781232
