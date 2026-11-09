import torch, random
from collections import Counter, defaultdict



def group_data_fun(lengths, generator=None):

    # counter is decrease order.
    counter = Counter(lengths)
    grouped_indices = defaultdict(list)
    for idx, item in enumerate(lengths):
        grouped_indices[item].append(idx)

    grouped_indices = dict(grouped_indices)
    sorted_indices = [grouped_indices[item] for (item, _) in sorted(counter.items(), key=lambda x: x[1], reverse=True)]

    # shuffle in each group
    shuffle_sorted_indices = []
    for indice in sorted_indices:
        shuffle_idx = torch.randperm(len(indice), generator=generator).tolist()
        shuffle_sorted_indices.extend([indice[idx] for idx in shuffle_idx])

    return shuffle_sorted_indices


def split_to_even_chunks(megabatch, 
                         lengths, 
                         world_size, 
                         batch_size):

    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    
    batch_size=2, world_size=2
    [1, 2, 3, 4] -> [[1, 2], [3, 4]]
    [1, 2, 3] -> [[1, 2], [3]]
    [1, 2] -> [[1], [2]]
    [1] -> [[1], []]
    """

    chunks = [megabatch[i::world_size] for i in range(world_size)]

    pad_chunks = []
    for idx, chunk in enumerate(chunks):
        if batch_size != len(chunk):
            assert batch_size > len(chunk)
            if len(chunk) != 0:     # [[1, 2], [3]] -> [[1, 2], [3, 3]]
                chunk = chunk + [random.choice(chunk) for _ in range(batch_size - len(chunk))]
            else:
                chunk = random.choice(pad_chunks) # [[1], []] -> [[1], [1]]
        pad_chunks.append(chunk)

    return pad_chunks


def last_group_data_fun(shuffled_megabatches, lengths):

    # lengths ['1x256x256', '1x256x256', '1x256x256' ...]

    re_shuffled_megabatches = []
    for i_megabatch, megabatch in enumerate(shuffled_megabatches):

        re_megabatch = []
        for i_batch, batch in enumerate(megabatch):
            assert len(batch) != 0

            len_each_batch = [lengths[i] for i in batch]    # ['1x256x256', '1x256x256']
            idx_length_dict = dict([*zip(batch, len_each_batch)])   # {0: '1x256x256', 100: '1x256x256'}
            count_dict = Counter(len_each_batch)    # # {'1x256x256': 2} or {'1x256x256': 1, '1x768x256': 1}

            if len(count_dict) != 1:
                sorted_by_value = sorted(count_dict.items(), key=lambda item: item[1])  # {'1x256x256': 1, '1x768x256': 1}
                pick_length = sorted_by_value[-1][0]    # the highest frequency.
                candidate_batch = [idx for idx, length in idx_length_dict.items() if length == pick_length]
                random_select_batch = [random.choice(candidate_batch) for i in range(len(len_each_batch) - len(candidate_batch))]
                batch = candidate_batch + random_select_batch

            re_megabatch.append(batch)
        re_shuffled_megabatches.append(re_megabatch)
    return re_shuffled_megabatches



def get_length_grouped_indices(lengths,
                               batch_size,
                               world_size,
                               gradient_accumulation_size,
                               initial_global_step,
                               generator=None,
                               group_data=False,
                               seed=42):

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    if generator is None:
        generator = torch.Generator().manual_seed(seed) # every rank will generate a fixed order but random index.
    
    if group_data:
        indices = group_data_fun(lengths, generator)
    else:
        indices = torch.randperm(len(lengths), generator=generator).tolist()

    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i+megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    # megabatch_len = [[lengths[i] for i in megabatch] for megabatch in megabatches]

    megabatches = [split_to_even_chunks(megabatch, lengths, world_size, batch_size) for megabatch in megabatches]

    indices_mega = torch.randperm(len(megabatches), generator=generator).tolist()
    shuffled_megabatches = [megabatches[i] for i in indices_mega]

    if group_data:
        shuffled_megabatches = last_group_data_fun(shuffled_megabatches, lengths)

    shuffled_megabatches = shuffled_megabatches[initial_global_step:]

    return [batch for megabatch in shuffled_megabatches for batch in megabatch]
    
