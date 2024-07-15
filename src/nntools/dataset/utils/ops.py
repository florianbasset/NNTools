import copy
import numpy as np
from torch import default_generator, randperm
from nntools.dataset.viewer import Viewer


def random_split(dataset, lengths, generator=default_generator):
    if sum(lengths) == 1:
        lengths = [int(length * len(dataset)) for length in lengths[:-1]]
        lengths.append(len(dataset) - sum(lengths))  # To prevent rounding error

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    datasets = []
    for split, (offset, length) in enumerate(zip(np.cumsum(lengths), lengths)):
        d = copy.deepcopy(dataset)
        # We need to explicit call the attrs post init callback since deepcopy does not call it
        # d.__attrs_post_init__()
        # We also need to explicitely copy the composer
        d.img_filepath = copy.deepcopy(dataset.img_filepath)
        d.gts = copy.deepcopy(dataset.gts)
        d.composer = copy.deepcopy(dataset.composer)
        d.ignore_keys = copy.deepcopy(dataset.ignore_keys)
        d.viewer = Viewer(d)
        indx = indices[offset - length : offset]
        d.subset(indx)
        d.id = d.id + f"_split_{split}"
        if dataset.use_cache:
            d.cache = copy.deepcopy(dataset.cache)
            d.cache.d = d

        datasets.append(d)
    return tuple(datasets)


def split(dataset, indices):
    datasets = []
    for split, indx in enumerate(indices):
        d = copy.deepcopy(dataset)
        d.img_filepath = copy.deepcopy(dataset.img_filepath)
        d.gts = copy.deepcopy(dataset.gts)
        d.composer = copy.deepcopy(dataset.composer)
        d.ignore_keys = copy.deepcopy(dataset.ignore_keys)
        d.viewer = Viewer(d)
        d.subset(indx)
        d.id = d.id + f"_split_{split}"
        if dataset.use_cache:
            d.cache = copy.deepcopy(dataset.cache)
            d.cache.d = d

        datasets.append(d)
    return tuple(datasets)
